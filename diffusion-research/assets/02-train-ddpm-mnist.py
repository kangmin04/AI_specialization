"""
02장 실습 3~4: 작은 U-Net + MNIST로 DDPM 학습, 그리고 샘플링 과정 시각화.
채널 수를 작게 잡아 CPU/MPS(Apple Silicon)에서도 몇 분 안에 끝나게 구성.

실행:
    python3 02-train-ddpm-mnist.py
MNIST는 torchvision이 자동으로 다운로드한다 (최초 1회, 인터넷 필요).
GPU가 있으면(MPS 또는 CUDA) 자동으로 사용한다.
"""
import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib
import matplotlib.pyplot as plt

try:
    matplotlib.rcParams["font.family"] = "Apple SD Gothic Neo"
    matplotlib.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(OUT_DIR, ".mnist_data")

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("device:", device)

torch.manual_seed(0)

T = 1000
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


# ---------------- Sinusoidal time embedding (5장 참고) ----------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) 정수 timestep
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class ResBlock(nn.Module):
    """Conv block + time embedding injection (채널별 bias로 broadcast)."""
    def __init__(self, in_ch, out_ch, t_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.time_proj = nn.Linear(t_emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]  # 5장의 핵심: 시간 정보 주입
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    """
    아주 작은 U-Net. 28x28 -> 14x14 -> 7x7 -> 14x14 -> 28x28.
    구조 자체는 computer-vision/03-object-detection.md의 U-Net과 동일하고,
    달라지는 부분은 각 ResBlock에 시간 임베딩을 주입한다는 점뿐이다.
    """
    def __init__(self, base_ch=32, t_emb_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(t_emb_dim),
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )

        self.in_conv = nn.Conv2d(1, base_ch, 3, padding=1)

        # Encoder
        self.enc1 = ResBlock(base_ch, base_ch, t_emb_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)  # 28->14

        self.enc2 = ResBlock(base_ch, base_ch * 2, t_emb_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)  # 14->7

        # Bottleneck
        self.bottleneck = ResBlock(base_ch * 2, base_ch * 2, t_emb_dim)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)  # 7->14
        self.dec2 = ResBlock(base_ch * 2 + base_ch * 2, base_ch, t_emb_dim)  # skip concat

        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1)  # 14->28
        self.dec1 = ResBlock(base_ch + base_ch, base_ch, t_emb_dim)  # skip concat

        self.out_conv = nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)

        x0 = self.in_conv(x)
        e1 = self.enc1(x0, t_emb)          # (B, C, 28, 28)
        d1 = self.down1(e1)                # (B, C, 14, 14)

        e2 = self.enc2(d1, t_emb)          # (B, 2C, 14, 14)
        d2 = self.down2(e2)                # (B, 2C, 7, 7)

        b = self.bottleneck(d2, t_emb)     # (B, 2C, 7, 7)

        u2 = self.up2(b)                   # (B, 2C, 14, 14)
        u2 = torch.cat([u2, e2], dim=1)    # skip connection
        u2 = self.dec2(u2, t_emb)          # (B, C, 14, 14)

        u1 = self.up1(u2)                  # (B, C, 28, 28)
        u1 = torch.cat([u1, e1], dim=1)    # skip connection
        u1 = self.dec1(u1, t_emb)          # (B, C, 28, 28)

        return self.out_conv(u1)           # (B, 1, 28, 28) 예측된 noise


def q_sample(x0, t, noise):
    """1.2절 닫힌 형태 forward process."""
    ab = alpha_bars[t][:, None, None, None]
    return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise


# ---------------- 데이터 ----------------
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: x * 2 - 1),  # [-1, 1]
])
train_ds = torchvision.datasets.MNIST(
    root=DATA_DIR, train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)

model = TinyUNet(base_ch=32, t_emb_dim=128).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"model params: {n_params:,}")

opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

EPOCHS = 3  # MNIST 검증 목적이므로 짧게. L_simple이 빠르게 내려가는지만 확인.
step = 0
t0 = time.time()
loss_history = []
for epoch in range(EPOCHS):
    for x0, _ in train_loader:
        x0 = x0.to(device)
        b = x0.shape[0]
        t = torch.randint(0, T, (b,), device=device)
        noise = torch.randn_like(x0)
        xt = q_sample(x0, t, noise)

        pred_noise = model(xt, t)
        loss = F.mse_loss(pred_noise, noise)  # 3.2절: L_simple

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f"epoch {epoch} step {step} loss {loss.item():.4f} elapsed {time.time()-t0:.1f}s")
        loss_history.append(loss.item())
        step += 1

print(f"training done in {time.time()-t0:.1f}s, total steps {step}")

# ---------------- Loss curve ----------------
plt.figure(figsize=(6, 4))
plt.plot(loss_history, alpha=0.4, label="raw step loss")
window = 50
if len(loss_history) >= window:
    ma = [
        sum(loss_history[max(0, i - window):i + 1]) / len(loss_history[max(0, i - window):i + 1])
        for i in range(len(loss_history))
    ]
    plt.plot(ma, color="#d62728", label=f"{window}-step moving avg")
plt.xlabel("training step")
plt.ylabel("L_simple (MSE)")
plt.title("DDPM training loss on MNIST (tiny U-Net)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02-training-loss.png"), dpi=150)
plt.close()
print("saved training loss curve")


# ---------------- 샘플링 (4장 알고리즘 그대로) ----------------
@torch.no_grad()
def sample(model, n_samples=8, record_every=100):
    model.eval()
    x = torch.randn(n_samples, 1, 28, 28, device=device)
    snapshots = {}
    for t_step in reversed(range(T)):
        t_batch = torch.full((n_samples,), t_step, device=device, dtype=torch.long)
        pred_noise = model(x, t_batch)
        alpha_t = alphas[t_step]
        alpha_bar_t = alpha_bars[t_step]
        beta_t = betas[t_step]

        coef = beta_t / torch.sqrt(1 - alpha_bar_t)
        mean = (1 / torch.sqrt(alpha_t)) * (x - coef * pred_noise)  # 2.2절 mu_theta

        if t_step > 0:
            z = torch.randn_like(x)
            sigma_t = torch.sqrt(beta_t)
            x = mean + sigma_t * z
        else:
            x = mean

        if t_step % record_every == 0 or t_step == T - 1:
            snapshots[t_step] = x.clone().cpu()
    model.train()
    return x.cpu(), snapshots


n_samples = 8
final_x, snapshots = sample(model, n_samples=n_samples, record_every=100)

# (c) 최종 생성 샘플 grid
fig, axes = plt.subplots(1, n_samples, figsize=(2 * n_samples, 2.2))
for i in range(n_samples):
    img = (final_x[i, 0].clamp(-1, 1) + 1) / 2
    axes[i].imshow(img.numpy(), cmap="gray", vmin=0, vmax=1)
    axes[i].axis("off")
plt.suptitle("DDPM으로 생성한 MNIST 샘플 (3 epoch 학습, 실제 실행 결과)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02-generated-samples.png"), dpi=150)
plt.close()
print("saved generated samples")

# (d) 샘플링 중간 과정 시각화 (하나의 샘플이 노이즈->숫자로 변하는 과정)
ts_sorted = sorted(snapshots.keys(), reverse=True)
fig, axes = plt.subplots(1, len(ts_sorted), figsize=(2 * len(ts_sorted), 2.4))
sample_idx = 0
for ax, t_step in zip(axes, ts_sorted):
    img = (snapshots[t_step][sample_idx, 0].clamp(-1, 1) + 1) / 2
    ax.imshow(img.numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"t={t_step}")
    ax.axis("off")
plt.suptitle("Reverse process: 순수 노이즈에서 숫자가 나타나는 과정 (실제 실행 결과)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02-sampling-process-grid.png"), dpi=150)
plt.close()
print("saved sampling process grid")

print("ALL DONE")
