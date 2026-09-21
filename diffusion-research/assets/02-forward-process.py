"""
02장 실습 2: Forward process 시각화.
학습 없이 순수 수식(닫힌 형태 x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps)만으로
이미지에 노이즈가 점점 섞이는 과정을 확인한다.

실행: 이 스크립트가 있는 assets/ 디렉토리에서, 혹은 저장소 어디서든
    python3 02-forward-process.py
MNIST는 torchvision이 자동으로 다운로드한다 (최초 1회, 인터넷 필요).
"""
import os
import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt

# 한글 폰트 (macOS 기준). 없는 환경이면 이 줄을 지워도 그래프 자체는 정상 생성됨.
try:
    matplotlib.rcParams["font.family"] = "Apple SD Gothic Neo"
    matplotlib.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(OUT_DIR, ".mnist_data")

torch.manual_seed(0)

T = 1000
beta_start, beta_end = 1e-4, 0.02
betas = torch.linspace(beta_start, beta_end, T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

# ---- (a) noise schedule 그래프 ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(betas.numpy(), color="#d62728")
axes[0].set_title(r"$\beta_t$ (linear schedule)")
axes[0].set_xlabel("timestep t")
axes[0].set_ylabel(r"$\beta_t$")
axes[0].grid(alpha=0.3)

axes[1].plot(alpha_bars.numpy(), color="#1f77b4")
axes[1].set_title(r"$\bar\alpha_t = \prod_{s=1}^t (1-\beta_s)$")
axes[1].set_xlabel("timestep t")
axes[1].set_ylabel(r"$\bar\alpha_t$")
axes[1].grid(alpha=0.3)

plt.tight_layout()
schedule_path = os.path.join(OUT_DIR, "02-noise-schedule.png")
plt.savefig(schedule_path, dpi=150)
plt.close()
print("saved", schedule_path)

# ---- (b) forward process grid: 실제 MNIST 이미지에 t를 늘려가며 노이즈 주입 ----
ds = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True)
img, label = ds[7]  # 아무 숫자나 하나 고정 (실제로는 '3')
x0 = torchvision.transforms.functional.to_tensor(img) * 2 - 1  # [-1, 1] 정규화, shape (1,28,28)

ts_to_show = [0, 50, 100, 200, 300, 500, 700, 1000]
fig, axes = plt.subplots(1, len(ts_to_show), figsize=(2 * len(ts_to_show), 2.4))

for ax, t in zip(axes, ts_to_show):
    if t == 0:
        xt = x0
    else:
        idx = t - 1  # alpha_bars는 0-indexed로 t=1..T 저장
        ab = alpha_bars[idx]
        eps = torch.randn_like(x0)
        xt = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps
    img_show = (xt.squeeze(0).clamp(-1, 1) + 1) / 2  # 다시 [0,1]로
    ax.imshow(img_show.numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"t={t}")
    ax.axis("off")

plt.suptitle(f"Forward process: MNIST digit '{label}' 에 t가 커질수록 노이즈가 섞이는 과정 (실제 실행 결과)")
plt.tight_layout()
forward_path = os.path.join(OUT_DIR, "02-forward-process-grid.png")
plt.savefig(forward_path, dpi=150)
plt.close()
print("saved", forward_path)

# alpha_bar 값들도 출력해서 문서에 실제 숫자 인용 가능하게
for t in ts_to_show:
    if t == 0:
        print(f"t={t}: alpha_bar=1.0 (원본)")
    else:
        print(
            f"t={t}: alpha_bar={alpha_bars[t-1].item():.6f}, "
            f"sqrt(alpha_bar)={alpha_bars[t-1].sqrt().item():.4f}, "
            f"sqrt(1-alpha_bar)={(1-alpha_bars[t-1]).sqrt().item():.4f}"
        )
