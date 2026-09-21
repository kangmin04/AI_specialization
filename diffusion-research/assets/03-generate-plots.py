"""
03-score-based-sde.md 용 시각자료 생성 + 실습 스크립트.
외부 데이터셋 다운로드 없이 sklearn.datasets.make_moons(로컬 생성)만 사용.
신경망은 PyTorch 없이 numpy로 2-hidden-layer MLP를 직접 구현(forward/backward 수동 미분).

실행: python3 03-generate-plots.py  (같은 폴더에 PNG 2개 생성)

내용:
  1. 2D moons 데이터에 여러 noise scale(sigma)을 섞어 Denoising Score Matching으로
     score network s_theta(x, sigma)를 학습 (NCSN, Song & Ermon 2019 스타일).
  2. 학습된 score field를 quiver plot으로 시각화.
  3. Annealed Langevin dynamics로 순수 노이즈 -> moons 모양 샘플링 과정을 시각화.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

plt.rcParams["font.size"] = 11
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 0. 데이터: make_moons를 표준화(평균 0, 분산 1 근처)해서 사용
# ---------------------------------------------------------------------------
X_raw, _ = make_moons(n_samples=3000, noise=0.06, random_state=0)
X_raw = X_raw.astype(np.float64)
data_mean = X_raw.mean(axis=0)
X_centered = X_raw - data_mean
data_scale = X_centered.std()  # 가로/세로 비율을 유지하기 위해 스칼라 하나로 나눔
X = X_centered / data_scale
N_DATA = X.shape[0]

# ---------------------------------------------------------------------------
# 1. Noise scale 스케줄 (NCSN 스타일 geometric sequence)
#    가장 큰 sigma는 데이터 전체를 거의 뒤덮을 정도로 크게, 가장 작은 sigma는
#    데이터 분산보다 확실히 작게 잡는다.
# ---------------------------------------------------------------------------
SIGMA_MAX, SIGMA_MIN, L = 1.2, 0.05, 10
sigmas = np.exp(np.linspace(np.log(SIGMA_MAX), np.log(SIGMA_MIN), L))

# ---------------------------------------------------------------------------
# 2. Score network: 입력 [x1, x2, log(sigma)] -> tanh(128) -> tanh(128) -> 2
#    numpy로 forward/backward를 직접 구현한다 (자동미분 라이브러리 없이).
# ---------------------------------------------------------------------------
D_IN, H, D_OUT = 3, 128, 2


def init_params(rng):
    p = {}
    p["W1"] = rng.normal(0, 1, (D_IN, H)) * np.sqrt(2.0 / D_IN)
    p["b1"] = np.zeros(H)
    p["W2"] = rng.normal(0, 1, (H, H)) * np.sqrt(2.0 / H)
    p["b2"] = np.zeros(H)
    p["W3"] = rng.normal(0, 1, (H, D_OUT)) * np.sqrt(1.0 / H)
    p["b3"] = np.zeros(D_OUT)
    return p


def forward(p, x_in):
    z1 = x_in @ p["W1"] + p["b1"]
    a1 = np.tanh(z1)
    z2 = a1 @ p["W2"] + p["b2"]
    a2 = np.tanh(z2)
    out = a2 @ p["W3"] + p["b3"]
    return out, (x_in, a1, a2)


def backward(p, cache, dout):
    x_in, a1, a2 = cache
    grads = {}
    grads["W3"] = a2.T @ dout
    grads["b3"] = dout.sum(0)
    da2 = dout @ p["W3"].T
    dz2 = da2 * (1 - a2 ** 2)
    grads["W2"] = a1.T @ dz2
    grads["b2"] = dz2.sum(0)
    da1 = dz2 @ p["W2"].T
    dz1 = da1 * (1 - a1 ** 2)
    grads["W1"] = x_in.T @ dz1
    grads["b1"] = dz1.sum(0)
    return grads


def score_fn(p, x, sigma):
    """s_theta(x, sigma). x: (N,2) array, sigma: scalar."""
    log_sigma_col = np.full((x.shape[0], 1), np.log(sigma))
    x_in = np.concatenate([x, log_sigma_col], axis=1)
    out, _ = forward(p, x_in)
    return out


# ---------------------------------------------------------------------------
# 3. 학습: Denoising Score Matching, NCSN 가중치 lambda(sigma) = sigma^2
#    loss = mean[ sigma^2 * || s_theta(x_tilde, sigma) - (-eps/sigma) ||^2 ]
#         = mean[ || sigma * s_theta(x_tilde, sigma) + eps ||^2 ]   (동치, 더 안정적)
#    Adam optimizer도 numpy로 직접 구현.
# ---------------------------------------------------------------------------
params = init_params(rng)
m = {k: np.zeros_like(v) for k, v in params.items()}
v = {k: np.zeros_like(v) for k, v in params.items()}
beta1, beta2, eps_adam, lr = 0.9, 0.999, 1e-8, 2e-3

N_ITERS = 6000
BATCH = 256
loss_history = []

for it in range(1, N_ITERS + 1):
    idx = rng.integers(0, N_DATA, size=BATCH)
    x0 = X[idx]
    sigma_idx = rng.integers(0, L, size=BATCH)
    sigma_batch = sigmas[sigma_idx]  # (BATCH,)
    eps_noise = rng.normal(size=(BATCH, 2))
    x_tilde = x0 + sigma_batch[:, None] * eps_noise

    log_sigma_col = np.log(sigma_batch)[:, None]
    x_in = np.concatenate([x_tilde, log_sigma_col], axis=1)
    out, cache = forward(params, x_in)  # s_theta(x_tilde, sigma)

    # loss_i = || sigma_i * out_i + eps_i ||^2 (합, batch 평균)
    residual = sigma_batch[:, None] * out + eps_noise  # (BATCH, 2)
    loss = np.mean(np.sum(residual ** 2, axis=1))
    loss_history.append(loss)

    # d(loss)/d(out) = 2 * sigma * residual / BATCH
    dout = 2.0 * sigma_batch[:, None] * residual / BATCH

    grads = backward(params, cache, dout)

    t = it
    for k in params:
        m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
        v[k] = beta2 * v[k] + (1 - beta2) * (grads[k] ** 2)
        m_hat = m[k] / (1 - beta1 ** t)
        v_hat = v[k] / (1 - beta2 ** t)
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)

    if it % 1000 == 0 or it == 1:
        print(f"iter {it:5d}  loss {loss:.4f}")

print(f"최종 loss (최근 200 iter 평균): {np.mean(loss_history[-200:]):.4f}")

# ---------------------------------------------------------------------------
# 4. (a) 학습된 score field quiver plot
#    시각적으로 화살표가 잘 보이는 중간 크기 sigma를 사용 (너무 작으면 데이터
#    바로 위에서만 화살표가 크고 나머지 영역은 거의 0에 가까워 안 보인다).
# ---------------------------------------------------------------------------
QUIVER_SIGMA = sigmas[np.argmin(np.abs(sigmas - 0.3))]

grid_n = 22
lim = 2.2
gx, gy = np.meshgrid(np.linspace(-lim, lim, grid_n), np.linspace(-lim, lim, grid_n))
grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
grid_scores = score_fn(params, grid_pts, QUIVER_SIGMA)

# 화살표 길이는 시각화를 위해 정규화(방향만 강조), 색은 원래 크기(magnitude)로 표현
mag = np.linalg.norm(grid_scores, axis=1) + 1e-8
unit = grid_scores / mag[:, None]

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(X[:, 0], X[:, 1], s=5, c="lightgray", label="학습 데이터 (moons)", zorder=1)
q = ax.quiver(
    grid_pts[:, 0], grid_pts[:, 1], unit[:, 0], unit[:, 1], mag,
    cmap="viridis", scale=25, width=0.005, zorder=2,
)
plt.colorbar(q, ax=ax, label="||score|| (화살표 크기, 방향은 정규화)")
ax.set_title(f"학습된 score field s_theta(x, sigma={QUIVER_SIGMA:.2f})\n"
             f"화살표는 각 지점에서 log p_sigma(x)가 커지는 방향을 가리킴")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend(loc="upper right")
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
fig.tight_layout()
fig.savefig("03-score-quiver.png", dpi=140)
plt.close(fig)
print("저장: 03-score-quiver.png")

# ---------------------------------------------------------------------------
# 5. (b) Annealed Langevin Dynamics 샘플링
#    NCSN 알고리즘: sigma를 큰 값에서 작은 값으로 내려가며, 각 sigma에서
#    step size alpha_i = step * (sigma_i / sigma_min)^2 로 Langevin 반복.
# ---------------------------------------------------------------------------
N_SAMPLES = 500
STEPS_PER_SIGMA = 100
STEP_SIZE = 4e-5

x_t = rng.normal(scale=SIGMA_MAX, size=(N_SAMPLES, 2))  # 순수 노이즈에서 시작

snapshots = {0: x_t.copy()}
snapshot_sigma_indices = [0, 2, 5, L - 1]  # 초기/중간/중간/최종 근처 sigma level

for i, sigma_i in enumerate(sigmas):
    alpha_i = STEP_SIZE * (sigma_i / SIGMA_MIN) ** 2
    for step in range(STEPS_PER_SIGMA):
        s = score_fn(params, x_t, sigma_i)
        z = rng.normal(size=x_t.shape)
        x_t = x_t + (alpha_i / 2.0) * s + np.sqrt(alpha_i) * z
    if i in snapshot_sigma_indices:
        snapshots[i + 1] = x_t.copy()

snapshots[L] = x_t.copy()  # 최종 결과 확실히 저장

# 스냅샷 4개를 뽑아서 초기 -> 중간 -> 중간 -> 최종 순서로 시각화
snap_keys = sorted(snapshots.keys())
chosen_keys = [snap_keys[0], snap_keys[len(snap_keys) // 3], snap_keys[2 * len(snap_keys) // 3], snap_keys[-1]]
chosen_keys = sorted(set(chosen_keys))
if len(chosen_keys) < 4:
    chosen_keys = snap_keys[: 4]

fig, axes = plt.subplots(1, len(chosen_keys), figsize=(4.4 * len(chosen_keys), 5.2), sharex=True, sharey=True)
titles = []
for key in chosen_keys:
    if key == 0:
        titles.append("초기: 순수 랜덤 노이즈\n(x ~ N(0, sigma_max^2 I))")
    elif key == L:
        titles.append(f"최종 (sigma={sigmas[-1]:.3f})\nLangevin 종료")
    else:
        titles.append(f"중간 (sigma level {key}/{L})\nsigma={sigmas[min(key, L-1)]:.3f}")

for ax, key, title in zip(axes, chosen_keys, titles):
    ax.scatter(X[:, 0], X[:, 1], s=3, c="lightgray", zorder=1, label="실제 데이터")
    ax.scatter(snapshots[key][:, 0], snapshots[key][:, 1], s=6, c="crimson", zorder=2, label="샘플")
    ax.set_title(title, fontsize=9, pad=10)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
axes[0].legend(loc="upper right", fontsize=8)
fig.suptitle("Annealed Langevin Dynamics: 랜덤 노이즈 -> moons 분포로 수렴하는 과정 (실제 실행 결과)", y=1.06)
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig("03-langevin-sampling.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("저장: 03-langevin-sampling.png")

# ---------------------------------------------------------------------------
# 6. 최종 샘플 품질을 대략적인 수치로도 확인 (각 샘플에서 가장 가까운 실제
#    데이터 점까지의 평균 거리 - 초기 노이즈 대비 최종 샘플이 얼마나 데이터
#    쪽으로 이동했는지 정량적으로 비교)
# ---------------------------------------------------------------------------
def mean_nearest_dist(samples, data):
    d2 = ((samples[:, None, :] - data[None, :, :]) ** 2).sum(-1)
    return np.sqrt(d2.min(axis=1)).mean()


init_dist = mean_nearest_dist(snapshots[0], X)
final_dist = mean_nearest_dist(snapshots[L], X)
print(f"초기 노이즈 -> 최근접 데이터점 평균 거리: {init_dist:.3f}")
print(f"최종 샘플   -> 최근접 데이터점 평균 거리: {final_dist:.3f}")
