"""
04챕터 toy 실험: 2D Gaussian mixture를 '진짜 데이터 분포'로 놓고
- 해석적으로(analytic) 정확한 score/epsilon을 계산해서 (신경망 학습 없이)
- (A) DDPM(확률적, T=1000) vs DDIM(결정론적, steps=20~50) 샘플링 경로 비교
- (B) Classifier-Free Guidance: scale w를 키울 때 분포가 특정 모드로 좁혀지는 현상 관찰

00번 챕터(가우시안 곱), 02번 챕터(DDPM 알고리즘), 03번 챕터(score = -eps/sqrt(1-abar)) 식을
그대로 재사용한다. 신경망 대신 '오라클(oracle) score'를 쓰는 이유: 목표가 guidance/DDIM
자체의 수학적 효과를 깨끗하게 보는 것이라, 학습 오차라는 잡음을 섞고 싶지 않아서다.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(0)

# ---------------------------------------------------------------
# 0. 진짜 데이터 분포: 2D 평면 위 3개의 봉우리(mode)를 가진 Gaussian mixture
#    각 mode를 '클래스 y=0,1,2'라고 생각하면 그대로 classifier(-free) guidance의
#    "조건부 p(x|y)" 실험 세팅이 된다.
# ---------------------------------------------------------------
MEANS = np.array([[-4.0, -4.0], [4.0, 4.0], [4.0, -4.0]])
COMP_VAR = 0.35  # 각 mode의 (등방성) 분산
K = len(MEANS)
WEIGHTS = np.ones(K) / K

# ---------------------------------------------------------------
# 1. DDPM의 variance-preserving noise schedule (02번 챕터 그대로)
# ---------------------------------------------------------------
T = 1000
betas = np.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alpha_bars = np.cumprod(alphas)


def marginal_params(t_idx, means=MEANS, weights=WEIGHTS):
    """시각 t에서 노이즈 낀 주변분포 p_t(x_t)의 GMM 파라미터.
    가우시안을 가우시안으로 convolve하면 다시 가우시안이므로(00번 챕터 1.1),
    mixture의 각 성분도 평균/분산만 바뀐 가우시안으로 남는다."""
    ab = alpha_bars[t_idx]
    m = np.sqrt(ab) * means
    var = ab * COMP_VAR + (1 - ab)
    return m, var, weights


def gmm_score(x, means, var, weights):
    """등방성 공분산 var(스칼라, 성분마다 같을 수도 다를 수도)를 갖는 Gaussian mixture의
    score = grad_x log sum_k w_k N(x; m_k, var*I).
    책임도(responsibility)로 가중한 각 성분 score의 평균이 된다 (00번 챕터 1.3의 '여러 가우시안을
    섞는다'는 아이디어의 연장선)."""
    var = np.atleast_1d(var).astype(float)
    if var.shape[0] == 1:
        var = np.repeat(var, means.shape[0])
    diffs = x[:, None, :] - means[None, :, :]  # (N,K,2)
    sq = np.sum(diffs ** 2, axis=-1) / var[None, :]  # (N,K)
    log_norm = -0.5 * sq - np.log(2 * np.pi * var)[None, :]
    log_w = np.log(weights)[None, :]
    logp_k = log_w + log_norm
    logp_k -= logp_k.max(axis=1, keepdims=True)
    p_k = np.exp(logp_k)
    resp = p_k / p_k.sum(axis=1, keepdims=True)  # (N,K)
    comp_score = -diffs / var[None, :, None]
    return np.sum(resp[:, :, None] * comp_score, axis=1)


def eps_uncond(x, t_idx):
    """무조건부 score -> epsilon_theta 로 환산 (03번 챕터 식: s = -eps/sqrt(1-abar))"""
    m, var, w = marginal_params(t_idx)
    s = gmm_score(x, m, var, w)
    sigma = np.sqrt(1 - alpha_bars[t_idx])
    return -sigma * s


def eps_cond(x, t_idx, k):
    """조건부 score (조건 y=k는 그냥 그 mode 하나만 남긴 가우시안)"""
    ab = alpha_bars[t_idx]
    m = np.sqrt(ab) * MEANS[k:k + 1]
    var = ab * COMP_VAR + (1 - ab)
    s = gmm_score(x, m, np.array([var]), np.array([1.0]))
    sigma = np.sqrt(1 - ab)
    return -sigma * s


def eps_cfg(x, t_idx, k, w):
    """Classifier-Free Guidance 외삽 (섹션 2.2 식 그대로)"""
    e_u = eps_uncond(x, t_idx)
    e_c = eps_cond(x, t_idx, k)
    return e_u + w * (e_c - e_u)


# ---------------------------------------------------------------
# 2. DDPM ancestral sampling (02번 챕터 섹션 4 알고리즘)
# ---------------------------------------------------------------
def ddpm_sample(n, eps_fn, record_traj_n=0):
    x = rng.normal(size=(n, 2))
    traj = [x[:record_traj_n].copy()] if record_traj_n else None
    for t in reversed(range(T)):
        z = rng.normal(size=x.shape) if t > 0 else 0.0
        beta_t, alpha_t, ab_t = betas[t], alphas[t], alpha_bars[t]
        eps = eps_fn(x, t)
        x = (1 / np.sqrt(alpha_t)) * (x - (beta_t / np.sqrt(1 - ab_t)) * eps) + np.sqrt(beta_t) * z
        if record_traj_n:
            traj.append(x[:record_traj_n].copy())
    return x, traj


# ---------------------------------------------------------------
# 3. DDIM deterministic sampling, 스텝 수를 듬성듬성 건너뜀 (섹션 1.2/1.3)
# ---------------------------------------------------------------
def ddim_sample(n, eps_fn, steps, record_traj_n=0):
    seq = np.unique(np.linspace(0, T - 1, steps).astype(int))[::-1]
    x = rng.normal(size=(n, 2))
    traj = [x[:record_traj_n].copy()] if record_traj_n else None
    for i, t in enumerate(seq):
        t_prev = seq[i + 1] if i + 1 < len(seq) else -1
        ab_t = alpha_bars[t]
        ab_prev = alpha_bars[t_prev] if t_prev >= 0 else 1.0
        eps = eps_fn(x, t)
        x0_hat = (x - np.sqrt(1 - ab_t) * eps) / np.sqrt(ab_t)
        # sigma_t = 0 -> 결정론적 (식 그대로: sqrt(abar_prev)*x0_hat + sqrt(1-abar_prev)*eps)
        x = np.sqrt(ab_prev) * x0_hat + np.sqrt(1 - ab_prev) * eps
        if record_traj_n:
            traj.append(x[:record_traj_n].copy())
    return x, traj


# =================================================================
# 실험 A: DDPM(1000 step, 확률적) vs DDIM(20 step, 결정론적)
# =================================================================
print("실험 A: DDPM vs DDIM 실행 중...")
N_SAMPLES = 800
N_TRAJ = 6

ddpm_final, ddpm_traj = ddpm_sample(N_SAMPLES, eps_uncond, record_traj_n=N_TRAJ)
ddim_final, ddim_traj = ddim_sample(N_SAMPLES, eps_uncond, steps=20, record_traj_n=N_TRAJ)

# 참고용 진짜 데이터 샘플 (mixture에서 직접 샘플링)
true_k = rng.choice(K, size=2000, p=WEIGHTS)
true_samples = MEANS[true_k] + np.sqrt(COMP_VAR) * rng.normal(size=(2000, 2))

fig, axes = plt.subplots(2, 2, figsize=(11, 10))

# (top-left) DDPM trajectories
ax = axes[0, 0]
ax.scatter(true_samples[:, 0], true_samples[:, 1], s=4, c="lightgray", alpha=0.5, label="true data")
ddpm_traj_arr = np.array(ddpm_traj)  # (T+1, N_TRAJ, 2)
for i in range(N_TRAJ):
    path = ddpm_traj_arr[:, i, :]
    ax.plot(path[:, 0], path[:, 1], lw=0.6, alpha=0.8)
ax.set_title(f"DDPM trajectory (stochastic, T={T} steps)")
ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
ax.legend(loc="upper left", fontsize=8)

# (top-right) DDIM trajectories
ax = axes[0, 1]
ax.scatter(true_samples[:, 0], true_samples[:, 1], s=4, c="lightgray", alpha=0.5, label="true data")
ddim_traj_arr = np.array(ddim_traj)  # (steps+1, N_TRAJ, 2)
for i in range(N_TRAJ):
    path = ddim_traj_arr[:, i, :]
    ax.plot(path[:, 0], path[:, 1], lw=1.2, alpha=0.9, marker="o", markersize=2)
ax.set_title("DDIM trajectory (deterministic, 20 steps)")
ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
ax.legend(loc="upper left", fontsize=8)

# (bottom-left) DDPM final samples
ax = axes[1, 0]
ax.scatter(true_samples[:, 0], true_samples[:, 1], s=4, c="lightgray", alpha=0.4, label="true data")
ax.scatter(ddpm_final[:, 0], ddpm_final[:, 1], s=8, c="tab:blue", alpha=0.6, label="DDPM samples")
ax.set_title(f"DDPM final samples ({T} NFE)")
ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
ax.legend(loc="upper left", fontsize=8)

# (bottom-right) DDIM final samples
ax = axes[1, 1]
ax.scatter(true_samples[:, 0], true_samples[:, 1], s=4, c="lightgray", alpha=0.4, label="true data")
ax.scatter(ddim_final[:, 0], ddim_final[:, 1], s=8, c="tab:red", alpha=0.6, label="DDIM samples")
ax.set_title("DDIM final samples (20 NFE)")
ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
ax.legend(loc="upper left", fontsize=8)

fig.suptitle("DDPM(stochastic, 1000 NFE) vs DDIM(deterministic, 20 NFE) on a toy 2D GMM\n"
             "(NFE = Number of Function Evaluations, i.e. 신경망 호출 횟수)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("/Users/kangmin/Desktop/personal-study/AI/diffusion-research/assets/04-ddim-vs-ddpm-trajectory.png", dpi=140)
print("saved 04-ddim-vs-ddpm-trajectory.png")

# 정량 비교용 통계 출력 (문서에 표로 옮길 것)
def nn_distance_to_true(samples, true_pts):
    # 각 샘플에서 진짜 데이터 클라우드까지 최근접 거리 평균 (분포 근접도 대략적 지표)
    d2 = ((samples[:, None, :] - true_pts[None, :, :]) ** 2).sum(-1)
    return np.sqrt(d2.min(axis=1)).mean()

print(f"[A] DDPM avg nearest-true-sample distance: {nn_distance_to_true(ddpm_final, true_samples):.4f}")
print(f"[A] DDIM avg nearest-true-sample distance: {nn_distance_to_true(ddim_final, true_samples):.4f}")

# =================================================================
# 실험 B: Classifier-Free Guidance scale sweep
# =================================================================
print("\n실험 B: CFG scale sweep 실행 중...")
TARGET_K = 0  # mode 0 = (-4,-4)을 '조건 y'로 지정
W_LIST = [0.0, 1.0, 3.0, 7.0, 15.0]
N_CFG = 500

cfg_results = {}
for w in W_LIST:
    fn = lambda x, t, w=w: eps_cfg(x, t, TARGET_K, w)
    samples, _ = ddim_sample(N_CFG, fn, steps=50)
    cfg_results[w] = samples

fig2, axes2 = plt.subplots(1, len(W_LIST) + 1, figsize=(4 * (len(W_LIST) + 1), 4))
for ax, w in zip(axes2[:-1], W_LIST):
    s = cfg_results[w]
    ax.scatter(true_samples[:, 0], true_samples[:, 1], s=3, c="lightgray", alpha=0.4)
    ax.scatter(s[:, 0], s[:, 1], s=8, c="tab:purple", alpha=0.6)
    ax.scatter(*MEANS[TARGET_K], marker="*", c="black", s=120, label="target mode y")
    ax.set_title(f"w = {w}")
    ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
    ax.set_aspect("equal")
axes2[0].legend(loc="upper left", fontsize=7)

# 우측 패널: 정량 지표 (fidelity: 목표 mode까지 평균거리 / diversity: 샘플 표준편차)
ws = np.array(W_LIST)
mean_dist = np.array([np.linalg.norm(cfg_results[w] - MEANS[TARGET_K], axis=1).mean() for w in W_LIST])
spread = np.array([cfg_results[w].std(axis=0).mean() for w in W_LIST])
# 다른 두 mode(원치 않는 mode)에 떨어진 샘플 비율 (조건 불충실도)
other_means = np.delete(MEANS, TARGET_K, axis=0)
def frac_near_other(s):
    d_target = np.linalg.norm(s - MEANS[TARGET_K], axis=1)
    d_other = np.min(np.linalg.norm(s[:, None, :] - other_means[None, :, :], axis=2), axis=1)
    return float(np.mean(d_other < d_target))
other_frac = np.array([frac_near_other(cfg_results[w]) for w in W_LIST])

ax = axes2[-1]
ax.plot(ws, mean_dist, "o-", color="tab:red", label="fidelity: mean dist to target mode (↓ 좋음)")
ax.plot(ws, spread, "s-", color="tab:blue", label="diversity: sample std (↓ 다양성 감소)")
ax.plot(ws, other_frac * mean_dist.max(), "^--", color="tab:green",
        label="다른 mode에 더 가까운 샘플 비율 (스케일 조정, ↓ 좋음)")
ax.set_xlabel("guidance scale w")
ax.set_title("Fidelity ↑ vs Diversity ↓ trade-off")
ax.legend(fontsize=7, loc="upper right")

fig2.suptitle(f"Classifier-Free Guidance: scale w를 키울수록 target mode(y={TARGET_K})로 분포가 좁혀진다", fontsize=12)
fig2.tight_layout(rect=[0, 0, 1, 0.92])
fig2.savefig("/Users/kangmin/Desktop/personal-study/AI/diffusion-research/assets/04-cfg-scale-sweep.png", dpi=140)
print("saved 04-cfg-scale-sweep.png")

print("\n=== 정량 결과 요약 (표로 옮길 것) ===")
print(f"{'w':>6} | {'mean_dist_to_target':>20} | {'sample_std':>10} | {'frac_near_other_mode':>20}")
for w, md, sp, of in zip(ws, mean_dist, spread, other_frac):
    print(f"{w:6.1f} | {md:20.4f} | {sp:10.4f} | {of:20.4f}")

print("\nDONE")
