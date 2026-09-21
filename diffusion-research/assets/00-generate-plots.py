"""
00-math-foundations.md 용 시각자료 생성 스크립트.
신경망 학습이나 외부 데이터셋 없이, numpy로 만든 가우시안 예시만으로 그린다.
실행: python3 00-generate-plots.py  (같은 폴더에 PNG 2개 생성)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 11
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["font.family"] = "AppleGothic"  # 한글 라벨이 깨지지 않도록
plt.rcParams["axes.unicode_minus"] = False


def gaussian_pdf(x, mu, var):
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)


# ---------------------------------------------------------------------------
# 그림 1: 두 가우시안을 곱하면 어떻게 되는가 (섹션 1.3, 실습 과제 1과 동일한 숫자)
# mu1=0, s1=1 (var1=1) / mu2=2, s2=2 (var2=4)
# 결과: var = 0.8, mu = 0.4  (본문 1.3 공식으로 계산한 값)
# ---------------------------------------------------------------------------
mu1, var1 = 0.0, 1.0 ** 2
mu2, var2 = 2.0, 2.0 ** 2

precision1 = 1.0 / var1
precision2 = 1.0 / var2
var_prod = 1.0 / (precision1 + precision2)
mu_prod = (mu1 * precision1 + mu2 * precision2) * var_prod

x = np.linspace(-5, 8, 1000)
p1 = gaussian_pdf(x, mu1, var1)
p2 = gaussian_pdf(x, mu2, var2)
p_prod_normalized = gaussian_pdf(x, mu_prod, var_prod)
# 정규화 전의 실제 곱 (스케일이 작아서 비교용으로 별도 축에 그림)
p_raw_product = p1 * p2

fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
ax1.plot(x, p1, color="#4C72B0", lw=2.2, label="N(x; mu1=0, s1^2=1)")
ax1.plot(x, p2, color="#DD8452", lw=2.2, label="N(x; mu2=2, s2^2=4)")
ax1.plot(
    x,
    p_prod_normalized,
    color="#55A868",
    lw=2.6,
    ls="--",
    label=f"결과 (정규화됨): mu={mu_prod:.2f}, var={var_prod:.2f}",
)
ax1.axvline(mu_prod, color="#55A868", lw=1, alpha=0.5, ls=":")
ax1.set_xlabel("x")
ax1.set_ylabel("확률밀도 (probability density)")
ax1.set_title("두 가우시안을 곱하면: 결과 평균은 '정밀도 가중 평균'")
ax1.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig("00-gaussian-product.png", dpi=160)
plt.close(fig)


# ---------------------------------------------------------------------------
# 그림 2: ELBO가 log p(x)의 하한(lower bound)이라는 개념도 (섹션 2.4)
# q가 최적화 과정에서 true posterior p(z|x)에 가까워질수록
# KL(q||p(z|x))가 줄어들고 ELBO(q)가 log p(x)에 붙는다는 것을 보여준다.
# 실제 학습 곡선이 아니라 개념을 보여주기 위해 만든 예시 곡선이다.
# ---------------------------------------------------------------------------
steps = np.linspace(0, 1, 200)
log_px = np.full_like(steps, 1.0)  # log p(x)는 q와 무관한 상수이므로 수평선
kl_gap = 0.55 * np.exp(-4.0 * steps)  # 최적화가 진행될수록 KL이 줄어드는 모습(예시)
elbo = log_px - kl_gap

fig, ax2 = plt.subplots(figsize=(7.5, 4.8))
ax2.plot(steps, log_px, color="#4C72B0", lw=2.4, label="log p(x)  (q와 무관한 상수)")
ax2.plot(steps, elbo, color="#55A868", lw=2.4, label="ELBO(q)  (최적화 대상)")
ax2.fill_between(
    steps,
    elbo,
    log_px,
    color="#C44E52",
    alpha=0.25,
    label="KL(q(z) || p(z|x))  (log p(x) - ELBO)",
)
ax2.set_xlabel("최적화 진행 방향 (q가 p(z|x)에 가까워지는 방향)")
ax2.set_ylabel("값 (value)")
ax2.set_title("ELBO는 log p(x)의 하한이다 — 간격 = KL divergence")
ax2.set_xticks([])
ax2.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig("00-elbo-lower-bound.png", dpi=160)
plt.close(fig)

print("saved: 00-gaussian-product.png, 00-elbo-lower-bound.png")
print(f"gaussian product check -> mu={mu_prod}, var={var_prod}")
