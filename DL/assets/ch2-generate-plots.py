"""
chapter2-improving-deep-neural-networks.md (DL Course 2) 용 시각자료 생성 스크립트.
외부 데이터셋이나 딥러닝 프레임워크 없이 numpy로 만든 장난감 예시만으로 그린다.

생성되는 그림 (전부 이 스크립트와 같은 assets/ 폴더에 저장):
  ch2-bias-variance.png       underfit / just right / overfit decision boundary 비교
  ch2-early-stopping.png      iteration에 따른 train/dev error와 ||w|| (early stopping 지점)
  ch2-tanh-linear.png         z가 작으면 tanh(z) ≈ z (L2가 네트워크를 '선형에 가깝게' 만드는 이유)
  ch2-contours-normalization.png  길쭉한 등고선 vs 둥근 등고선 위의 gradient descent 경로
  ch2-vanishing-exploding.png 층을 지날 때 신호 크기: 배율별 지수 증가/감소 + 초기화별 실제 activation std
  ch2-ewma.png                β에 따른 지수가중평균 곡선 + bias correction 효과
  ch2-optimizers.png          f = 0.5(x^2 + 25y^2) 위 GD / Momentum / RMSprop / Adam 궤적
  ch2-lr-decay.png            learning rate decay 스케줄 + 고정 α vs decay α의 noisy GD 궤적
  ch2-saddle-plateau.png      local minimum vs saddle point 곡면, plateau에서의 loss 곡선
  ch2-hparam-search.png       grid vs random search, uniform vs log-scale 샘플링 비율

실행: 저장소 어디서든
    python3 DL/assets/ch2-generate-plots.py
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

plt.rcParams["font.size"] = 11
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["font.family"] = "AppleGothic"  # 한글 라벨이 깨지지 않도록
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"  # log 축 눈금(10^-5 등) 수식 폰트

OUT = os.path.dirname(os.path.abspath(__file__))


def save(fig, name):
    # log 축 눈금을 mathtext(10^{-5}) 대신 "1e-05" 형태로: AppleGothic에 수학 minus 글리프가 없어서
    for ax in fig.axes:
        for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            if scale == "log":
                axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda val, _: f"{val:g}"))
                axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ---------------------------------------------------------------------------
# 그림 1: Bias / Variance — 같은 데이터에 세 가지 모델
#   진짜 경계: 원 x1^2 + x2^2 = 1.2 (안쪽 = 클래스 1), 라벨 10%는 뒤집음(노이즈)
#   underfit  = 로지스틱 회귀(직선)
#   just right= 로지스틱 회귀 + 2차 feature(원/타원을 그릴 수 있음)
#   overfit   = 1-nearest-neighbor(모든 train 점을 외움 → train error 0)
# ---------------------------------------------------------------------------
rng = np.random.RandomState(3)


def make_data(n):
    X = rng.uniform(-2, 2, size=(n, 2))
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 < 1.2 * 1.2).astype(float)
    flip = rng.rand(n) < 0.1
    y[flip] = 1 - y[flip]
    return X, y


Xtr, ytr = make_data(120)
Xdv, ydv = make_data(2000)


def logistic_fit(F, y, lr=0.5, iters=5000):
    w = np.zeros(F.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-F @ w))
        w -= lr * F.T @ (p - y) / len(y)
    return w


def feat_lin(X):
    return np.c_[np.ones(len(X)), X]


def feat_quad(X):
    return np.c_[np.ones(len(X)), X, X ** 2, X[:, :1] * X[:, 1:]]


w_lin = logistic_fit(feat_lin(Xtr), ytr)
w_quad = logistic_fit(feat_quad(Xtr), ytr)


def pred_lin(X):
    return (feat_lin(X) @ w_lin > 0).astype(float)


def pred_quad(X):
    return (feat_quad(X) @ w_quad > 0).astype(float)


def pred_1nn(X):
    d = ((X[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    return ytr[d.argmin(1)]


gx, gy = np.meshgrid(np.linspace(-2, 2, 300), np.linspace(-2, 2, 300))
G = np.c_[gx.ravel(), gy.ravel()]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (title, f) in zip(
    axes,
    [
        ("High bias (underfitting)\n직선 — 원 모양을 못 그림", pred_lin),
        ("Just right\n2차 곡선 — 진짜 경계와 비슷", pred_quad),
        ("High variance (overfitting)\n1-NN — 노이즈 점까지 다 감쌈", pred_1nn),
    ],
):
    Z = f(G).reshape(gx.shape)
    ax.contourf(gx, gy, Z, levels=[-0.5, 0.5, 1.5], colors=["#f4c7c3", "#c6dbef"], alpha=0.8)
    ax.contour(gx, gy, Z, levels=[0.5], colors="k", linewidths=1.2)
    ax.scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], c="#1f5fa8", s=22, label="클래스 1 (train)")
    ax.scatter(Xtr[ytr == 0, 0], Xtr[ytr == 0, 1], c="#c0392b", marker="x", s=22, label="클래스 0 (train)")
    circ = plt.Circle((0, 0), 1.2, fill=False, ls="--", color="green", lw=1.5)
    ax.add_patch(circ)
    tr_err = np.mean(f(Xtr) != ytr) * 100
    dv_err = np.mean(f(Xdv) != ydv) * 100
    ax.set_title(f"{title}\ntrain error {tr_err:.0f}%  /  dev error {dv_err:.0f}%", fontsize=11)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.grid(False)
    print(f"[bias-variance] {title.splitlines()[0]}: train {tr_err:.1f}%, dev {dv_err:.1f}%")
axes[0].legend(loc="lower left", fontsize=9)
fig.suptitle("초록 점선 = 진짜 경계, 검은 선 = 모델이 학습한 경계 (라벨 10%는 일부러 뒤집은 노이즈)", y=1.02)
save(fig, "ch2-bias-variance.png")


# ---------------------------------------------------------------------------
# 그림 2: Early stopping — 12차 다항식 회귀를 GD로 오래 학습
#   train 15개 / dev 300개, y = sin(3x) + noise. dev error는 U자, ||w||는 계속 증가.
# ---------------------------------------------------------------------------
rng = np.random.RandomState(4)


def data_1d(n):
    x = rng.uniform(-1, 1, n)
    return x, np.sin(3 * x) + rng.randn(n) * 0.3


xtr, ytr1 = data_1d(15)
xdv, ydv1 = data_1d(300)
D = 12


def poly(x):
    return np.stack([x ** k for k in range(D + 1)], 1)


Ftr, Fdv = poly(xtr), poly(xdv)
w = np.zeros(D + 1)
its = np.unique(np.logspace(0, 5.3, 80).astype(int))
tr_hist, dv_hist, norm_hist = [], [], []
it = 0
for T in its:
    while it < T:
        w -= 0.5 * Ftr.T @ (Ftr @ w - ytr1) / len(xtr)
        it += 1
    tr_hist.append(np.mean((Ftr @ w - ytr1) ** 2))
    dv_hist.append(np.mean((Fdv @ w - ydv1) ** 2))
    norm_hist.append(np.linalg.norm(w))
tr_hist, dv_hist, norm_hist = map(np.array, (tr_hist, dv_hist, norm_hist))
best = dv_hist.argmin()
print(f"[early-stopping] best iter {its[best]}, dev {dv_hist[best]:.3f}, final dev {dv_hist[-1]:.3f}, "
      f"||w|| at best {norm_hist[best]:.2f}, final {norm_hist[-1]:.2f}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(its, tr_hist, lw=2, label="train error (계속 감소)")
ax.plot(its, dv_hist, lw=2, label="dev error (U자)")
ax.axvline(its[best], color="gray", ls="--")
ax.annotate(f"여기서 멈춤\n(iteration ≈ {its[best]})", xy=(its[best], dv_hist[best]),
            xytext=(its[best] * 4, 0.22), arrowprops=dict(arrowstyle="->"), fontsize=10)
ax.set_xscale("log")
ax.set_xlabel("iteration (log scale)")
ax.set_ylabel("MSE")
ax2 = ax.twinx()
ax2.plot(its, norm_hist, color="purple", ls=":", lw=2, label="‖w‖ (오른쪽 축)")
ax2.set_yscale("log")
ax2.set_ylabel("‖w‖ (log scale)", color="purple")
ax2.grid(False)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=9)
ax.set_title("Early stopping: 학습이 진행될수록 ‖w‖가 커지고, 어느 순간부터 dev error가 다시 오른다")
save(fig, "ch2-early-stopping.png")


# ---------------------------------------------------------------------------
# 그림 3: tanh는 0 근처에서 거의 직선
# ---------------------------------------------------------------------------
z = np.linspace(-3, 3, 400)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.axvspan(-0.5, 0.5, color="gold", alpha=0.3, label="|z| < 0.5: tanh(z) ≈ z (거의 선형)")
ax.plot(z, np.tanh(z), lw=2.5, label="tanh(z)")
ax.plot(z, z, "--", color="gray", label="z (직선)")
for zz in [0.1, 2.0]:
    ax.plot(zz, np.tanh(zz), "ko")
    ax.annotate(f"tanh({zz:g}) = {np.tanh(zz):.4f}" if zz < 1 else f"tanh({zz:g}) = {np.tanh(zz):.3f}",
                xy=(zz, np.tanh(zz)), xytext=(zz + 0.3, np.tanh(zz) - 0.6 if zz < 1 else 0.4),
                arrowprops=dict(arrowstyle="->"), fontsize=10)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel("z")
ax.set_title("W가 작아지면 z도 작아져서, activation이 노란 구간(선형)만 쓰게 된다")
ax.legend(loc="upper left", fontsize=9)
save(fig, "ch2-tanh-linear.png")


# ---------------------------------------------------------------------------
# 그림 4: 입력 정규화 전/후 등고선 + GD 경로
#   왼쪽: J = 0.5(w1^2 + 25 w2^2), α=0.075 (한계 2/25=0.08 바로 아래)
#   오른쪽: J = 0.5(w1^2 + w2^2), α=0.5 (한계 2)
#   둘 다 같은 스텝 수(30)만 진행
# ---------------------------------------------------------------------------
def gd_path(a, b, lr, start, steps):
    p = np.array(start, float)
    path = [p.copy()]
    for _ in range(steps):
        p = p - lr * np.array([a * p[0], b * p[1]])
        path.append(p.copy())
    return np.array(path)


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (a, b, lr, title) in zip(
    axes,
    [
        (1, 25, 0.075, "정규화 전: 길쭉한 그릇 (w₂ 방향이 25배 가파름)\nα = 0.075 (발산 한계 2/25 = 0.08 바로 아래)"),
        (1, 1, 0.5, "정규화 후: 둥근 그릇\nα = 0.5 (발산 한계 2)"),
    ],
):
    path = gd_path(a, b, lr, (-9, 2), 30)
    xx, yy = np.meshgrid(np.linspace(-10, 3, 300), np.linspace(-3.5, 3.5, 300))
    ax.contour(xx, yy, 0.5 * (a * xx ** 2 + b * yy ** 2), levels=np.geomspace(0.05, 150, 18), cmap="viridis", alpha=0.6)
    ax.plot(path[:, 0], path[:, 1], "o-", color="crimson", ms=3.5, lw=1.2)
    ax.plot(0, 0, "k*", ms=14)
    ax.set_aspect("equal")
    ax.set_xlim(-10, 3)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel("w₁")
    ax.set_ylabel("w₂")
    dist = np.linalg.norm(path[-1])
    ax.set_title(f"{title}\n30 step 후 최솟값까지 거리 = {dist:.3f}", fontsize=10.5)
    print(f"[contours] a={a}, b={b}, lr={lr}: 30 step 후 거리 {dist:.4f}")
fig.suptitle("같은 30 step: 길쭉하면 지그재그 + 느린 전진, 둥글면 곧장 최솟값(★)으로", y=1.02)
save(fig, "ch2-contours-normalization.png")


# ---------------------------------------------------------------------------
# 그림 5: Vanishing / Exploding
#   왼쪽: 한 층당 배율 r일 때 L층 후 크기 r^L
#   오른쪽: 실제 50층 ReLU 네트워크(폭 100, b=0)에 입력 넣고 층별 activation std
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
L = np.arange(0, 51)
for r, c in [(1.5, "#c0392b"), (1.1, "#e67e22"), (1.0, "gray"), (0.9, "#2980b9"), (0.5, "#1a237e")]:
    axes[0].plot(L, r ** L.astype(float), lw=2, color=c, label=f"한 층당 ×{r}")
axes[0].set_yscale("log")
axes[0].set_xlabel("층 수 L")
axes[0].set_ylabel("신호 크기 (log scale)")
axes[0].set_title("선형 네트워크: 배율이 1에서 조금만 벗어나도 지수적으로 폭발/소멸")
axes[0].legend(fontsize=9)

rng = np.random.RandomState(0)
n, depth = 100, 50
x0 = rng.randn(n, 500)
for name, scale_fn, c in [
    ("randn × 0.01 (너무 작음)", lambda n_prev: 0.01, "#2980b9"),
    ("He: randn × √(2/n)", lambda n_prev: np.sqrt(2 / n_prev), "green"),
    ("randn × 1 (너무 큼)", lambda n_prev: 1.0, "#c0392b"),
]:
    a = x0.copy()
    stds = [a.std()]
    for _ in range(depth):
        W = rng.randn(n, n) * scale_fn(n)
        a = np.maximum(0, W @ a)
        stds.append(a.std())
    stds = np.array(stds)
    axes[1].plot(np.arange(depth + 1), np.maximum(stds, 1e-300), lw=2, color=c, label=name)
    print(f"[vanishing] {name}: std at layer 10 = {stds[10]:.3g}, layer 50 = {stds[50]:.3g}")
axes[1].set_yscale("log")
axes[1].set_xlabel("층 번호")
axes[1].set_ylabel("activation 표준편차 (log scale)")
axes[1].set_title("실제 ReLU 50층 (폭 100): 초기화 스케일에 따른 activation 크기")
axes[1].legend(fontsize=9)
save(fig, "ch2-vanishing-exploding.png")


# ---------------------------------------------------------------------------
# 그림 6: EWMA — Lab 4와 같은 온도 데이터(seed 1)
# ---------------------------------------------------------------------------
np.random.seed(1)
days = np.arange(1, 181)
temps = 15 + 10 * np.sin(days / 30) + np.random.randn(180) * 3


def ewma(x, beta, bias_correction=False):
    v, out = 0.0, []
    for t, xt in enumerate(x, start=1):
        v = beta * v + (1 - beta) * xt
        out.append(v / (1 - beta ** t) if bias_correction else v)
    return np.array(out)


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(days, temps, s=8, c="gray", alpha=0.6, label="실제 기온 θ(t) (노이즈)")
for beta, c in [(0.5, "#e67e22"), (0.9, "#c0392b"), (0.98, "green")]:
    axes[0].plot(days, ewma(temps, beta, True), lw=2, color=c, label=f"β={beta} (≈ 최근 {1/(1-beta):.0f}일 평균)")
axes[0].set_xlabel("day")
axes[0].set_ylabel("기온")
axes[0].set_title("β가 클수록 매끄럽지만 늦게 반응한다 (오른쪽으로 밀림)\n(세 곡선 모두 bias correction 적용)")
axes[0].legend(fontsize=9)

d = 40
axes[1].scatter(days[:d], temps[:d], s=12, c="gray", alpha=0.6, label="실제 기온")
axes[1].plot(days[:d], ewma(temps, 0.98)[:d], lw=2, color="green", ls="--", label="β=0.98, 보정 없음 (0에서 출발)")
axes[1].plot(days[:d], ewma(temps, 0.98, True)[:d], lw=2, color="green", label="β=0.98, 보정 v/(1-β^t)")
axes[1].set_xlabel("day")
axes[1].set_title("Bias correction: 초반 구간 확대 (처음 40일)")
axes[1].legend(fontsize=9)
print(f"[ewma] 보정 없는 v_1={ewma(temps,0.98)[0]:.2f}, 보정 {ewma(temps,0.98,True)[0]:.2f}, θ_1={temps[0]:.2f}")
save(fig, "ch2-ewma.png")


# ---------------------------------------------------------------------------
# 그림 7: Optimizer 궤적 — f = 0.5(x^2 + 25 y^2), 시작 (-9, 2), 60 step
#   GD/Momentum: α=0.075 (GD 한계 0.08 바로 아래), Momentum β=0.9
#   RMSprop: α=0.3, β=0.9 (bias correction 없음)
#   Adam: α=0.5, β1=0.9, β2=0.999 + bias correction
# ---------------------------------------------------------------------------
def grad_f(p):
    return np.array([p[0], 25 * p[1]])


def run_opt(opt, lr, steps=60, beta1=0.9, beta2=0.999, eps=1e-8):
    p = np.array([-9.0, 2.0])
    v, s = np.zeros(2), np.zeros(2)
    path = [p.copy()]
    for t in range(1, steps + 1):
        g = grad_f(p)
        if opt == "gd":
            p = p - lr * g
        elif opt == "momentum":
            v = beta1 * v + (1 - beta1) * g
            p = p - lr * v
        elif opt == "rmsprop":
            s = beta2 * s + (1 - beta2) * g ** 2
            p = p - lr * g / (np.sqrt(s) + eps)
        elif opt == "adam":
            v = beta1 * v + (1 - beta1) * g
            s = beta2 * s + (1 - beta2) * g ** 2
            p = p - lr * (v / (1 - beta1 ** t)) / (np.sqrt(s / (1 - beta2 ** t)) + eps)
        path.append(p.copy())
    return np.array(path)


configs = [
    ("gd", dict(lr=0.075), "Gradient Descent (α=0.075)"),
    ("momentum", dict(lr=0.075), "Momentum (α=0.075, β=0.9)"),
    ("rmsprop", dict(lr=0.3, beta2=0.9), "RMSprop (α=0.3, β=0.9)"),
    ("adam", dict(lr=0.5), "Adam (α=0.5, β1=0.9, β2=0.999)"),
]
xx, yy = np.meshgrid(np.linspace(-10, 2, 300), np.linspace(-3, 3, 300))
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, (opt, kw, title) in zip(axes.ravel(), configs):
    path = run_opt(opt, **kw)
    ax.contour(xx, yy, 0.5 * (xx ** 2 + 25 * yy ** 2), levels=np.geomspace(0.05, 150, 18), cmap="viridis", alpha=0.5)
    ax.plot(path[:, 0], path[:, 1], "o-", color="crimson", ms=3, lw=1.2)
    ax.plot(0, 0, "k*", ms=13)
    ax.set_xlim(-10, 2)
    ax.set_ylim(-3, 3)
    ax.set_title(f"{title}\n60 step 후 위치 ({path[-1][0]:.3f}, {path[-1][1]:.3f})", fontsize=10.5)
    print(f"[optimizers] {opt}: first steps {np.round(path[:3], 2).tolist()}, final {np.round(path[-1], 4)}")
fig.suptitle("f = ½(x² + 25y²), 시작점 (-9, 2): 세로(y) 진동을 어떻게 다루는가", y=1.0)
fig.tight_layout()
save(fig, "ch2-optimizers.png")


# ---------------------------------------------------------------------------
# 그림 8: Learning rate decay
#   왼쪽: α0=0.2에서 네 가지 스케줄
#   오른쪽: gradient에 노이즈가 섞인(mini-batch 흉내) GD — 고정 α vs 1/(1+kt) decay
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ep = np.arange(1, 51)
a0 = 0.2
axes[0].plot(ep, a0 / (1 + 1 * ep), lw=2, label="1/(1 + 1·epoch) · $\\alpha_0$")
axes[0].plot(ep, 0.95 ** ep * a0, lw=2, label="0.95^epoch · $\\alpha_0$ (exponential)")
axes[0].plot(ep, 1 / np.sqrt(ep) * a0, lw=2, label="k/√epoch · $\\alpha_0$ (k=1)")
axes[0].step(ep, a0 * 0.5 ** ((ep - 1) // 10), where="post", lw=2, label="10 epoch마다 절반 (staircase)")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("α")
axes[0].set_title("$\\alpha_0$ = 0.2에서 시작하는 decay 스케줄들")
axes[0].legend(fontsize=9)

rng = np.random.RandomState(0)


def noisy_gd(schedule, steps=300):
    p = np.array([-4.0, 3.0])
    path = [p.copy()]
    for t in range(steps):
        g = p + rng.randn(2) * 2.0  # 진짜 gradient(=p) + mini-batch 노이즈
        p = p - schedule(t) * g
        path.append(p.copy())
    return np.array(path)


p_fixed = noisy_gd(lambda t: 0.3)
p_decay = noisy_gd(lambda t: 0.3 / (1 + 0.05 * t))
d_fixed = np.linalg.norm(p_fixed, axis=1)
d_decay = np.linalg.norm(p_decay, axis=1)
axes[1].plot(d_fixed, color="#e67e22", lw=1.2, label="고정 α = 0.3")
axes[1].plot(d_decay, color="#1f5fa8", lw=1.5, label="α = 0.3 / (1 + 0.05t)")
axes[1].set_xlabel("step t")
axes[1].set_ylabel("최솟값(0,0)까지 거리")
axes[1].set_title("노이즈 섞인 gradient(mini-batch 흉내)로 J = ½‖w‖² 최소화\n"
                  "고정 α는 최솟값 주변에서 계속 크게 흔들리고, decay는 점점 좁혀 들어간다", fontsize=10.5)
axes[1].legend(fontsize=9)
r_fixed = d_fixed[-100:].mean()
r_decay = d_decay[-100:].mean()
print(f"[lr-decay] 마지막 100 step 평균 거리: fixed {r_fixed:.3f}, decay {r_decay:.3f}")
save(fig, "ch2-lr-decay.png")


# ---------------------------------------------------------------------------
# 그림 9: Local minimum vs saddle point, plateau
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(18, 5.2))
fig.subplots_adjust(wspace=0.35)
u, v = np.meshgrid(np.linspace(-2, 2, 60), np.linspace(-2, 2, 60))
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
ax1.plot_surface(u, v, u ** 2 + v ** 2, cmap="viridis", alpha=0.85, linewidth=0)
ax1.scatter([0], [0], [0], color="red", s=40)
ax1.set_title("Local minimum: J = w₁² + w₂²\n모든 방향으로 위로 휨 (그릇)", fontsize=10.5)
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
ax2.plot_surface(u, v, u ** 2 - v ** 2, cmap="coolwarm", alpha=0.85, linewidth=0)
ax2.scatter([0], [0], [0], color="black", s=40)
ax2.set_title("Saddle point: J = w₁² - w₂²\nw₁ 방향은 위로, w₂ 방향은 아래로 휨 (말안장)", fontsize=10.5)
for a in (ax1, ax2):
    a.set_xlabel("w₁")
    a.set_ylabel("w₂")
    a.view_init(elev=25, azim=-60)

ax3 = fig.add_subplot(1, 3, 3)
# J(w) = 1 - tanh(w): w가 작은(음수) 쪽은 거의 평평 → GD가 한참 기어감
w = -4.0
hist = []
for _ in range(400):
    hist.append(1 - np.tanh(w))
    w -= 1.0 * (-(1 - np.tanh(w) ** 2))
ax3.plot(hist, lw=2)
ax3.set_xlabel("iteration")
ax3.set_ylabel("cost J")
ax3.set_title("Plateau: gradient ≈ 0인 평지를 지나느라\n한참 동안 cost가 거의 안 줄어든다", fontsize=10.5)
k = int(np.argmax(np.array(hist) < 1.9))
ax3.annotate("plateau 구간", xy=(k / 2, 1.99), xytext=(k / 2, 1.5), arrowprops=dict(arrowstyle="->"), ha="center")
print(f"[saddle] plateau: cost가 1.9 아래로 내려가는 iteration = {k}")
save(fig, "ch2-saddle-plateau.png")


# ---------------------------------------------------------------------------
# 그림 10: 하이퍼파라미터 탐색 — grid vs random, uniform vs log-scale
# ---------------------------------------------------------------------------
rng = np.random.RandomState(0)
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
g = np.linspace(0.1, 0.9, 5)
GX, GY = np.meshgrid(g, g)
axes[0].scatter(GX, GY, c="#1f5fa8", s=40)
axes[0].scatter(g, np.full(5, -0.05), marker="|", c="crimson", s=300, clip_on=False)
axes[0].set_title("Grid search (5×5 = 25번)\n빨간 눈금(실제로 시도된 α 값) = 5개뿐", fontsize=10.5)
R = rng.rand(25, 2)
axes[1].scatter(R[:, 0], R[:, 1], c="#1f5fa8", s=40)
axes[1].scatter(R[:, 0], np.full(25, -0.05), marker="|", c="crimson", s=300, clip_on=False)
axes[1].set_title("Random search (25번)\n빨간 눈금(실제로 시도된 α 값) = 25개", fontsize=10.5)
for a in axes[:2]:
    a.set_xlim(0, 1)
    a.set_ylim(-0.08, 1)
    a.set_xlabel("하이퍼파라미터 1 (중요, 예: α)")
    a.set_ylabel("하이퍼파라미터 2 (별로 안 중요, 예: ε)")
    a.set_aspect("equal")

N = 100000
uni = rng.uniform(1e-4, 1, N)
logu = 10 ** (-4 * rng.rand(N))
edges = [1e-4, 1e-3, 1e-2, 1e-1, 1]
labels = ["0.0001\n~ 0.001", "0.001\n~ 0.01", "0.01\n~ 0.1", "0.1\n~ 1"]
fu = np.histogram(uni, edges)[0] / N * 100
fl = np.histogram(logu, edges)[0] / N * 100
xpos = np.arange(4)
b1 = axes[2].bar(xpos - 0.2, fu, 0.4, label="uniform(0.0001, 1)")
b2 = axes[2].bar(xpos + 0.2, fl, 0.4, label="10^r, r ~ uniform(-4, 0)")
for bars in (b1, b2):
    for bar in bars:
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{bar.get_height():.1f}%", ha="center", fontsize=9)
axes[2].set_xticks(xpos)
axes[2].set_xticklabels(labels)
axes[2].set_ylabel("뽑힌 비율 (%)")
axes[2].set_ylim(0, 100)
axes[2].set_title("α를 [0.0001, 1]에서 뽑을 때 자릿수 구간별 비율", fontsize=10.5)
axes[2].legend(fontsize=9)
print("[hparam] uniform:", np.round(fu, 2), " log:", np.round(fl, 2))
save(fig, "ch2-hparam-search.png")
