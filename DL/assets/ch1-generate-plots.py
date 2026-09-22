"""
chapter1-neural-networks-and-deep-learning.md 용 시각자료 생성 스크립트.
외부 데이터셋 없이 numpy로 만든 장난감 데이터만 사용한다. (학습도 numpy로 몇 천 스텝짜리 아주 작은 것뿐)

실행: python3 DL/assets/ch1-generate-plots.py   (이 스크립트와 같은 폴더에 PNG 7개 생성)

생성 파일:
  1. ch1-scale-performance.png     — 데이터 양 vs 성능 개념도 (전통 알고리즘 plateau vs 신경망 크기별)
  2. ch1-logloss-curve.png         — y=1/y=0일 때 cross-entropy loss(-log a, -log(1-a)) vs 제곱오차
  3. ch1-gradient-descent-1d.png   — 1D 로지스틱 회귀 cost J(w) 위에서 gradient descent가 내려가는 모습
  4. ch1-activation-functions.png  — sigmoid/tanh/ReLU/Leaky ReLU와 그 도함수 (saturation 구간 표시)
  5. ch1-broadcasting.png          — numpy broadcasting 3가지 경우 (행 복제 / 열 복제 / (5,1)+(1,5) 함정)
  6. ch1-decision-boundary.png     — 꽃잎(planar) 데이터에서 로지스틱 회귀(직선) vs 1-hidden-layer NN(비선형) 경계
  7. ch1-learning-rate-curves.png  — learning rate별 cost vs iteration (너무 작음 / 적당 / 너무 큼)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.size"] = 11
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------------------
# 1. 데이터 양 vs 성능 (개념도 — 실제 측정값 아님, 강의 그래프를 재현한 모양)
# ---------------------------------------------------------------------------
x = np.linspace(0, 10, 300)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(x, 0.45 * (1 - np.exp(-x / 1.2)), label="전통 알고리즘 (SVM, logistic regression)", lw=2.5, color="gray")
ax.plot(x, 0.55 * (1 - np.exp(-x / 2.0)), label="작은 신경망", lw=2.5)
ax.plot(x, 0.72 * (1 - np.exp(-x / 3.0)), label="중간 신경망", lw=2.5)
ax.plot(x, 0.95 * (1 - np.exp(-x / 4.5)), label="큰 신경망", lw=2.5)
ax.axvspan(0, 1.5, color="orange", alpha=0.12)
ax.text(0.1, 0.8, "데이터 적은 구간:\n순위 뒤죽박죽\n(feature 설계가 좌우)", fontsize=9, va="top")
ax.set_xlabel("(라벨 있는) 데이터 양 →")
ax.set_ylabel("성능 →")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("개념도: 데이터가 많을수록 큰 신경망이 계속 좋아진다")
ax.legend(loc="lower right", fontsize=9)
save(fig, "ch1-scale-performance.png")

# ---------------------------------------------------------------------------
# 2. Loss 곡선: -log a (y=1), -log(1-a) (y=0), 제곱오차 비교
# ---------------------------------------------------------------------------
a = np.linspace(0.005, 0.995, 400)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, y in zip(axes, [1, 0]):
    ce = -np.log(a) if y == 1 else -np.log(1 - a)
    ax.plot(a, ce, lw=2.5, label="cross-entropy " + ("$-\\log a$" if y == 1 else "$-\\log(1-a)$"))
    ax.plot(a, (a - y) ** 2, lw=2.5, ls="--", label="제곱오차 $(a-y)^2$")
    ax.set_xlabel("예측 확률 $a$")
    ax.set_title(f"정답 y = {y}")
    ax.set_ylim(0, 5)
    ax.legend()
axes[0].set_ylabel("loss")
axes[0].annotate("확신하고 틀림 (a=0.01)\nloss = 4.61", xy=(0.01, 4.61), xytext=(0.18, 2.6),
                 arrowprops=dict(arrowstyle="->"), fontsize=9)
axes[0].annotate("거의 맞춤 → loss ≈ 0", xy=(0.98, 0.02), xytext=(0.45, 1.2),
                 arrowprops=dict(arrowstyle="->"), fontsize=9)
fig.suptitle("맞출수록 0, 확신하고 틀릴수록 폭발 — 제곱오차는 최대 1에서 멈춘다", y=1.02)
save(fig, "ch1-logloss-curve.png")

# ---------------------------------------------------------------------------
# 3. 1D gradient descent: feature 1개, b=0 고정인 로지스틱 회귀 cost J(w)
# ---------------------------------------------------------------------------
xs = np.array([-2.0, -1.2, -0.5, 0.3, 0.8, 1.5, 2.2, -0.2, 0.9, -0.9])
ys = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 1])   # 일부러 섞어서(완벽 분리 불가) 최솟값이 유한한 곳에 생기게


def J_of_w(w):
    A = sigmoid(np.outer(np.atleast_1d(w), xs))
    return -np.mean(ys * np.log(A) + (1 - ys) * np.log(1 - A), axis=1)


def dJ_dw(w):
    A = sigmoid(w * xs)
    return np.mean((A - ys) * xs)


ws = np.linspace(-4, 6, 400)
Js = J_of_w(ws)
w = -3.5
alpha = 2.0
path = [w]
for _ in range(8):
    w = w - alpha * dJ_dw(w)
    path.append(w)
path = np.array(path)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(ws, Js, lw=2.5, color="C0", label="cost $J(w)$ (convex — 바닥이 하나)")
ax.plot(path, J_of_w(path), "o-", color="C3", ms=6, lw=1.2, label=f"gradient descent ($\\alpha$={alpha})")
for i in range(3):
    ax.annotate(f"{i}", (path[i], J_of_w(path[i])[0]), textcoords="offset points", xytext=(6, 6), fontsize=9, color="C3")
w_best = ws[np.argmin(Js)]
ax.axvline(w_best, color="gray", ls=":", lw=1)
ax.text(w_best + 0.1, Js.min() + 1.0, "최솟값", color="gray")
ax.annotate("기울기 음수 → w를 키움", xy=(path[0], J_of_w(path[0])[0]), xytext=(-3.0, Js.max() * 0.6),
            arrowprops=dict(arrowstyle="->"), fontsize=9)
ax.set_xlabel("w")
ax.set_ylabel("J(w)")
ax.set_title("기울기의 반대 방향으로 한 발짝씩 — 바닥에 가까울수록 기울기가 작아 보폭도 작아진다", fontsize=10)
ax.legend(loc="upper right")
save(fig, "ch1-gradient-descent-1d.png")

# ---------------------------------------------------------------------------
# 4. Activation functions + 도함수
# ---------------------------------------------------------------------------
z = np.linspace(-6, 6, 600)
s = sigmoid(z)
acts = [
    ("Sigmoid", s, s * (1 - s)),
    ("tanh", np.tanh(z), 1 - np.tanh(z) ** 2),
    ("ReLU", np.maximum(0, z), (z > 0).astype(float)),
    ("Leaky ReLU (0.01)", np.maximum(0.01 * z, z), np.where(z > 0, 1.0, 0.01)),
]
fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
for j, (name, g, dg) in enumerate(acts):
    axes[0, j].plot(z, g, lw=2.5, color="C0")
    axes[0, j].set_title(name)
    axes[1, j].plot(z, dg, lw=2.5, color="C1")
    axes[1, j].set_xlabel("z")
    if j < 2:
        for ax in axes[:, j]:
            ax.axvspan(-6, -3, color="red", alpha=0.08)
            ax.axvspan(3, 6, color="red", alpha=0.08)
        axes[1, j].text(4.5, 0.5 * dg.max(), "saturation\n기울기≈0", ha="center", fontsize=9, color="darkred")
    if name.startswith("Leaky"):
        axes[1, j].text(-3, 0.3, "z 음수: 기울기 0.01\n(0이 아니라 살아있음)", ha="center", fontsize=9)
    if name.startswith("ReLU"):
        axes[1, j].text(-3, 0.5, "z 음수: 기울기 0\n(dying ReLU)", ha="center", fontsize=9, color="darkred")
        axes[1, j].text(3, 0.8, "z 양수: 항상 1", ha="center", fontsize=9)
axes[0, 0].set_ylabel("g(z)")
axes[1, 0].set_ylabel("g'(z)")
axes[1, 0].annotate("최대 0.25 (z=0)", xy=(0, 0.25), xytext=(-5.5, 0.2), arrowprops=dict(arrowstyle="->"), fontsize=9)
axes[1, 1].annotate("최대 1 (z=0)", xy=(0, 1.0), xytext=(-5.5, 0.8), arrowprops=dict(arrowstyle="->"), fontsize=9)
fig.suptitle("윗줄: activation g(z) / 아랫줄: 도함수 g'(z) — 빨간 구간에서 sigmoid·tanh의 기울기가 사라진다", y=1.0)
fig.tight_layout()
save(fig, "ch1-activation-functions.png")

# ---------------------------------------------------------------------------
# 5. Broadcasting 그림
# ---------------------------------------------------------------------------


def draw_grid(ax, x0, y0, rows, cols, color, alpha=1.0, texts=None, ghost_from=None, fs=10):
    """(x0, y0)=왼쪽 위. ghost_from=(r, c) 기준 원본 칸이 아닌 곳은 연하게(복제된 칸)."""
    for r in range(rows):
        for c in range(cols):
            is_ghost = ghost_from is not None and not ghost_from(r, c)
            ax.add_patch(Rectangle((x0 + c, y0 - r - 1), 1, 1, facecolor=color,
                                   alpha=0.25 if is_ghost else alpha, edgecolor="black", lw=1,
                                   ls="--" if is_ghost else "-"))
            if texts is not None:
                ax.text(x0 + c + 0.5, y0 - r - 0.5, texts(r, c), ha="center", va="center", fontsize=fs,
                        color="gray" if is_ghost else "black")


fig, axes = plt.subplots(3, 1, figsize=(10, 12))
# (a) (4,3) + (1,3)
ax = axes[0]
draw_grid(ax, 0, 4, 4, 3, "C0", 0.35, texts=lambda r, c: "A")
ax.text(3.5, 2, "+", fontsize=18, ha="center", va="center")
draw_grid(ax, 4, 4, 4, 3, "C1", 0.6, texts=lambda r, c: str(100 * (c + 1)), ghost_from=lambda r, c: r == 0)
ax.text(7.5, 2, "=", fontsize=18, ha="center", va="center")
draw_grid(ax, 8, 4, 4, 3, "C2", 0.35, texts=lambda r, c: f"A+{100 * (c + 1)}", fs=8)
ax.set_title("① (4,3) + (1,3): 행 벡터가 아래로 4번 복제된 것처럼")
ax.set_xlim(-0.2, 11.2); ax.set_ylim(-0.3, 4.3)
# (b) (4,3) + (4,1)
ax = axes[1]
draw_grid(ax, 0, 4, 4, 3, "C0", 0.35, texts=lambda r, c: "A")
ax.text(3.5, 2, "+", fontsize=18, ha="center", va="center")
draw_grid(ax, 4, 4, 4, 3, "C1", 0.6, texts=lambda r, c: f"b{r + 1}", ghost_from=lambda r, c: c == 0)
ax.text(7.5, 2, "=", fontsize=18, ha="center", va="center")
draw_grid(ax, 8, 4, 4, 3, "C2", 0.35, texts=lambda r, c: f"A+b{r + 1}", fs=8)
ax.set_title("② (4,3) + (4,1): 열 벡터가 옆으로 3번 복제된 것처럼 — 신경망의 Z = WA + b 가 이 경우")
ax.set_xlim(-0.2, 11.2); ax.set_ylim(-0.3, 4.3)
# (c) (5,1) + (1,5) 함정
ax = axes[2]
draw_grid(ax, 0, 5, 5, 1, "C0", 0.6, texts=lambda r, c: f"u{r + 1}")
ax.text(1.5, 2.5, "+", fontsize=18, ha="center", va="center")
draw_grid(ax, 2, 5, 1, 5, "C1", 0.6, texts=lambda r, c: f"v{c + 1}")
ax.text(7.5, 2.5, "=", fontsize=18, ha="center", va="center")
draw_grid(ax, 8, 5, 5, 5, "C3", 0.3, texts=lambda r, c: f"u{r + 1}+v{c + 1}", fs=7)
ax.set_title("③ (5,1) + (1,5): 양쪽 다 늘어나서 (5,5)! 에러 없이 조용히 생기는 버그", color="darkred")
ax.set_xlim(-0.2, 13.2); ax.set_ylim(-0.3, 5.3)
for ax in axes:
    ax.set_aspect("equal")
    ax.axis("off")
fig.text(0.5, 0.06, "점선 칸 = 실제로 메모리에 복사되는 게 아니라 numpy가 '복제된 것처럼' 계산해주는 칸", ha="center", fontsize=10)
save(fig, "ch1-broadcasting.png")

# ---------------------------------------------------------------------------
# 6. Decision boundary: 로지스틱 회귀 vs 1-hidden-layer NN (Lab 2와 같은 꽃잎 데이터)
# ---------------------------------------------------------------------------


def make_flower(m=400, seed=1):
    r_ = np.random.RandomState(seed)
    N = m // 2
    X = np.zeros((m, 2)); Y = np.zeros((m, 1))
    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + r_.randn(N) * 0.2
        r = 4 * np.sin(4 * t) + r_.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j
    return X.T, Y.T


X, Y = make_flower()
m = X.shape[1]

# (a) 로지스틱 회귀
w = np.zeros((2, 1)); b = 0.0
for _ in range(5000):
    A = sigmoid(w.T @ X + b)
    dZ = A - Y
    w -= 0.1 * (X @ dZ.T) / m
    b -= 0.1 * dZ.sum() / m
acc_lr = ((sigmoid(w.T @ X + b) > 0.5) == Y).mean()

# (b) 2-layer NN: tanh hidden 4개 + sigmoid output
n1 = 4
r_init = np.random.RandomState(3)
W1 = r_init.randn(n1, 2) * 0.01; b1 = np.zeros((n1, 1))
W2 = r_init.randn(1, n1) * 0.01; b2 = np.zeros((1, 1))
for _ in range(10000):
    Z1 = W1 @ X + b1; A1 = np.tanh(Z1)
    A2 = sigmoid(W2 @ A1 + b2)
    dZ2 = A2 - Y
    dW2 = dZ2 @ A1.T / m; db2 = dZ2.sum(axis=1, keepdims=True) / m
    dZ1 = (W2.T @ dZ2) * (1 - A1 ** 2)
    dW1 = dZ1 @ X.T / m; db1 = dZ1.sum(axis=1, keepdims=True) / m
    W1 -= 1.2 * dW1; b1 -= 1.2 * db1; W2 -= 1.2 * dW2; b2 -= 1.2 * db2


def nn_pred(P):
    return sigmoid(W2 @ np.tanh(W1 @ P + b1) + b2)


acc_nn = ((nn_pred(X) > 0.5) == Y).mean()

gx, gy = np.meshgrid(np.linspace(-5, 5, 300), np.linspace(-5, 5, 300))
G = np.vstack([gx.ravel(), gy.ravel()])
fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
for ax, prob, title in [
    (axes[0], sigmoid(w.T @ G + b), f"로지스틱 회귀 (뉴런 1개)\n경계는 직선뿐 — 정확도 {acc_lr:.0%}"),
    (axes[1], nn_pred(G), f"hidden layer 1개 (tanh 4 units)\n직선 여러 개를 조합한 비선형 경계 — 정확도 {acc_nn:.0%}"),
]:
    ax.contourf(gx, gy, prob.reshape(gx.shape), levels=[0, 0.5, 1], colors=["#f4c7c3", "#c6dbef"], alpha=0.8)
    ax.contour(gx, gy, prob.reshape(gx.shape), levels=[0.5], colors="k", linewidths=1.5)
    ax.scatter(X[0], X[1], c=np.where(Y[0] == 1, "C0", "C3"), s=12, edgecolors="none")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal"); ax.grid(False)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
fig.suptitle("Lab 2의 꽃잎 데이터 — 파란 점 = y=1, 빨간 점 = y=0 — 검은 선이 decision boundary", y=1.02)
save(fig, "ch1-decision-boundary.png")
print(f"  logistic acc={acc_lr:.3f}, nn acc={acc_nn:.3f}")

# ---------------------------------------------------------------------------
# 7. Learning rate별 cost 곡선 (로지스틱 회귀, 스케일 안 맞춘 2D 데이터)
# ---------------------------------------------------------------------------
m = 200
X = np.vstack([rng.normal(0, 1, m), rng.normal(0, 1, m)])
Y = ((X[0] + 0.5 * X[1] + rng.normal(0, 0.7, m)) > 0).astype(float).reshape(1, m)
X = X * np.array([[3.0], [1.0]])   # 첫 feature 스케일을 키워서 α가 크면 튀게 만든다


def run(alpha, iters=150):
    w = np.zeros((2, 1)); b = 0.0; costs = []
    for _ in range(iters):
        A = np.clip(sigmoid(w.T @ X + b), 1e-12, 1 - 1e-12)
        costs.append(-np.mean(Y * np.log(A) + (1 - Y) * np.log(1 - A)))
        dZ = A - Y
        w -= alpha * (X @ dZ.T) / m
        b -= alpha * dZ.sum() / m
    return np.array(costs)


fig, ax = plt.subplots(figsize=(7.5, 4.5))
for alpha, lab in [(0.003, "$\\alpha$=0.003 (너무 작음: 느리다)"), (0.1, "$\\alpha$=0.1 (적당)"), (5.0, "$\\alpha$=5 (너무 큼: 튄다)")]:
    c = run(alpha)
    ax.plot(c, lw=2, label=lab)
    print(f"  alpha={alpha}: cost first={c[0]:.3f} last={c[-1]:.3f} max={c.max():.3f}")
ax.set_xlabel("iteration")
ax.set_ylabel("cost J")
ax.set_ylim(0, 1.6)
ax.set_title("learning rate만 바꿔서 cost vs iteration을 겹쳐 그리기")
ax.legend()
save(fig, "ch1-learning-rate-curves.png")
