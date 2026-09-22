"""
chapter5-sequence-models.md 용 시각자료 생성 스크립트.
외부 데이터/다운로드 없이 numpy로 손으로 만든 toy 데이터만 사용한다.

실행: python3 DL/assets/ch5-generate-plots.py   (이 파일과 같은 폴더에 PNG 8개 생성)

생성 파일:
  ch5-vanishing-gradient.png   : W_aa = s*I 를 반복해서 곱할 때 gradient 크기 (s=0.5/0.9/1.0/1.5), clipping 선
  ch5-gate-retention.png       : gate 값 g를 t번 곱했을 때 남는 기억의 비율 (0.5/0.9/0.99/0.99995)
  ch5-embedding-analogy.png    : 2D toy 임베딩 + man->woman, king->queen 평행 화살표, king-man+woman 위치
  ch5-beam-search-tree.png     : Lab 4 확률표 위에서 B=3 beam search가 남기는/버리는 후보 트리
  ch5-length-normalization.png : 짧은 문장 vs 긴 문장의 score (alpha = 0 / 0.7 / 1)
  ch5-attention-heatmap.png    : 번역 attention 가중치 alpha<t,t'> + 3토큰 self-attention 가중치
  ch5-sqrt-dk.png              : q.k 분포(d_k=64) 스케일링 전/후 + softmax가 한 곳에 몰리는 효과
  ch5-positional-encoding.png  : sin/cos positional encoding 히트맵 + 차원별 sin 곡선
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
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"   # log 축 눈금(10^-5 등)의 마이너스 기호 깨짐 방지
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

OUT = os.path.dirname(os.path.abspath(__file__))
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY = "#8a8985"
INK = "#0b0b0b"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def softmax(z, axis=-1):
    e = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# 1. Vanishing / exploding gradient
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.8))
steps = np.arange(0, 51)
for color, s in zip(C, [0.5, 0.9, 1.0, 1.5]):
    norms = np.sqrt(5) * s ** steps          # g0 = ones(5), W = s*I -> |g_k| = sqrt(5) s^k
    ax.semilogy(steps, norms, color=color, lw=2)
    ax.text(51, norms[-1], f"  s = {s}", color=INK, va="center", fontsize=10)
clip_val = 5.0
ax.axhline(clip_val, color=C[7], ls="--", lw=1.2)
ax.text(1, clip_val * 1.8, "clipping threshold (norm 5): 이 위로는 잘라낸다", color=C[7], fontsize=9)
ax.set_xlim(0, 58)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"1e{int(round(np.log10(v)))}"))   # mathtext 마이너스 깨짐 회피
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("거꾸로 거슬러 올라간 타임스텝 수 k")
ax.set_ylabel("gradient 크기 ||g|| (log 축)")
ax.set_title("BPTT: 같은 W_aa(= s·I)를 k번 곱하면 gradient가 지수적으로 줄거나 커진다")
save(fig, "ch5-vanishing-gradient.png")

# ---------------------------------------------------------------------------
# 2. Gate retention
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.8))
t = np.arange(0, 101)
gates = [(0.5, "Γ = 0.5"), (0.9, "Γ = 0.9"), (0.99, "Γ = 0.99"), (1 / (1 + np.exp(-10)), "Γ = σ(10) ≈ 0.99995")]
label_y = [0.03, 0.12, None, None]   # 0 근처에 겹치는 두 라벨은 위치를 손으로 벌림
for color, (g, label), ly in zip(C, gates, label_y):
    ax.plot(t, g ** t, color=color, lw=2)
    ax.text(101, g ** 100 if ly is None else ly, f"  {label}: 100 step 뒤 {g ** 100:.3g}",
            color=INK, va="center", fontsize=9.5)
    if ly is not None:
        ax.plot([100, 102], [g ** 100, ly], color=color, lw=1)
ax.set_xlim(0, 150)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_ylim(-0.03, 1.05)
ax.set_xlabel("타임스텝 t")
ax.set_ylabel("처음 기억 중 남아있는 비율  Γ^t")
ax.set_title("기억 유지 비율 Γ (GRU의 1-Γu, LSTM의 Γf)를 t번 곱했을 때 남는 기억")
save(fig, "ch5-gate-retention.png")

# ---------------------------------------------------------------------------
# 3. Embedding analogy (2D toy)
# ---------------------------------------------------------------------------
words = {
    "man": (1.0, 0.0), "woman": (-1.0, 0.0),
    "king": (0.95, 0.93), "queen": (-0.97, 0.95),
    "boy": (0.9, -0.35), "girl": (-0.9, -0.35),
    "apple": (0.05, -0.95), "orange": (0.12, -1.0),
}
fig, ax = plt.subplots(figsize=(7.2, 6))
for w, (x, y) in words.items():
    ax.scatter(x, y, s=60, color=C[0], zorder=3, edgecolor="white", linewidth=1.5)
    ax.annotate(w, (x, y), textcoords="offset points", xytext=(7, 6), fontsize=11, color=INK)
arrow = dict(arrowstyle="->", lw=2)
ax.annotate("", xy=words["woman"], xytext=words["man"], arrowprops={**arrow, "color": C[1]})
ax.annotate("", xy=words["queen"], xytext=words["king"], arrowprops={**arrow, "color": C[1]})
ax.text(0, 0.07, "e_woman - e_man", color=C[1], ha="center", fontsize=10)
ax.text(0, 1.0, "e_queen - e_king", color=C[1], ha="center", fontsize=10)
target = np.array(words["king"]) - np.array(words["man"]) + np.array(words["woman"])
ax.scatter(*target, marker="*", s=260, color=C[3], zorder=4, edgecolor=INK, linewidth=0.6)
ax.annotate("king - man + woman\n= (-1.05, 0.93)", target, textcoords="offset points",
            xytext=(-20, -38), fontsize=10, color=INK, ha="center")
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.25, 1.3)
ax.set_xlabel("축 1 (대략 '성별' 느낌: + 남성 / - 여성)")
ax.set_ylabel("축 2 (대략 '왕족' 느낌)")
ax.set_title("Toy word embedding: 같은 관계 = 같은 방향의 차이 벡터")
ax.set_aspect("equal")
save(fig, "ch5-embedding-analogy.png")

# ---------------------------------------------------------------------------
# 4. Beam search tree (Lab 4 확률표, B=3)
# ---------------------------------------------------------------------------
TABLE = {
    (): {"jane": 0.9, "in": 0.1},
    ("jane",): {"is": 1.0},
    ("jane", "is"): {"going": 0.5, "visiting": 0.4, "in": 0.1},
    ("jane", "is", "going"): {"to": 0.6, "in": 0.4},
    ("jane", "is", "going", "to"): {"september": 0.55, "africa": 0.35, "<eos>": 0.1},
    ("jane", "is", "going", "in"): {"september": 0.3, "africa": 0.7},
    ("jane", "is", "visiting"): {"africa": 0.95, "september": 0.05},
    ("jane", "is", "visiting", "africa"): {"in": 0.9, "<eos>": 0.1},
    ("jane", "is", "visiting", "africa", "in"): {"september": 1.0},
}
def nwp(prefix):
    return TABLE.get(tuple(prefix), {"<eos>": 1.0})

B = 3
beams = [((), 1.0, False)]
pos = {(): (0, 0.0)}
levels = []   # per step: list of (seq, word, p, parent, kept)
for step in range(1, 7):
    cands = []
    for seq, p, done in beams:
        if done:
            cands.append((seq, p, True, None))
            continue
        for w, pr in nwp(seq).items():
            new = seq + (w,)
            cands.append((new, p * pr, w == "<eos>", seq))
    cands.sort(key=lambda c: -c[1])
    kept = cands[:B]
    new_nodes = [(c, i < B) for i, c in enumerate(cands) if c[3] is not None]
    levels.append(new_nodes)
    beams = [(s, p, d) for s, p, d, _ in kept]
    if all(d for _, _, d in beams):
        break

fig, ax = plt.subplots(figsize=(12, 6.2))
ax.grid(False)
ax.axis("off")
for step, nodes in enumerate(levels, start=1):
    n = len(nodes)
    # 부모 위치(위->아래) 순, 같은 부모 안에서는 확률 순으로 세로 배치 (선이 꼬이지 않게)
    nodes = sorted(nodes, key=lambda nd: (-pos[nd[0][3]][1], -nd[0][1]))
    for i, ((seq, p, done, parent), kept) in enumerate(nodes):
        y = (n - 1) / 2 - i
        pos[seq] = (step, y)
        px, py = pos[parent]
        ax.plot([px + 0.32, step - 0.32], [py, y], color=C[0] if kept else "#cfcdc6", lw=2 if kept else 1, zorder=1)
        face = "#e3eefb" if kept else "#f3f2ee"
        edge = C[0] if kept else "#cfcdc6"
        ax.text(step, y, f"{seq[-1]}\nP={p:.3g}", ha="center", va="center", fontsize=9.5,
                color=INK if kept else GRAY,
                bbox=dict(boxstyle="round,pad=0.35", fc=face, ec=edge, lw=1.5 if kept else 1), zorder=2)
ax.text(0, 0, "<start>", ha="center", va="center", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=INK))
for step in range(1, len(levels) + 1):
    ax.text(step, 3.3, f"step {step}", ha="center", fontsize=10, color=GRAY)
ax.set_xlim(-0.5, len(levels) + 0.5)
ax.set_ylim(-3.2, 3.6)
ax.set_title("Beam search (B=3): 파란 박스 = 누적 확률 상위 3개라 살아남은 후보, 회색 = 버려진 후보\n"
             "greedy는 step 3에서 going(0.45)만 따라가지만, beam은 visiting(0.36)도 살려둬서 step 4에서 역전된다",
             fontsize=11)
save(fig, "ch5-beam-search-tree.png")

# ---------------------------------------------------------------------------
# 5. Length normalization
# ---------------------------------------------------------------------------
cands = [("A: 3단어, 단어당 P=0.5", 3, 0.5), ("B: 6단어, 단어당 P=0.7", 6, 0.7)]
alphas = [0.0, 0.7, 1.0]
fig, ax = plt.subplots(figsize=(8, 4.6))
width = 0.36
x = np.arange(len(alphas))
for k, (name, T, p) in enumerate(cands):
    vals = [T * np.log(p) / T ** a for a in alphas]
    bars = ax.bar(x + (k - 0.5) * width, vals, width - 0.04, color=C[k], label=name)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v - 0.06, f"{v:.3f}", ha="center", va="top", fontsize=9.5, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(["α = 0\n(정규화 없음: A 승)", "α = 0.7\n(B 승)", "α = 1\n(단어당 평균: B 승)"])
ax.set_ylabel("score = (1/T^α) Σ log P   (0에 가까울수록 좋음)")
ax.set_ylim(-2.6, 0.1)
ax.axhline(0, color=INK, lw=0.8)
ax.legend(loc="lower right", frameon=False)
ax.set_title("Length normalization: 정규화 없으면 짧은 문장이 무조건 유리")
save(fig, "ch5-length-normalization.png")

# ---------------------------------------------------------------------------
# 6. Attention heatmap (번역 alpha) + 3-token self-attention
# ---------------------------------------------------------------------------
src = ["Jane", "visite", "l'Afrique", "en", "septembre"]
tgt = ["Jane", "visits", "Africa", "in", "September"]
E_scores = np.array([
    [3.0, 0.5, 1.0, -1.0, 0.0],
    [0.5, 3.0, 1.2, -0.5, -0.5],
    [0.0, 1.0, 3.5, 0.2, 0.0],
    [-0.5, -0.5, 0.5, 2.5, 1.5],
    [-1.0, -0.5, 0.0, 1.0, 3.5],
])
alpha = softmax(E_scores, axis=1)
Q = np.array([[1.0, 0], [0, 1], [1, 1]])
K = Q.copy()
W3 = softmax(Q @ K.T / np.sqrt(2), axis=1)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})
fig.subplots_adjust(wspace=0.45)
for ax, M, xl, yl, title in [
    (axes[0], alpha, src, tgt, "번역 attention α<t,t'> (행 = 출력 단어, 합=1)"),
    (axes[1], W3, ["토큰1", "토큰2", "토큰3"], ["토큰1", "토큰2", "토큰3"], "3토큰 self-attention 가중치 (본문 예제)"),
]:
    ax.grid(False)
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(xl)))
    ax.set_xticklabels(xl, rotation=30, ha="right")
    ax.set_yticks(range(len(yl)))
    ax.set_yticklabels(yl)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9.5,
                    color="white" if M[i, j] > 0.55 else INK)
    ax.set_title(title, fontsize=11)
axes[0].set_xlabel("입력(프랑스어) 단어 t'")
axes[0].set_ylabel("출력(영어) 단어 t")
axes[1].set_xlabel("Key (참고 대상)")
axes[1].set_ylabel("Query (지금 표현을 만드는 토큰)")
fig.colorbar(im, ax=axes, shrink=0.8, label="attention 가중치")
fig.savefig(os.path.join(OUT, "ch5-attention-heatmap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved ch5-attention-heatmap.png")

# ---------------------------------------------------------------------------
# 7. 왜 sqrt(d_k)로 나누나
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
d_k = 64
q = rng.standard_normal((20000, d_k))
k = rng.standard_normal((20000, d_k))
dots = (q * k).sum(axis=1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
bins = np.linspace(-35, 35, 90)
ax.hist(dots, bins=bins, color=C[1], alpha=0.75, label=f"q·k 그대로 (std ≈ {dots.std():.1f})")
ax.hist(dots / np.sqrt(d_k), bins=bins, color=C[0], alpha=0.75, label=f"q·k / √64 (std ≈ {(dots / np.sqrt(d_k)).std():.2f})")
ax.set_xlabel("attention score")
ax.set_ylabel("빈도")
ax.set_title("d_k = 64, 성분이 N(0,1)인 q, k의 내적 분포")
ax.legend(frameon=False, fontsize=9.5)
ax = axes[1]
z = np.array([8.0, -8.0, 4.0, 0.0])
w_raw, w_s = softmax(z), softmax(z / 8)
xx = np.arange(4)
b1 = ax.bar(xx - 0.2, w_raw, 0.38, color=C[1], label="softmax(score)")
b2 = ax.bar(xx + 0.2, w_s, 0.38, color=C[0], label="softmax(score / 8)")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.2f}", ha="center", fontsize=9)
ax.set_xticks(xx)
ax.set_xticklabels([f"key {i+1}\nscore {int(v)}" for i, v in enumerate(z)])
ax.set_ylim(0, 1.1)
ax.set_ylabel("attention 가중치")
ax.set_title("score가 크면 softmax가 한 곳에 몰림(≈one-hot) → gradient ≈ 0")
ax.legend(frameon=False, fontsize=9.5)
save(fig, "ch5-sqrt-dk.png")

# ---------------------------------------------------------------------------
# 8. Positional encoding
# ---------------------------------------------------------------------------
def positional_encoding(T, d):
    p = np.arange(T)[:, None]
    i = np.arange(d)[None, :]
    angle = p / np.power(10000, (2 * (i // 2)) / d)
    PE = np.zeros((T, d))
    PE[:, 0::2] = np.sin(angle[:, 0::2])
    PE[:, 1::2] = np.cos(angle[:, 1::2])
    return PE

T, d = 50, 64
PE = positional_encoding(T, d)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]
ax.grid(False)
im = ax.imshow(PE, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
ax.set_xlabel("임베딩 차원 index")
ax.set_ylabel("위치 pos")
ax.set_title("PE 행렬 (50 위치 × 64 차원): 한 행 = 그 위치의 '바코드'")
fig.colorbar(im, ax=ax, shrink=0.85)
ax = axes[1]
fine = np.linspace(0, T - 1, 500)
for color, dim in zip(C, [0, 8, 16, 32]):
    period = 2 * np.pi * 10000 ** (dim / d)
    # 선 = 연속 함수 sin(pos / 10000^(dim/d)), 점 = 실제로 쓰이는 정수 위치의 값
    ax.plot(fine, np.sin(fine / 10000 ** (dim / d)), color=color, lw=1.8, label=f"차원 {dim} (주기 ≈ {period:.0f})")
    ax.scatter(np.arange(T), PE[:, dim], color=color, s=10, zorder=3)
ax.set_xlim(0, T)
ax.legend(frameon=False, fontsize=9, loc="lower left", ncol=2)
ax.set_ylim(-1.45, 1.1)
ax.set_xlabel("위치 pos")
ax.set_ylabel("PE 값")
ax.set_title("짝수 차원(sin)만 뽑아봄: 앞 차원은 빠르게, 뒤 차원은 느리게 진동")
save(fig, "ch5-positional-encoding.png")
