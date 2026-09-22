"""
chapter4-convolutional-neural-networks.md (Course 4: CNN) 용 시각자료 생성 스크립트.

생성하는 PNG (모두 이 스크립트와 같은 폴더, 접두사 ch4-):
  ch4-conv-sliding.png       6x6 입력 * 3x3 필터 = 4x4 출력, 한 칸 계산을 숫자로 보여줌 (강의 숫자 그대로)
  ch4-edge-detection.png     밝→어 / 어→밝 이미지에 수직 edge 필터를 통과시킨 결과 heatmap (+30 / -30)
  ch4-padding-stride.png     same padding(p=1)으로 크기 유지, stride 2로 창이 건너뛰는 모습
  ch4-volume-conv.png        6x6x3 입력 * 3x3x3 필터 2개 = 4x4x2 (의사 3D 그림)
  ch4-pooling.png            4x4 입력에 2x2/s=2 max pooling vs average pooling
  ch4-iou.png                IoU 계산 예시 두 개 (0.6, 1/7)
  ch4-nms.png                Non-max suppression 전/후 (Lab 4와 같은 박스/점수)
  ch4-yolo-grid.png          3x3 grid에 객체 중심 배정 + 한 cell 안에서 b_x, b_y, b_h, b_w 읽는 법
  ch4-anchor-boxes.png       한 cell의 보행자/자동차를 IoU로 anchor 1/2에 배정
  ch4-transpose-conv.png     2x2 입력을 3x3 필터, stride 2 transpose conv로 5x5로 키우는 과정(겹치는 곳은 합)
  ch4-plain-vs-residual.png  블록 수에 따른 activation 크기: plain은 사라지고 residual은 유지 (Lab 3과 같은 설정)

실행: python3 DL/assets/ch4-generate-plots.py   (repo 루트에서; 어디서 실행해도 이 폴더에 저장됨)
의존성: numpy, matplotlib 만.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Rectangle, Polygon

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11

OUT = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, GREEN, RED, PURPLE, GRAY = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#666666"


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)


def draw_grid(ax, M, x0, y0, cell=1.0, fmt="{:g}", face=None, fontsize=11, title=None, lw=1.0):
    """행렬 M을 (x0, y0) 왼쪽 위 기준으로 칸+숫자로 그린다 (y축은 아래로 증가하게 invert 해서 씀)."""
    n_r, n_c = M.shape
    for i in range(n_r):
        for j in range(n_c):
            fc = "white" if face is None else face[i][j]
            ax.add_patch(Rectangle((x0 + j * cell, y0 + i * cell), cell, cell, fc=fc, ec="black", lw=lw))
            if M[i, j] is not None and not (isinstance(M[i, j], float) and np.isnan(M[i, j])):
                ax.text(x0 + (j + 0.5) * cell, y0 + (i + 0.5) * cell, fmt.format(M[i, j]),
                        ha="center", va="center", fontsize=fontsize)
    if title:
        ax.text(x0 + n_c * cell / 2, y0 - 0.45 * cell, title, ha="center", va="bottom", fontsize=fontsize + 1)


def conv2d(x, k, s=1):
    f = k.shape[0]
    n = (x.shape[0] - f) // s + 1
    return np.array([[np.sum(x[i * s:i * s + f, j * s:j * s + f] * k) for j in range(n)] for i in range(n)])


V_FILTER = np.array([[1, 0, -1]] * 3)

# ---------------------------------------------------------------------------
# 1. convolution sliding (강의 첫 예제 숫자)
# ---------------------------------------------------------------------------
X = np.array([[3, 0, 1, 2, 7, 4], [1, 5, 8, 9, 3, 1], [2, 7, 2, 5, 1, 3],
              [0, 1, 3, 1, 7, 8], [4, 2, 1, 6, 2, 8], [2, 4, 5, 2, 3, 9]])
Y = conv2d(X, V_FILTER)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for k, (r0, c0, col) in enumerate([(0, 0, "#FDE2C8"), (0, 1, "#D6E4F5")]):
    ax = axes[k]
    face_x = [["white"] * 6 for _ in range(6)]
    for i in range(3):
        for j in range(3):
            face_x[r0 + i][c0 + j] = col
    draw_grid(ax, X, 0, 0, face=face_x, title="입력 6x6")
    ax.text(6.6, 3.0, "*", fontsize=26, ha="center", va="center")
    draw_grid(ax, V_FILTER, 7.2, 1.5, face=[[col] * 3 for _ in range(3)], title="필터 3x3")
    ax.text(10.8, 3.0, "=", fontsize=24, ha="center", va="center")
    face_y = [["white"] * 4 for _ in range(4)]
    face_y[r0][c0] = col
    draw_grid(ax, Y, 11.4, 1.0, face=face_y, title="출력 4x4")
    win = X[r0:r0 + 3, c0:c0 + 3]
    terms = " + ".join(f"({win[i, 0]}-{win[i, 2]})" for i in range(3))
    ax.text(8.0, 6.9, f"창 위치 ({r0},{c0}): {terms} = {Y[r0, c0]}", ha="center", fontsize=11,
            bbox=dict(fc=col, ec="none", boxstyle="round,pad=0.3"))
    ax.set_xlim(-0.3, 15.7)
    ax.set_ylim(7.6, -1.0)
    ax.set_aspect("equal")
    ax.axis("off")
axes[0].set_title("① 왼쪽 위 3x3 창: 원소별 곱 후 전부 더함", fontsize=12)
axes[1].set_title("② 한 칸 오른쪽으로 이동해서 같은 계산 반복", fontsize=12)
save(fig, "ch4-conv-sliding.png")

# ---------------------------------------------------------------------------
# 2. edge detection heatmap (밝→어, 어→밝)
# ---------------------------------------------------------------------------
img1 = np.zeros((6, 6)); img1[:, :3] = 10
img2 = img1[:, ::-1].copy()
fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
for row, img in enumerate([img1, img2]):
    out = conv2d(img, V_FILTER)
    panels = [(img, "입력 (10=밝음, 0=어두움)", "gray", 0, 10),
              (V_FILTER, "수직 edge 필터", "gray", -1, 1),
              (out, "출력", "RdBu_r", -30, 30)]
    for c, (M, t, cmap, vmin, vmax) in enumerate(panels):
        ax = axes[row, c]
        ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M[i, j]
                if c == 0:
                    color = "black" if val > 5 else "white"
                elif c == 1:
                    color = "white" if val < 0 else "black"
                else:
                    color = "white" if abs(val) > 20 else "black"
                ax.text(j, i, f"{val:g}", ha="center", va="center", color=color, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(t, fontsize=11)
axes[0, 0].set_ylabel("밝음 → 어두움", fontsize=12)
axes[1, 0].set_ylabel("어두움 → 밝음", fontsize=12)
fig.suptitle("경계가 있는 가운데 두 열만 ±30, 나머지는 0 — 부호가 edge 방향을 알려준다", fontsize=12)
fig.tight_layout()
save(fig, "ch4-edge-detection.png")

# ---------------------------------------------------------------------------
# 3. padding & stride
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
ax = axes[0]
n, p, f = 4, 1, 3
M = np.full((n + 2 * p, n + 2 * p), np.nan, dtype=object)
face = [["#E6E6E6"] * (n + 2 * p) for _ in range(n + 2 * p)]
rng = np.random.RandomState(3)
for i in range(n + 2 * p):
    for j in range(n + 2 * p):
        if p <= i < n + p and p <= j < n + p:
            M[i, j] = int(rng.randint(1, 10)); face[i][j] = "white"
        else:
            M[i, j] = 0
draw_grid(ax, M, 0, 0, face=face, title="4x4 입력 + zero padding p=1 → 6x6")
ax.add_patch(Rectangle((0, 0), 3, 3, fill=False, ec=RED, lw=3))
ax.add_patch(Rectangle((3, 3), 3, 3, fill=False, ec=BLUE, lw=3, ls="--"))
draw_grid(ax, np.full((4, 4), None), 7.3, 1.0, face=[["#FADBD8" if (i, j) == (0, 0) else ("#D6E4F5" if (i, j) == (3, 3) else "white") for j in range(4)] for i in range(4)], title="출력 4x4 (same)")
ax.text(3.0, 6.4, "회색 = 0으로 채운 테두리. 모서리 픽셀도 이제 여러 창에 참여한다\n"
                  r"$(n+2p-f)/s+1 = (4+2-3)/1+1 = 4$  → 입력과 같은 크기", ha="center", va="top", fontsize=11)
ax.set_xlim(-0.3, 11.6); ax.set_ylim(8.3, -1.0); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Same padding", fontsize=13)

ax = axes[1]
n, f, s = 7, 3, 2
draw_grid(ax, np.full((n, n), None), 0, 0, title="7x7 입력, f=3, s=2")
cols = [RED, GREEN, PURPLE]
for k, c0 in enumerate([0, 2, 4]):
    off = 0.08 * k
    ax.add_patch(Rectangle((c0 + off, 0 + off), 3 - 2 * off, 3 - 2 * off, fill=False, ec=cols[k], lw=3))
ax.add_patch(Rectangle((0.2, 2.2), 2.6, 2.6, fill=False, ec=ORANGE, lw=3, ls="--"))
face = [["white"] * 3 for _ in range(3)]
face[0][0], face[0][1], face[0][2], face[1][0] = "#FADBD8", "#D5EFD9", "#E4DFF3", "#FBE3CF"
draw_grid(ax, np.full((3, 3), None), 8.3, 2.0, face=face, title="출력 3x3")
ax.text(3.5, 7.4, "창이 2칸씩 건너뛴다: 가로 시작 위치 0, 2, 4 (6은 필터가 밖으로 나감)\n"
                  r"$\lfloor (7+0-3)/2 \rfloor + 1 = 3$" "\n(주황 점선: 첫 줄이 끝나면 아래로도 2칸 내려감)", ha="center", va="top", fontsize=11)
ax.set_xlim(-0.3, 11.6); ax.set_ylim(9.2, -1.0); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Stride 2", fontsize=13)
save(fig, "ch4-padding-stride.png")

# ---------------------------------------------------------------------------
# 4. volume conv (의사 3D)
# ---------------------------------------------------------------------------
def slab(ax, x0, y0, n_rows, n_cols, depth, cell, colors, label, dx=0.35, dy=0.35):
    """depth장의 2D grid를 비스듬히 겹쳐서 3D 볼륨처럼 보이게 그림. 뒤쪽 장부터 그린다."""
    for d in reversed(range(depth)):
        ox, oy = x0 + d * dx, y0 + d * dy
        for i in range(n_rows):
            for j in range(n_cols):
                ax.add_patch(Rectangle((ox + j * cell, oy + i * cell), cell, cell,
                                       fc=colors[d], ec="black", lw=0.6, alpha=0.95))
    ax.text(x0 + (n_cols * cell + (depth - 1) * dx) / 2, y0 - 0.5, label, ha="center", fontsize=11)


fig, ax = plt.subplots(figsize=(13, 5))
slab(ax, 0, 0, 6, 6, 3, 0.6, ["#F4B6B6", "#B9E0B9", "#B6C8F4"], "입력 6x6x3 (R,G,B)")
ax.add_patch(Rectangle((0, 0), 1.8, 1.8, fill=False, ec="black", lw=3))
ax.text(5.0, 2.0, "*", fontsize=26, ha="center", va="center")
slab(ax, 5.8, 0.9, 3, 3, 3, 0.6, ["#F9D5D5", "#D6EED6", "#D5DFF9"], "필터 1 (3x3x3)")
slab(ax, 5.8, 4.6, 3, 3, 3, 0.6, ["#FFE3C2", "#FFE3C2", "#FFE3C2"], "필터 2 (3x3x3)")
ax.text(9.3, 2.0, "=", fontsize=24, ha="center", va="center")
slab(ax, 10.2, 0.2, 4, 4, 1, 0.6, ["#D5DFF9"], "필터1 결과 4x4")
slab(ax, 10.2, 3.9, 4, 4, 1, 0.6, ["#FFE3C2"], "필터2 결과 4x4")
ax.text(13.6, 2.6, "쌓기 →", fontsize=12, ha="center", va="center")
slab(ax, 14.4, 1.4, 4, 4, 2, 0.6, ["#D5DFF9", "#FFE3C2"], "출력 4x4x2")
ax.text(8.5, 8.5, "필터 한 개 = 27개 숫자. 겹치는 27칸을 곱해서 다 더하면 스칼라 1개 → 필터 하나당 출력 채널 1장.\n"
                  "필터의 깊이(3)는 입력 채널 수에 맞추고, 필터 개수(2)가 출력 채널 수가 된다.",
        ha="center", fontsize=11)
ax.set_xlim(-0.5, 17.5); ax.set_ylim(9.3, -1.0); ax.set_aspect("equal"); ax.axis("off")
save(fig, "ch4-volume-conv.png")

# ---------------------------------------------------------------------------
# 5. pooling
# ---------------------------------------------------------------------------
P = np.array([[1, 3, 2, 1], [2, 9, 1, 1], [1, 3, 2, 3], [5, 6, 1, 2]])
qc = [["#FADBD8", "#D6E4F5"], ["#D5EFD9", "#FBE3CF"]]
face = [[qc[i // 2][j // 2] for j in range(4)] for i in range(4)]
mx = P.reshape(2, 2, 2, 2).max(axis=(1, 3))
av = P.reshape(2, 2, 2, 2).mean(axis=(1, 3))
fig, ax = plt.subplots(figsize=(11, 4.6))
draw_grid(ax, P, 0, 0, face=face, title="입력 4x4 (f=2, s=2 → 영역 4개)")
for i in range(4):
    for j in range(4):
        if P[i, j] == mx[i // 2, j // 2]:
            ax.add_patch(Rectangle((j + 0.1, i + 0.1), 0.8, 0.8, fill=False, ec="black", lw=2.2))
ax.annotate("", xy=(6.0, 1.3), xytext=(4.3, 1.8), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(6.0, 3.9), xytext=(4.3, 2.4), arrowprops=dict(arrowstyle="->", lw=1.5))
draw_grid(ax, mx, 6.2, 0.3, face=qc, title="Max pooling")
draw_grid(ax, av, 6.2, 3.0, face=qc, title="", fmt="{:g}")
ax.text(7.2, 5.4, "Average pooling", ha="center", fontsize=12)
ax.text(11.8, 2.3, "굵은 테두리 = 각 영역의 max\n\n채널마다 따로 적용\n→ n_c는 그대로\n\n학습 파라미터 0개", ha="center",
        va="center", fontsize=11)
ax.set_xlim(-0.3, 14); ax.set_ylim(5.9, -1.0); ax.set_aspect("equal"); ax.axis("off")
save(fig, "ch4-pooling.png")

# ---------------------------------------------------------------------------
# 6. IoU
# ---------------------------------------------------------------------------
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter, union, inter / union


fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
for ax, (A, B, title) in zip(axes, [((0, 0, 4, 4), (1, 0, 5, 4), "많이 겹침"),
                                    ((2, 1, 4, 3), (1, 2, 3, 4), "조금 겹침 (Lab 4 첫 예제)")]):
    inter, union, v = iou(A, B)
    ax.add_patch(Rectangle((A[0], A[1]), A[2] - A[0], A[3] - A[1], fc=BLUE, alpha=0.25, ec=BLUE, lw=2.5, label="정답 box"))
    ax.add_patch(Rectangle((B[0], B[1]), B[2] - B[0], B[3] - B[1], fc=ORANGE, alpha=0.25, ec=ORANGE, lw=2.5, label="예측 box"))
    ix0, iy0 = max(A[0], B[0]), max(A[1], B[1]); ix1, iy1 = min(A[2], B[2]), min(A[3], B[3])
    ax.add_patch(Rectangle((ix0, iy0), ix1 - ix0, iy1 - iy0, fc="none", ec=RED, lw=2, hatch="///", label="교집합"))
    ax.set_xlim(-0.5, 5.5); ax.set_ylim(4.5, -0.5); ax.set_aspect("equal")
    ax.set_xticks(range(0, 6)); ax.set_yticks(range(0, 5)); ax.grid(True, alpha=0.3)
    aA = (A[2] - A[0]) * (A[3] - A[1]); aB = (B[2] - B[0]) * (B[3] - B[1])
    verdict = "0.5 이상 → correct" if v >= 0.5 else "0.5 미만 → 틀림"
    ax.set_title(f"{title}\n교집합 {inter:g}, 합집합 {aA}+{aB}-{inter:g}={union:g}\nIoU = {inter:g}/{union:g} = {v:.3f}  ({verdict})", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=10)
save(fig, "ch4-iou.png")

# ---------------------------------------------------------------------------
# 7. NMS (Lab 4 데이터)
# ---------------------------------------------------------------------------
boxes = np.array([[50, 50, 150, 150], [55, 45, 155, 145], [60, 60, 160, 170],
                  [200, 80, 280, 200], [205, 85, 285, 195], [100, 200, 130, 230]], dtype=float)
scores = np.array([0.9, 0.75, 0.8, 0.7, 0.85, 0.4])


def nms(boxes, scores, st=0.6, it=0.5):
    idx = [i for i in np.argsort(-scores) if scores[i] >= st]
    keep = []
    while idx:
        best = idx.pop(0); keep.append(best)
        idx = [i for i in idx if iou(boxes[best], boxes[i])[2] < it]
    return keep


keep = nms(boxes, scores)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
for k, ax in enumerate(axes):
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        if k == 0:
            color, lw, ls = (GRAY if scores[i] < 0.6 else BLUE), 2, ("--" if scores[i] < 0.6 else "-")
        else:
            if i not in keep:
                continue
            color, lw, ls = RED, 3, "-"
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=color, lw=lw, ls=ls))
        # 라벨 위치: 박스가 겹쳐서 글자가 겹치지 않도록 박스마다 따로 지정
        tx, ty, ha = {0: (x1 - 4, (y1 + y2) / 2, "right"), 1: (x2, y1 - 4, "right"), 2: (x2, y2 + 14, "right"),
                      3: (x1, y2 + 14, "left"), 4: (x2 + 4, (y1 + y2) / 2, "left"), 5: (x1, y1 - 4, "left")}[i]
        ax.text(tx, ty, f"#{i}  {scores[i]:.2f}", color=color, fontsize=10, ha=ha)
    ax.set_xlim(-25, 345); ax.set_ylim(260, 20); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
axes[0].set_title("NMS 전: 물체 2개에 box 6개\n(점선 #5는 score 0.6 미만이라 1단계에서 탈락)", fontsize=11)
axes[1].set_title(f"NMS 후: 남은 box {keep}\n#0(0.9)이 #1(IoU 0.82), #2(0.63)를 지우고, #4(0.85)가 #3(0.81)을 지움", fontsize=11)
save(fig, "ch4-nms.png")

# ---------------------------------------------------------------------------
# 8. YOLO grid
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
ax = axes[0]
for t in range(4):
    ax.plot([t, t], [0, 3], color="black", lw=1); ax.plot([0, 3], [t, t], color="black", lw=1)
objs = [("자동차 1", (0.2, 1.3, 1.1, 1.9), BLUE), ("자동차 2", (1.75, 1.35, 2.85, 1.95), ORANGE)]
for name, (x1, y1, x2, y2), c in objs:
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fc=c, alpha=0.3, ec=c, lw=2.5))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ax.plot(cx, cy, "o", color=c, ms=9)
    ax.add_patch(Rectangle((int(cx), int(cy)), 1, 1, fill=False, ec=c, lw=3.5, ls="--"))
    ax.text(cx, y1 - 0.08, name, color=c, ha="center", fontsize=11)
ax.text(1.5, 3.35, "점 = box 중심. 중심이 들어있는 cell(점선)만 그 물체를 책임진다.\n자동차 1은 두 cell에 걸쳐 있어도 중심이 있는 왼쪽 가운데 cell 담당.",
        ha="center", va="top", fontsize=10)
ax.set_xlim(-0.1, 3.1); ax.set_ylim(3.9, -0.1); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("3x3 grid: 출력 3x3x8 (anchor 없이, 클래스 3개)", fontsize=12)

ax = axes[1]
ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec="black", lw=2))
bx, by, bh, bw = 0.4, 0.3, 0.5, 0.9
ax.add_patch(Rectangle((bx - bw / 2, by - bh / 2), bw, bh, fc=ORANGE, alpha=0.3, ec=ORANGE, lw=2.5))
ax.plot(bx, by, "o", color=ORANGE, ms=9)
ax.annotate("", xy=(bx, -0.08), xytext=(0, -0.08), arrowprops=dict(arrowstyle="<->", color=RED))
ax.text(bx / 2, -0.12, r"$b_x=0.4$", color=RED, ha="center", fontsize=11)
ax.annotate("", xy=(1.08, by), xytext=(1.08, 0), arrowprops=dict(arrowstyle="<->", color=RED))
ax.text(1.12, by / 2, r"$b_y=0.3$", color=RED, va="center", fontsize=11)
ax.annotate("", xy=(bx + bw / 2, 0.62), xytext=(bx - bw / 2, 0.62), arrowprops=dict(arrowstyle="<->", color=PURPLE))
ax.text(bx, 0.7, r"$b_w=0.9$", color=PURPLE, ha="center", fontsize=11)
ax.annotate("", xy=(-0.12, by + bh / 2), xytext=(-0.12, by - bh / 2), arrowprops=dict(arrowstyle="<->", color=PURPLE))
ax.text(-0.16, by, r"$b_h=0.5$", color=PURPLE, ha="right", va="center", fontsize=11)
ax.text(0.5, 1.05, "cell 한 칸 = 한 변 1로 놓고 잰다.\n"
                   r"$b_x, b_y$: cell 왼쪽 위 기준 중심 위치 → 항상 0~1" "\n"
                   r"$b_h, b_w$: cell 크기 대비 비율 → 1을 넘을 수 있음" "\n"
                   r"$y = [1,\ 0.4,\ 0.3,\ 0.5,\ 0.9,\ 0,\ 1,\ 0]$ (자동차 = $c_2$)", ha="center", va="top", fontsize=11)
ax.set_xlim(-0.6, 1.5); ax.set_ylim(1.6, -0.3); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("한 cell 확대: 좌표를 cell 기준으로 표현", fontsize=12)
save(fig, "ch4-yolo-grid.png")

# ---------------------------------------------------------------------------
# 9. Anchor boxes
# ---------------------------------------------------------------------------
def ciou(w1, h1, w2, h2):
    inter = min(w1, w2) * min(h1, h2)
    return inter / (w1 * h1 + w2 * h2 - inter)


A1, A2 = (0.3, 0.8), (0.9, 0.4)
ped, car = (0.25, 0.7), (0.8, 0.45)
fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
ax = axes[0]
ax.add_patch(Rectangle((-0.5, -0.5), 1, 1, fill=False, ec="black", lw=1.5, ls=":"))
for (w, h), c, name in [(ped, GREEN, "보행자"), (car, BLUE, "자동차")]:
    ax.add_patch(Rectangle((-w / 2, -h / 2), w, h, fc=c, alpha=0.25, ec=c, lw=2.5, label=name))
ax.plot(0, 0, "ko")
ax.set_title("한 cell에 중심이 겹친 물체 2개\n(box 하나짜리 y로는 둘 다 못 적음)", fontsize=11)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=10, ncol=2)
ax.text(0.45, -0.45, "점선 = grid cell 한 칸", fontsize=9, ha="right")
ax2 = axes[1]
for (w, h), c, name in [(A1, RED, "anchor 1 (세로로 김)"), (A2, PURPLE, "anchor 2 (가로로 김)")]:
    ax2.add_patch(Rectangle((-w / 2, -h / 2), w, h, fill=False, ec=c, lw=2.5, ls="--", label=name))
ax2.set_title("미리 정해둔 모양 2개\n(실제로는 k-means 등으로 정함)", fontsize=11)
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=10)
ax3 = axes[2]
ax3.axis("off")
rows = [["", "anchor 1", "anchor 2", "배정"],
        ["보행자", f"{ciou(*ped, *A1):.2f}", f"{ciou(*ped, *A2):.2f}", "anchor 1"],
        ["자동차", f"{ciou(*car, *A1):.2f}", f"{ciou(*car, *A2):.2f}", "anchor 2"]]
tb = ax3.table(cellText=rows, loc="center", cellLoc="center")
tb.scale(1, 2.4); tb.set_fontsize(12)
ax3.text(0.5, 0.82, "IoU(중심 맞춰서 모양만 비교) → 큰 쪽에 배정\n→ y 앞 8칸은 보행자, 뒤 8칸은 자동차", ha="center", fontsize=11, transform=ax3.transAxes)
for ax in axes[:2]:
    ax.set_xlim(-0.6, 0.6); ax.set_ylim(0.6, -0.6); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
save(fig, "ch4-anchor-boxes.png")

# ---------------------------------------------------------------------------
# 10. Transpose convolution
# ---------------------------------------------------------------------------
inp = np.array([[2, 1], [3, 2]])
K = np.ones((3, 3), dtype=int)
s, f = 2, 3
n_out = (2 - 1) * s + f
out = np.zeros((n_out, n_out), dtype=int)
contrib = []
for i in range(2):
    for j in range(2):
        out[i * s:i * s + f, j * s:j * s + f] += inp[i, j] * K
        contrib.append((i, j))
fig, ax = plt.subplots(figsize=(13, 5.4))
cc = [["#FADBD8", "#D6E4F5"], ["#D5EFD9", "#FBE3CF"]]
draw_grid(ax, inp, 0, 1.5, face=cc, title="입력 2x2")
draw_grid(ax, K, 3.2, 1.0, title="필터 3x3 (전부 1)")
ax.text(2.6, 2.5, "transpose\nconv", fontsize=9, ha="center", va="center")
face = [["white"] * n_out for _ in range(n_out)]
cover = [[[] for _ in range(n_out)] for _ in range(n_out)]
for (i, j) in contrib:
    for a in range(f):
        for b in range(f):
            cover[i * s + a][j * s + b].append(cc[i][j])
for a in range(n_out):
    for b in range(n_out):
        face[a][b] = cover[a][b][0] if len(cover[a][b]) == 1 else "#D0D0D0"
ax.text(7.0, 2.5, "=", fontsize=22, ha="center", va="center")
draw_grid(ax, out, 7.8, 0.0, face=face, title="출력 5x5 (s=2, p=0)")
ax.add_patch(Rectangle((8.8, 1.0), 3, 3, fill=False, ec=RED, lw=2.5, ls="--"))
ax.text(14.2, 2.5, "각 입력 값 x 필터 를\n출력에 stride 2 간격으로 '도장' 찍음.\n\n회색 칸 = 도장이 겹친 곳 → 더함\n(예: 가운데 = 2+1+3+2 = 8)\n\n"
                   "크기: $(n-1)s + f - 2p$\n$= (2-1)\\cdot2+3-0 = 5$\n빨간 점선: p=1이면 테두리를\n잘라내서 3x3만 남김",
        ha="left", va="center", fontsize=11)
ax.set_xlim(-0.3, 18.5); ax.set_ylim(5.5, -1.0); ax.set_aspect("equal"); ax.axis("off")
save(fig, "ch4-transpose-conv.png")

# ---------------------------------------------------------------------------
# 11. plain vs residual (Lab 3 설정 그대로)
# ---------------------------------------------------------------------------
def relu(z):
    return np.maximum(0, z)


np.random.seed(0)
a0 = np.abs(np.random.randn(8, 1))


def deep_norms(n_blocks, residual, scale=0.1):
    rng = np.random.RandomState(0)
    a = a0.copy(); norms = [np.linalg.norm(a)]
    for _ in range(n_blocks):
        W1, W2 = rng.randn(8, 8) * scale, rng.randn(8, 8) * scale
        z = W2 @ relu(W1 @ a)
        a = relu(z + a) if residual else relu(z)
        norms.append(np.linalg.norm(a))
    return np.array(norms)


plain, res = deep_norms(50, False), deep_norms(50, True)
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.semilogy(plain + 1e-300, color=RED, lw=2.2, label="plain block (skip 없음)")
ax.semilogy(res, color=BLUE, lw=2.2, label="residual block ($+a^{[l]}$)")
ax.set_xlabel("쌓은 블록 수")
ax.set_ylabel(r"activation 크기 $\|a\|$ (log scale)")
ax.set_title(f"weight가 작을 때(×0.1): 50블록 후 plain {plain[-1]:.1e}, residual {res[-1]:.1f}", fontsize=11)
ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([10.0 ** k for k in range(0, -81, -20)]))
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"1e{int(round(np.log10(y)))}"))
ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.grid(True, which="major", alpha=0.3)
ax.legend()
save(fig, "ch4-plain-vs-residual.png")
