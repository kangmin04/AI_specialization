"""
06-syncsde-deep-dive.md 용 시각자료 생성 스크립트.
00-generate-plots.py와 같은 스타일(색상 팔레트, 한글 폰트)을 그대로 따른다.
실행: python3 06-generate-plots.py  (같은 폴더에 PNG 3개 생성)

주의: 그림 2(correlation spectrum)는 논문에 실제로 등장하는 그래프가 아니라,
00장에서 검증한 "정밀도 가중 평균" 공식을 SyncSDE의 correlation 항에 적용해보면
어떤 모양이 되는지를 보여주는 **개념적 스케치**다. 본문에도 동일하게 명시한다.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams["font.size"] = 11
plt.rcParams["axes.grid"] = False
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREEN = "#55A868"
RED = "#C44E52"
PURPLE = "#8172B2"
GRAY = "#666666"


# ---------------------------------------------------------------------------
# 그림 1: "하나의 어려운 목적함수 = 항A(자기자신) + 항B(다른 것과의 관계)" 패턴이
# 00장 ELBO 분해와 06장 SyncSDE 논문 (6)식에서 똑같이 반복된다는 것을 보여주는 대조 다이어그램.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 7.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

def box(x, y, w, h, text, color, fontsize=10, weight="normal"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.15",
                        linewidth=1.6, edgecolor=color, facecolor=color, alpha=0.15)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#222222", weight=weight, linespacing=1.4)

def arrow(x1, y1, x2, y2, color=GRAY):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                         linewidth=1.6, color=color)
    ax.add_patch(a)

ax.text(2.5, 9.55, "00장: ELBO 분해", ha="center", fontsize=13, weight="bold", color=BLUE)
ax.text(7.5, 9.55, "06장: SyncSDE 목적함수 분해 (논문 식 6)", ha="center", fontsize=13, weight="bold", color=ORANGE)

box(0.6, 8.1, 3.8, 0.9, "log p(x)\n(직접 계산 불가능)", GRAY, 10)
box(5.6, 8.1, 3.8, 0.9, "log p(y_t^i | X~^i)\n(직접 계산 불가능)", GRAY, 10)
arrow(2.5, 8.1, 2.5, 7.3)
arrow(7.5, 8.1, 7.5, 7.3)

box(0.6, 6.4, 3.8, 0.9, "ELBO(q)\n(계산 가능한 하한으로 대체)", BLUE, 10)
box(5.6, 6.4, 3.8, 0.9, "score 분해\n∇log p(y|X~) = ∇log p(y) + ∇log p(X~|y)", ORANGE, 9.3)
arrow(2.5, 6.4, 2.5, 5.6)
arrow(7.5, 6.4, 7.5, 5.6)

box(0.6, 4.5, 1.75, 1.0, "항 1\n재구성(reconstruction)\nE_q[log p(x|z)]", GREEN, 8.6)
box(2.65, 4.5, 1.75, 1.0, "항 2\n정규화(regularization)\n-KL(q(z)||p(z))", RED, 8.6)
box(5.6, 4.5, 1.85, 1.0, "항 1\nfidelity\n∇log p(y_t^i)\n(02장 diffusion score)", GREEN, 8.6)
box(7.55, 4.5, 1.85, 1.0, "항 2\ncorrelation\n∇log p(X~_t^i|y_t^i)\n(task별 설계 대상)", RED, 8.6)

ax.text(1.475, 4.15, "+", ha="center", fontsize=16, color="#222222", weight="bold")
ax.text(6.475, 4.15, "+", ha="center", fontsize=16, color="#222222", weight="bold")

box(0.9, 2.2, 8.2, 1.35,
    "같은 구조: 직접 최적화할 수 없는 하나의 objective를\n"
    "'자기 자신에 대한 항' + '다른 대상과의 관계를 다루는 항' 두 개로 쪼갠다.\n"
    "ELBO의 재구성/정규화 항이 diffusion 자체에 대한 것이라면,\n"
    "SyncSDE의 fidelity/correlation 항은 '나 자신' vs '다른 trajectory들과의 관계'로 쪼갠 것.",
    PURPLE, 10.5)

arrow(2.5, 4.5, 2.5, 3.62, color=PURPLE)
arrow(7.5, 4.5, 7.5, 3.62, color=PURPLE)

fig.suptitle("목적함수의 두 항 분해: ELBO 패턴이 SyncSDE에서 다시 등장한다", fontsize=13, y=0.985)
fig.tight_layout()
fig.savefig("06-objective-decomposition.png", dpi=160)
plt.close(fig)


# ---------------------------------------------------------------------------
# 그림 2 (개념적 스케치, 논문 원문의 그래프 아님):
# correlation 항의 "정밀도(precision)"가 커질수록, 두 trajectory의 결합 추정치가
# "각자 독립적인 추정치" 쪽에서 "완전히 같아지는(=사실상 평균) 지점" 쪽으로 이동한다는 것을
# 00장 1.3절의 정밀도 가중 평균 공식 그대로 시각화한다.
#   결합 위치 = own_precision*own_value + corr_precision*other_value) / (own_precision + corr_precision)
# x축을 corr_precision / own_precision (상대적 correlation 강도, log scale)으로 두면
# 결합 위치는 0(완전 독립) ~ 1(other_value와 완전히 같아짐, 즉 사실상 평균) 사이를 부드럽게 움직인다.
# ---------------------------------------------------------------------------
ratio = np.logspace(-2, 2, 400)  # corr_precision / own_precision
position = ratio / (1.0 + ratio)  # 0 = 완전 독립, 1 = other_value와 완전 일치(=사실상 평균)

fig, ax = plt.subplots(figsize=(9.5, 6.2))
ax.plot(ratio, position, color=PURPLE, lw=2.8, zorder=3)
ax.set_xscale("log")
ax.set_xticks([1e-2, 1e-1, 1e0, 1e1, 1e2])
ax.set_xticklabels(["0.01", "0.1", "1", "10", "100"])  # 유니코드 마이너스 폰트 문제 회피
ax.set_ylim(-0.08, 1.08)
ax.set_xlabel("correlation 항의 상대적 정밀도  =  corr_precision / own_precision   (로그 스케일)", fontsize=10)
ax.set_ylabel("결합된 추정치의 위치\n(0 = 내 궤적 그대로, 1 = 다른 궤적과 완전히 같아짐)", fontsize=10)
ax.set_title("Correlation 강도에 따른 스펙트럼 (개념적 스케치 — 00장 정밀도 가중 평균 공식 적용)", fontsize=12.5, pad=14)

ax.axhline(0.0, color=GREEN, lw=1.2, ls=":", zorder=1)
ax.axhline(1.0, color=RED, lw=1.2, ls=":", zorder=1)

box_kwargs = dict(boxstyle="round,pad=0.35", linewidth=0)
ax.text(0.013, 0.30, "독립 생성\n(correlation precision -> 0)\n각 trajectory가 서로 무관하게 생성됨",
        color="white", fontsize=8.8, va="center", ha="left",
        bbox=dict(facecolor=GREEN, alpha=0.85, **box_kwargs))
ax.text(55, 0.72, "사실상 평균\n(correlation precision -> 매우 큼)\n두 trajectory가 강제로 같아짐",
        color="white", fontsize=8.8, va="center", ha="center",
        bbox=dict(facecolor=RED, alpha=0.85, **box_kwargs))

ax.annotate("MultiDiffusion의 '평균'은 이 스펙트럼의\n한쪽 극단에 가까운 특수 사례였을 뿐이다",
            xy=(9, position[np.searchsorted(ratio, 9)]), xytext=(0.02, 0.86),
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.3), fontsize=8.8, color=GRAY, ha="left")
ax.annotate("SyncSDE는 태스크마다 이 위치를\n(correlation 모델의 형태·강도)\n다르게 설계한다",
            xy=(1.0, position[np.searchsorted(ratio, 1.0)]), xytext=(1.4, 0.08),
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.3), fontsize=8.8, color=GRAY, ha="left")

fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.14)
fig.savefig("06-correlation-spectrum.png", dpi=160)
plt.close(fig)


# ---------------------------------------------------------------------------
# 그림 3: 논문이 다루는 5개 태스크 한눈에 보기 (레포/논문에서 확인된 정보 기준)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")

tasks = [
    ("1. 마스크 기반 T2I\n(mask-based T2I)", "경계 안팎 영역", "마스크 기반 precision matrix M\n(논문 식 8-11, Sec 3.3.1)",
     "src/mask_based_T2I.py"),
    ("2. 실제 이미지 편집\n(real image editing)", "보존할 배경 vs\n수정할 영역", "임계값으로 이진화한 마스크 +\n동일한 precision 구조 (식 12, Sec 3.3.2)",
     "src/real_image_editing.py"),
    ("3. 파노라마/와이드 이미지\n(wide image)", "인접 패치의\n겹치는 영역", "겹치지 않는 픽셀을 마스킹하는\nM_i (식 13-14, Sec 3.3.3)",
     "src/wide_image.py"),
    ("4. 모호한 이미지\n(ambiguous image)", "기하 변환으로\n연결된 뷰들", "등방성(uniform) 분산\n(식 16-17, Sec 3.3.4)",
     "src/ambiguous_image.py"),
    ("5. 3D 메시 텍스처링\n(mesh texturing)", "카메라 시점 간\n겹치는 영역", "시점별 배경 마스크 기반\nprecision 구조 (식 18-19, Sec 3.3.5)",
     "src/mesh_texturing.py"),
]

col_x = [0.3, 2.7, 5.4, 8.5, 10.9]
headers = ["태스크", "무엇이 서로\n상관되는가", "correlation 모델\n(논문에서 확인)", "레포 진입점"]
header_x = [1.4, 4.0, 7.5, 11.5]
for hx, htext in zip(header_x, headers):
    ax.text(hx, 6.6, htext, ha="center", fontsize=10.5, weight="bold", color="#222222")
ax.plot([0.1, 11.9], [6.25, 6.25], color="#999999", lw=1.2)

row_h = 1.12
colors = [BLUE, ORANGE, GREEN, PURPLE, RED]
for i, (name, corr_target, model, entry) in enumerate(tasks):
    y = 5.9 - i * row_h
    c = colors[i]
    box_ = FancyBboxPatch((0.1, y - row_h + 0.18), 11.8, row_h - 0.25,
                           boxstyle="round,pad=0.05,rounding_size=0.1",
                           linewidth=1.2, edgecolor=c, facecolor=c, alpha=0.08)
    ax.add_patch(box_)
    ax.text(1.4, y - row_h / 2 + 0.1, name, ha="center", va="center", fontsize=9.5, weight="bold", color=c)
    ax.text(4.0, y - row_h / 2 + 0.1, corr_target, ha="center", va="center", fontsize=9)
    ax.text(7.5, y - row_h / 2 + 0.1, model, ha="center", va="center", fontsize=8.7)
    ax.text(11.5, y - row_h / 2 + 0.1, entry, ha="center", va="center", fontsize=8.3, family="monospace", color="#444444")

fig.suptitle("SyncSDE가 다루는 5개 태스크 — '무엇이 상관되는가'가 태스크마다 다르다", fontsize=13, y=0.98)
fig.tight_layout()
fig.savefig("06-five-tasks-overview.png", dpi=160)
plt.close(fig)

print("saved: 06-objective-decomposition.png, 06-correlation-spectrum.png, 06-five-tasks-overview.png")
