"""
chapter3-structuring-ml-projects.md (DL Course 3) 용 시각자료 생성 스크립트.
이 코스는 수식보다 "숫자를 보고 판단하는 법"이 핵심이라, 노트에 나오는 표의 숫자들을
그대로 그림으로 옮겨서 한눈에 비교할 수 있게 만든다. numpy + matplotlib만 쓴다.

실행 (repo 루트에서):
    python3 DL/assets/ch3-generate-plots.py
→ 이 스크립트와 같은 폴더(DL/assets/)에 PNG 6개 생성:

  1. ch3-confusion-precision-recall.png : 고양이 분류기 A(P=95%, R=90%)의 confusion matrix와
                                          precision/recall이 각각 어느 칸으로 계산되는지
  2. ch3-f1-vs-mean.png                 : precision=100%로 고정하고 recall을 바꿀 때
                                          산술평균 vs F1(조화평균) 비교
  3. ch3-performance-over-time.png      : 시간에 따른 정확도 곡선 — human-level을 넘으면 느려지고
                                          Bayes optimal error(천장)에 점근 (개념 스케치)
  4. ch3-error-decomposition.png        : 노트의 표 예시들을 Bayes(human) / avoidable bias /
                                          variance / data mismatch 누적 막대로 분해
  5. ch3-error-analysis-ceiling.png     : error analysis 표(dog/great cat/blurry/filter)의 비율과
                                          "완벽히 고쳤을 때 dev error" 천장
  6. ch3-mislabeled-share.png           : dev error 10% vs 2%에서 라벨 오류 0.6%가 차지하는 몫
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 11
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.dirname(os.path.abspath(__file__))

# 공통 색상 (error 성분별로 고정해서 그림끼리 같은 의미 = 같은 색)
C_BAYES = "#9aa0a6"     # 회색: 줄일 수 없는 부분
C_BIAS = "#d9534f"      # 빨강: avoidable bias
C_VAR = "#f0ad4e"       # 주황: variance
C_MISMATCH = "#5b8def"  # 파랑: data mismatch
C_OK = "#3c9d5d"        # 초록
C_DARK = "#333333"


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ---------------------------------------------------------------------------
# 1. Confusion matrix + precision / recall
#    P = 95%, R = 90%가 정확히 떨어지는 정수: TP=171, FN=19 (고양이 190장),
#    FP=9 → P = 171/180 = 0.95, R = 171/190 = 0.90. 전체 1000장 → TN = 801.
# ---------------------------------------------------------------------------
TP, FN, FP = 171, 19, 9
TN = 1000 - TP - FN - FP
assert abs(TP / (TP + FP) - 0.95) < 1e-12 and abs(TP / (TP + FN) - 0.90) < 1e-12

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]
ax.grid(False)
cells = [[TP, FN], [FP, TN]]
names = [["TP\n(진짜 고양이를\n고양이라고 함)", "FN\n(고양이를 놓침)"],
         ["FP\n(고양이 아닌데\n고양이라고 함)", "TN\n(고양이 아닌 걸\n아니라고 함)"]]
colors = [[C_OK, C_BIAS], [C_VAR, "#d0d4d8"]]
for r in range(2):
    for c in range(2):
        ax.add_patch(plt.Rectangle((c, 1 - r), 1, 1, color=colors[r][c], alpha=0.75, ec="white", lw=3))
        ax.text(c + 0.5, 1 - r + 0.62, names[r][c], ha="center", va="center", fontsize=10.5)
        ax.text(c + 0.5, 1 - r + 0.2, f"{cells[r][c]}장", ha="center", va="center",
                fontsize=15, fontweight="bold")
ax.set_xlim(-0.05, 2.05)
ax.set_ylim(-0.05, 2.05)
ax.set_xticks([0.5, 1.5])
ax.set_xticklabels(["모델: 고양이다 (y_hat=1)", "모델: 아니다 (y_hat=0)"])
ax.set_yticks([1.5, 0.5])
ax.set_yticklabels(["실제 고양이\n(y=1)", "실제 고양이 아님\n(y=0)"])
ax.xaxis.tick_top()
ax.set_title("Confusion matrix — 사진 1000장 (Classifier A)", pad=30)
for s in ax.spines.values():
    s.set_visible(False)

ax = axes[1]
ax.axis("off")
txt = (
    "Precision (정밀도) = 모델이 '고양이'라고 한 것 중 진짜\n"
    f"   = TP / (TP + FP) = {TP} / ({TP} + {FP}) = {TP/(TP+FP):.0%}\n"
    "   → 왼쪽 그림의 '첫 번째 열'만 본다\n\n"
    "Recall (재현율) = 진짜 고양이 중 모델이 찾아낸 것\n"
    f"   = TP / (TP + FN) = {TP} / ({TP} + {FN}) = {TP/(TP+FN):.0%}\n"
    "   → 왼쪽 그림의 '첫 번째 행'만 본다\n\n"
    "고양이 판정 기준을 까다롭게 하면\n"
    "   FP↓ (precision↑)  but  FN↑ (recall↓)\n"
    "기준을 느슨하게 하면 그 반대 → trade-off"
)
ax.text(0.0, 0.5, txt, va="center", ha="left", fontsize=11.5, linespacing=1.55)
save(fig, "ch3-confusion-precision-recall.png")


# ---------------------------------------------------------------------------
# 2. F1(조화평균) vs 산술평균 — precision = 100% 고정, recall만 변화
# ---------------------------------------------------------------------------
R = np.linspace(0.01, 1.0, 300)
P = 1.0
arith = (P + R) / 2
f1 = 2 * P * R / (P + R)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(R * 100, arith * 100, color=C_MISMATCH, lw=2.5, label="산술평균 (P+R)/2")
ax.plot(R * 100, f1 * 100, color=C_BIAS, lw=2.5, label="F1 = 2PR/(P+R)  (조화평균)")
ax.plot(R * 100, R * 100, color=C_BAYES, lw=1.2, ls="--", label="min(P, R) = R")
r0 = 0.10
a0, f0 = (P + r0) / 2, 2 * P * r0 / (P + r0)
ax.scatter([r0 * 100, r0 * 100], [a0 * 100, f0 * 100], color=[C_MISMATCH, C_BIAS], zorder=5, s=50)
ax.annotate(f"겁쟁이 모델 (P=100%, R=10%)\n산술평균 {a0:.0%} → 그럴듯해 보임", (r0 * 100, a0 * 100),
            xytext=(22, 72), arrowprops=dict(arrowstyle="->", color=C_DARK), fontsize=10)
ax.annotate(f"F1 {f0:.0%} → 형편없다고 정직하게 말함", (r0 * 100, f0 * 100),
            xytext=(22, 12), arrowprops=dict(arrowstyle="->", color=C_DARK), fontsize=10)
ax.set_xlabel("Recall (%)   [precision은 100%로 고정]")
ax.set_ylabel("합친 점수 (%)")
ax.set_xlim(0, 100)
ax.set_ylim(0, 105)
ax.set_title("F1은 작은 쪽(recall)에 끌려간다")
ax.legend(loc="center right")
save(fig, "ch3-f1-vs-mean.png")


# ---------------------------------------------------------------------------
# 3. 시간에 따른 성능 곡선 (개념 스케치 — 실제 데이터 아님)
# ---------------------------------------------------------------------------
t = np.linspace(0, 10, 400)
bayes_acc, human_acc = 97.0, 93.0
acc = bayes_acc - (bayes_acc - 50) * np.exp(-0.45 * t)   # Bayes 천장에 점근
t_cross = -np.log((bayes_acc - human_acc) / (bayes_acc - 50)) / 0.45

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(t, acc, color=C_MISMATCH, lw=2.8, label="알고리즘 정확도")
ax.axhline(bayes_acc, color=C_DARK, ls="-", lw=1.5)
ax.axhline(human_acc, color=C_BIAS, ls="--", lw=1.5)
ax.text(0.1, bayes_acc + 0.8, "Bayes optimal error에 해당하는 정확도 (이론적 천장, 넘을 수 없음)", fontsize=10)
ax.text(0.1, human_acc - 2.6, "Human-level performance", color=C_BIAS, fontsize=10)
ax.axvline(t_cross, color=C_BAYES, ls=":", lw=1.2)
ax.fill_between(t, acc, bayes_acc, where=t >= t_cross, color=C_BAYES, alpha=0.2)
ax.annotate("human-level 전: 빠르게 상승\n(사람 라벨, 사람 기준 error analysis,\n bias/variance 분석 사용 가능)",
            (2.0, bayes_acc - (bayes_acc - 50) * np.exp(-0.9)), xytext=(3.3, 62),
            arrowprops=dict(arrowstyle="->", color=C_DARK), fontsize=9.5)
ax.annotate("human-level 이후: 느려짐\n(남은 여지 작음 + 사람 도구 못 씀)",
            (7.5, bayes_acc - (bayes_acc - 50) * np.exp(-0.45 * 7.5)), xytext=(6.2, 80),
            arrowprops=dict(arrowstyle="->", color=C_DARK), fontsize=9.5)
ax.set_xlabel("시간 / 투입한 노력")
ax.set_ylabel("정확도 (%)")
ax.set_ylim(45, 101)
ax.set_xlim(0, 10)
ax.set_xticks([])
ax.set_title("성능은 human-level까지 빠르게, 그 뒤로는 Bayes 천장에 천천히 접근 (개념도)")
save(fig, "ch3-performance-over-time.png")


# ---------------------------------------------------------------------------
# 4. Error 분해 누적 막대 — 노트의 표 숫자 그대로
#    (human, train, train-dev, dev). train-dev가 없는 예시는 variance를 dev-train으로.
# ---------------------------------------------------------------------------
examples = [
    ("Case 1\nhuman 1 / train 8 / dev 10", 1, 8, None, 10),
    ("Case 2\nhuman 7.5 / train 8 / dev 10", 7.5, 8, None, 10),
    ("Train-dev 예 1\n0 / 1 / train-dev 9 / dev 10", 0, 1, 9, 10),
    ("Train-dev 예 2\n0 / 1 / train-dev 1.5 / dev 10", 0, 1, 1.5, 10),
    ("셋 다 있는 예\n4 / 7 / train-dev 10 / dev 12", 4, 7, 10, 12),
]
fig, ax = plt.subplots(figsize=(10.5, 5.6))
ax.grid(axis="y", visible=False)
for i, (label, h, tr, td, dv) in enumerate(examples):
    y = len(examples) - 1 - i
    if td is None:
        segs = [(h, C_BAYES), (tr - h, C_BIAS), (dv - tr, C_VAR)]
    else:
        segs = [(h, C_BAYES), (tr - h, C_BIAS), (td - tr, C_VAR), (dv - td, C_MISMATCH)]
    left = 0
    for w, c in segs:
        ax.barh(y, w, left=left, color=c, edgecolor="white", height=0.6)
        if w >= 0.9:
            ax.text(left + w / 2, y, f"{w:g}", ha="center", va="center", fontsize=10,
                    color="white" if c != C_BAYES else C_DARK, fontweight="bold")
        left += w
    # 가장 큰 병목 표시 (Bayes 제외)
    parts = {"avoidable bias": tr - h, "variance": (td if td is not None else dv) - tr}
    if td is not None:
        parts["data mismatch"] = dv - td
    top = max(parts.values())
    worst = " = ".join(k for k, v in parts.items() if abs(v - top) < 1e-9)
    ax.text(dv + 0.25, y, f"→ 1순위: {worst}", va="center", fontsize=10)
ax.set_yticks(range(len(examples)))
ax.set_yticklabels([e[0] for e in examples][::-1], fontsize=9.5)
ax.set_xlim(0, 18)
ax.set_xlabel("Error (%) — 막대 끝이 dev error")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_BAYES, label="Bayes 근사 (human-level) — 못 줄임"),
                   Patch(color=C_BIAS, label="Avoidable bias = train - human"),
                   Patch(color=C_VAR, label="Variance = train-dev(또는 dev) - train"),
                   Patch(color=C_MISMATCH, label="Data mismatch = dev - train-dev")],
          loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9.5)
ax.set_title("Dev error를 원인별로 쌓아보기 — 가장 긴 색 조각부터 공략")
save(fig, "ch3-error-decomposition.png")


# ---------------------------------------------------------------------------
# 5. Error analysis 천장
# ---------------------------------------------------------------------------
cats = ["Dog", "Great cat\n(사자/표범)", "Blurry", "Filter 왜곡"]
share = np.array([8, 43, 61, 12])
dev_err = 10.0
ceiling = dev_err * (1 - share / 100)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
bars = ax.bar(cats, share, color=[C_BAYES, C_MISMATCH, C_MISMATCH, C_BAYES])
for b, s in zip(bars, share):
    ax.text(b.get_x() + b.get_width() / 2, s + 1.2, f"{s}%", ha="center", fontsize=11)
ax.set_ylim(0, 72)
ax.set_ylabel("틀린 dev 샘플 중 비율 (%)")
ax.set_title("틀린 샘플 100개를 훑어본 결과\n(겹칠 수 있어서 합이 100% 넘어도 정상)")

ax = axes[1]
ax.axhline(dev_err, color=C_BIAS, ls="--", lw=1.5)
ax.text(3.45, dev_err + 0.25, "현재 dev error 10%", color=C_BIAS, ha="right", fontsize=10)
bars = ax.bar(cats, ceiling, color=[C_BAYES, C_MISMATCH, C_MISMATCH, C_BAYES])
for b, c, s in zip(bars, ceiling, share):
    ax.text(b.get_x() + b.get_width() / 2, c - 0.6, f"{c:.1f}%", ha="center", fontsize=11,
            color="white", fontweight="bold")
    ax.annotate("", xy=(b.get_x() + b.get_width() / 2, c), xytext=(b.get_x() + b.get_width() / 2, dev_err),
                arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.2))
ax.set_ylim(0, 11.5)
ax.set_ylabel("그 카테고리를 100% 고쳤을 때 dev error (%)")
ax.set_title("각 아이디어의 천장(ceiling) = 10% × (1 - 비율)\n화살표 길이 = 최대로 얻을 수 있는 개선폭")
save(fig, "ch3-error-analysis-ceiling.png")


# ---------------------------------------------------------------------------
# 6. Mislabeled data가 dev error에서 차지하는 몫
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 3.6))
ax.grid(axis="y", visible=False)
rows = [("상황 1\n(dev error 10%)", 9.4, 0.6), ("상황 2\n(dev error 2%)", 1.4, 0.6)]
for i, (lab, other, mis) in enumerate(rows):
    y = 1 - i
    ax.barh(y, other, color=C_BAYES, height=0.55, edgecolor="white")
    ax.barh(y, mis, left=other, color=C_BIAS, height=0.55, edgecolor="white")
    tot = other + mis
    if other > 3:
        ax.text(other / 2, y, f"다른 원인 {other}%", ha="center", va="center", fontsize=10)
        msg = f"라벨 오류 {mis}% = 전체 error의 {mis / tot:.0%}"
    else:
        msg = f"다른 원인 {other}% + 라벨 오류 {mis}%\n→ 라벨 오류 = 전체 error의 {mis / tot:.0%}"
    ax.text(tot + 0.15, y, msg, va="center", fontsize=10, color=C_BIAS)
ax.set_yticks([1, 0])
ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(0, 15)
ax.set_xlabel("Dev error (%)")
ax.set_title("같은 0.6%라도 전체 error가 작아지면 비중이 커진다")
save(fig, "ch3-mislabeled-share.png")
