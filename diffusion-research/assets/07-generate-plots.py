"""
07-research-transition.md 용 시각자료 생성 스크립트 (옵션 A 실습).

목표: "미니 파노라마" 두 개의 diffusion-like trajectory를 합칠 때,
겹치는 영역을 (a) 단순 평균 (b) 거리 기반 가중 평균(feathering)
(c) 한쪽에 치우친 가중 평균(biased) 으로 결합하면 결과가 어떻게
달라지는지 실제로 돌려서 비교한다.

실제 DDPM 대신, "노이즈 -> 목표 패턴으로 서서히 수렴하면서 노이즈가
줄어드는" 아주 단순화된 반복 과정으로 diffusion의 reverse process를
흉내낸다 (진짜 학습된 score 없이, 목표로 선형 보간 + 감쇠하는 가우시안
노이즈를 더하는 방식 - MultiDiffusion/SyncSDE의 "여러 trajectory를 매
스텝 합친다"는 구조만 재현하는 것이 목적).

실행: python3 07-generate-plots.py  (같은 폴더에 PNG 2개 생성 후
diffusion-research/assets/ 로 복사)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 11
plt.rcParams["axes.grid"] = False
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. 두 개의 "윈도우"(=두 개의 독립적인 diffusion trajectory)와 겹치는 영역 정의
#    캔버스: 32(H) x 64(W). 윈도우 A: col [0, 40), 윈도우 B: col [24, 64)
#    겹치는 영역: col [24, 40) (폭 16)
# ---------------------------------------------------------------------------
H, W = 32, 64
win_w = 40
overlap_start, overlap_end = 24, 40  # A의 오른쪽 끝 겹침 / B의 왼쪽 끝 겹침
overlap_w = overlap_end - overlap_start

xx, yy = np.meshgrid(np.arange(win_w), np.arange(H))

# 윈도우 A의 "목표(clean) 패턴": 세로 줄무늬 (수직 sine)
target_A = 0.5 + 0.5 * np.sin(2 * np.pi * xx / 10.0)
# 윈도우 B의 "목표(clean) 패턴": 대각선 체커보드 느낌 (다른 주파수 sine)
xx_b, yy_b = np.meshgrid(np.arange(win_w), np.arange(H))
target_B = 0.5 + 0.5 * np.sin(2 * np.pi * (xx_b + yy_b) / 7.0)

T = 40  # 총 스텝 수 (단순화된 reverse process)
step_rate = 0.12  # 매 스텝 목표로 끌어당기는 비율
noise_scale0 = 1.0  # 초기 노이즈 크기


def run_denoising_process(overlap_combine_fn, label):
    """단순화된 reverse diffusion: 매 스텝
    1) 전역 캔버스에서 윈도우 A, B 패치를 잘라온다
    2) 각 패치를 각자의 목표 쪽으로 한 스텝 이동 + 감쇠하는 노이즈 추가 (독립적 trajectory)
    3) 겹치는 영역만 overlap_combine_fn으로 합치고, 나머지는 각자 값 그대로 캔버스에 반영
    """
    canvas = rng.normal(0, noise_scale0, size=(H, W))  # 순수 노이즈에서 시작
    for t in range(T):
        noise_scale = noise_scale0 * (1 - (t + 1) / T)  # 뒤로 갈수록 노이즈 감쇠 (DDPM과 동일한 직관)

        patch_A = canvas[:, 0:win_w]
        patch_B = canvas[:, W - win_w:W]

        prop_A = patch_A + step_rate * (target_A - patch_A) + rng.normal(0, noise_scale * 0.3, size=patch_A.shape)
        prop_B = patch_B + step_rate * (target_B - patch_B) + rng.normal(0, noise_scale * 0.3, size=patch_B.shape)

        new_canvas = canvas.copy()
        # A만 담당하는 영역 (겹치지 않는 왼쪽 부분)
        new_canvas[:, 0:overlap_start] = prop_A[:, 0:overlap_start]
        # B만 담당하는 영역 (겹치지 않는 오른쪽 부분)
        new_canvas[:, overlap_end:W] = prop_B[:, (overlap_end - (W - win_w)):]
        # 겹치는 영역: A의 제안값 / B의 제안값을 결합 규칙으로 합침
        a_seg = prop_A[:, overlap_start:overlap_end]
        b_seg = prop_B[:, (overlap_start - (W - win_w)):(overlap_end - (W - win_w))]
        new_canvas[:, overlap_start:overlap_end] = overlap_combine_fn(a_seg, b_seg)

        canvas = new_canvas
    return canvas


def combine_simple_average(a, b):
    """(a) MultiDiffusion 방식: 단순 평균."""
    return 0.5 * a + 0.5 * b


def combine_distance_weighted(a, b):
    """(b) 거리 기반 가중 평균 (feathering): 겹침 구간 내에서 자기 쪽 경계에 가까울수록
    자기 값의 가중치를 높이고, 반대쪽 경계로 갈수록 낮춘다 (선형 램프)."""
    w = np.linspace(1.0, 0.0, overlap_w)  # 왼쪽(A 쪽 경계)=1 -> 오른쪽(B 쪽 경계)=0
    w = w[None, :]
    return w * a + (1 - w) * b


def combine_biased(a, b):
    """(c) 한쪽에 강한 가중치: A의 trajectory가 훨씬 더 신뢰할 만하다고
    가정하는(강한 correlation) 경우 - SyncSDE 식으로 말하면 "항 2"를
    특정 방향으로 강하게 잡은 것에 해당."""
    wA = 0.85
    return wA * a + (1 - wA) * b


canvas_simple = run_denoising_process(combine_simple_average, "단순 평균 (MultiDiffusion)")
canvas_weighted = run_denoising_process(combine_distance_weighted, "거리 기반 가중 평균 (feathering)")
canvas_biased = run_denoising_process(combine_biased, "편향된 가중 평균 (A 쪽에 0.85)")


# ---------------------------------------------------------------------------
# 정량 비교: 겹치는 영역 양쪽 경계에서의 불연속성(이음매 강도)을
# "경계를 가로지르는 열(column) 간 픽셀 값 차이의 평균 절댓값"으로 측정
# ---------------------------------------------------------------------------
def seam_discontinuity(canvas, boundary_col):
    left = canvas[:, boundary_col - 1]
    right = canvas[:, boundary_col]
    return np.mean(np.abs(right - left))


results = {}
for name, canvas in [
    ("단순 평균", canvas_simple),
    ("거리 가중 평균", canvas_weighted),
    ("편향 평균", canvas_biased),
]:
    d_left = seam_discontinuity(canvas, overlap_start)
    d_right = seam_discontinuity(canvas, overlap_end)
    results[name] = (d_left, d_right, (d_left + d_right) / 2)

print("결합 방식별 이음매 불연속성 (작을수록 매끄러움):")
for name, (dl, dr, avg) in results.items():
    print(f"  {name:12s}: 왼쪽 경계={dl:.4f}, 오른쪽 경계={dr:.4f}, 평균={avg:.4f}")

# ---------------------------------------------------------------------------
# 그림 1: 세 결합 방식의 최종 캔버스 비교 (+ 겹침 구간 표시)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(9, 7.5))
titles = [
    f"(a) 단순 평균 - MultiDiffusion 방식\n이음매 불연속성 평균 = {results['단순 평균'][2]:.4f}",
    f"(b) 거리 기반 가중 평균 (feathering)\n이음매 불연속성 평균 = {results['거리 가중 평균'][2]:.4f}",
    f"(c) 편향된 가중 평균 (A쪽 0.85) - 강한 correlation 가정\n이음매 불연속성 평균 = {results['편향 평균'][2]:.4f}",
]
for ax, canvas, title in zip(axes, [canvas_simple, canvas_weighted, canvas_biased], titles):
    ax.imshow(canvas, cmap="viridis", vmin=-0.3, vmax=1.3, aspect="auto")
    ax.axvline(overlap_start, color="white", linestyle="--", linewidth=1)
    ax.axvline(overlap_end, color="white", linestyle="--", linewidth=1)
    ax.set_title(title, fontsize=10)
    ax.set_yticks([])
    ax.set_xticks([0, overlap_start, overlap_end, W])
fig.suptitle("미니 파노라마: 겹치는 영역 결합 방식에 따른 결과 비교 (실제 실행 결과)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("07-overlap-combination-comparison.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 그림 2: 이해 -> 실습 -> 연구자 전환 흐름 다이어그램
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis("off")

boxes = [
    (0.4, "이해\n(00~06번 챕터)\n수식 유도 + 논문 정독"),
    (3.8, "실습/검증\n(이 챕터 섹션 1)\n미니 재현 프로젝트로\n직접 확인"),
    (7.2, "연구자 습관\n(이 챕터 섹션 2~4)\n논문 팔로우 + 랩 서칭\n+ 자기 점검"),
]
box_w, box_h = 2.6, 1.6
for x, text in boxes:
    ax.add_patch(plt.Rectangle((x, 0.8), box_w, box_h, fill=True,
                                facecolor="#dbe9f6", edgecolor="#2b6cb0", linewidth=1.5))
    ax.text(x + box_w / 2, 0.8 + box_h / 2, text, ha="center", va="center", fontsize=9.5)

for x0, x1 in [(0.4 + box_w, 3.8), (3.8 + box_w, 7.2)]:
    ax.annotate("", xy=(x1, 1.6), xytext=(x0, 1.6),
                arrowprops=dict(arrowstyle="-|>", color="#2b6cb0", linewidth=2))

ax.text(5, 2.75, "\"읽고 이해한다\"에서 \"스스로 검증하고 질문을 던진다\"로", ha="center", fontsize=10, style="italic")
fig.tight_layout()
fig.savefig("07-understanding-to-research-flow.png", dpi=150)
plt.close(fig)

print("저장 완료: 07-overlap-combination-comparison.png, 07-understanding-to-research-flow.png")
