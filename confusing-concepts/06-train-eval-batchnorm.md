# model.train() / model.eval(), 그리고 BatchNorm

## 헷갈렸던 점

> `model.eval()`이 뭔가? BatchNorm 등등.
> lab02 Part 2에서는 학습 중인데 왜 `train()`이 아니라 `eval()`을 쓰나?

## 결론

- `train()` / `eval()`은 **학습을 할지 말지 정하는 스위치가 아니다.** BatchNorm, Dropout 같은 몇몇 층의 **동작 방식을 바꾸는 모드 스위치**다. Conv, Linear는 모드와 무관하게 똑같이 동작한다.
- **BatchNorm = 중간 층마다 다시 하는 z-score 정규화.** train 모드는 **지금 배치의** 평균/분산을 쓰고 장기 통계(`running_mean/var`)를 갱신한다. eval 모드는 **쌓아 둔 고정 통계**를 쓴다.
- `requires_grad=False`로는 BatchNorm의 `running_mean/var`를 못 얼린다 (gradient가 아니라 forward 때 갱신되는 값이라서). 그래서 backbone을 완전히 얼리려면 `eval()`도 필요하다.
- `eval()`이어도 gradient는 계산된다. gradient를 끄는 건 `torch.no_grad()`의 일이다.

## 1. BatchNorm이 하는 일

ML 코스에서 입력 feature를 z-score 정규화하면 학습이 잘 된다고 배웠다. 그런데 그건 **입력에서 한 번**만 하는 것이다. 깊은 네트워크에서는 층을 지날 때마다 분포가 계속 바뀌어서(어떤 층은 평균 50, 어떤 층은 0.001) 학습이 불안정해진다.

**BatchNorm = 중간 층마다 z-score 정규화를 다시 해 주는 층.**

```
conv 출력 → [BatchNorm] → ReLU → 다음 conv ...
```

채널마다:

```
1) x̂ = (x - μ) / √(σ² + ε)     ← z-score (평균 0, 분산 1)
2) y = γ · x̂ + β                ← 학습되는 스케일 γ, 이동 β
```

- `μ`, `σ²`: 정규화에 쓰는 평균, 분산. **어디서 가져오느냐가 train/eval의 차이.**
- `ε`: 0으로 나누기 방지 (1e-5)
- `γ`(weight), `β`(bias): **학습 파라미터.** 평균 0, 분산 1이 항상 최선은 아니니 모델이 적절한 스케일/위치를 찾게 한다.

### BatchNorm이 가진 두 종류의 값

```
bn1: BatchNorm2d(64, eps=1e-05, momentum=0.1, ...)
   파라미터: ['weight', 'bias']
   buffer:   ['running_mean', 'running_var', 'num_batches_tracked']
```

| 종류 | 이름 | 어떻게 바뀌나 |
|---|---|---|
| **파라미터** | `weight`(γ), `bias`(β) | gradient로 업데이트. `requires_grad`로 얼릴 수 있음 |
| **buffer** (통계) | `running_mean`, `running_var` | **forward 할 때 자동 갱신.** gradient와 무관 → `requires_grad`로 못 얼림 |

`64` = 채널 수. 채널마다 μ, σ², γ, β가 하나씩.

## 2. train 모드 vs eval 모드

### train 모드: 지금 배치로 통계를 낸다

```
μ, σ² = 지금 배치(예: 64장)의 평균, 분산
```

동시에 장기 평균을 조금씩 갱신한다:

```
running_mean = 0.9 × running_mean + 0.1 × (이번 배치 평균)     ← momentum=0.1
running_var  = 0.9 × running_var  + 0.1 × (이번 배치 분산)
```

학습 내내 쌓이면서 `running_mean/var`에 **학습 데이터 전체의 평균적인 통계**가 담긴다. ResNet이면 ImageNet 100만 장의 통계.

### eval 모드: 쌓아 둔 통계를 그대로 쓴다

```
μ, σ² = running_mean, running_var   (고정, 갱신 안 함)
```

왜 나누나:
- 테스트 때 사진이 1장만 들어올 수 있다. 1장으로 평균/분산을 내면 그 사진에 맞춰 정규화돼서 엉망이 된다.
- 같은 사진인데 **같은 배치에 어떤 사진이 함께 있었느냐**에 따라 예측이 달라지면 안 된다.

### 실측: 모드에 따라 사진 1장 예측이 이렇게 달라진다

```
train 모드 사진1장: bucket   0.009
eval 모드 사진1장:  bobsled  0.253
```

train 모드에서는 사진 1장의 통계로 정규화하니 1등 확률이 0.009, 사실상 의미 없는 예측. **Part 1(추론)에서 `.eval()`이 필수인 이유.**

## 3. Part 2에서 학습 중인데 eval()을 쓰는 이유

### 실측: freeze해도 train 모드면 통계가 바뀐다

모든 파라미터를 `requires_grad=False`로 얼리고, **backward/step 없이 forward만 1번**:

```
원래 running_mean[:3]                        [0.0028, -0.0258, 0.0]
freeze + train() 모드로 forward 1번 후        [0.0136, -0.0912, 0.0]   ← 바뀜!
freeze + eval()  모드로 forward 1번 후        [0.0028, -0.0258, 0.0]   ← 그대로
```

`requires_grad=False`는 gradient로 바뀌는 파라미터(γ, β, conv 가중치)만 막는다. `running_mean/var`는 **train 모드에서 forward만 해도** 바뀐다.

Part 2를 `train()`으로 돌리면:
- CIFAR 배치가 들어올 때마다 backbone 안 BatchNorm 통계가 CIFAR 쪽으로 끌려간다.
- backbone이 내는 512개 특징이 조금씩 바뀐다.
- fc는 계속 변하는 특징에 맞추느라 흔들린다. "backbone을 얼렸다"가 반만 맞는 상태.

`eval()`이면 통계까지 ImageNet 값으로 고정 → backbone이 **진짜 고정된 특징 추출기**가 된다.

```
backbone을 완전히 얼리려면
├─ requires_grad = False   → conv 가중치, BN의 γ·β 고정      (gradient 경로 차단)
└─ model.eval()            → BN의 running_mean·var 고정     (forward 중 갱신 차단)
```

**eval 모드에서도 학습이 되나?** 된다. `eval()`은 BatchNorm/Dropout 동작만 바꾸고 gradient 계산은 막지 않는다. 새 fc는 `requires_grad=True`라서 `backward()` + `step()`으로 정상 학습된다.

## 4. Dropout도 모드에 따라 달라진다

Dropout(ML 코스 Course 2 정규화 기법): 학습 때 뉴런 일부를 **랜덤으로 0으로 꺼서** overfitting을 막는다. ResNet-18에는 없지만 다른 모델엔 자주 있다.

```
입력: [1, 1, 1, 1, 1, 1, 1, 1],  Dropout(p=0.5)
train 모드: [0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 2.0, 2.0]   ← 일부 끄고 나머지 ×2
eval 모드:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   ← 아무것도 안 함
```

- train: 랜덤으로 끄고, 살아남은 값은 `1/(1-p)` = 2배로 키워 평균 유지
- eval: 안 끈다. 예측에 랜덤성이 있으면 같은 사진에 매번 다른 답이 나오니까

## 5. 헷갈리는 네 가지 정리

| | 무엇을 바꾸나 | 언제 |
|---|---|---|
| `model.train()` | BN: **배치 통계** + running 통계 갱신 / Dropout **켜짐** | 일반 학습 (Step 1) |
| `model.eval()` | BN: **고정 running 통계** / Dropout **꺼짐** | 추론·평가 (Part 1), **backbone 얼린 fine-tuning** (Part 2) |
| `torch.no_grad()` | gradient 계산·기록 끔 (층 동작은 그대로) | 추론·평가 때 메모리/속도 절약 |
| `requires_grad=False` | 특정 **파라미터**의 업데이트 막음 | freeze |

```python
# Step 1 — 처음부터 학습
model.train()             # BN 통계도 MNIST에 맞춰 새로 쌓아야 하니 train

# Part 1 — 추론
model.eval()              # 고정 통계 (사진 1장이어도 정상)
with torch.no_grad():     # gradient 필요 없음

# Part 2 — backbone 얼리고 fc만 학습
model.eval()              # BN 통계까지 고정 → backbone 완전 동결
# no_grad는 쓰면 안 됨! fc는 gradient가 필요하다
```

## 기억할 것

- `train()`/`eval()` = BatchNorm·Dropout에게 "학습 중이니 배치 통계 쓰고 랜덤으로 꺼라" vs "실전이니 고정값 쓰고 랜덤성 빼라"를 알려 주는 스위치. **gradient와는 별개.**
- BatchNorm에는 gradient로 바뀌는 값(γ, β)과 forward로 바뀌는 값(running_mean/var)이 둘 다 있다.
- backbone 완전 동결 = `requires_grad=False` + `eval()`.
- 추론 = `eval()` + `no_grad()`.
