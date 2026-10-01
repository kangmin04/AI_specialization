# Fine-tuning(전이 학습): fc 교체와 freeze

## 헷갈렸던 점

> 파인튜닝을 처음 해 봐서 기본 개념부터 모르겠다.
> fc가 대체 뭔가? fully connected라는데, 그걸 어떻게 "교체"하고 "동결"한다는 건지 후반부 코드가 너무 어렵다.

## 결론

- **fc = Keras `Dense` = PyTorch `nn.Linear`**. ML 코스에서 배운 뉴런 한 층(`z = W·a + b`)이다. ResNet의 fc는 특징 512개 → 클래스 점수 1000개.
- `fc`는 torchvision이 마지막 층에 붙인 **이름(속성)**일 뿐이다. **교체 = 속성에 새 층을 대입**(`model.fc = nn.Linear(512, 10)`)하는 것.
- **동결(freeze) = 가중치마다 붙은 `requires_grad` 스위치를 끄는 것.** 끄면 gradient가 안 생겨서 업데이트가 안 된다.
- Part 2가 하는 일: **ResNet 앞부분은 사진 → 특징 512개로 바꾸는 고정 함수로 쓰고, 그 512개로 10클래스 softmax regression(fc 한 층)만 새로 학습.**

## 1. 왜 pretrained 모델을 가져다 쓰나

### 문제: 데이터는 적은데 모델은 크다

ResNet-18 파라미터는 약 1,100만 개. 처음부터 학습하려면 ImageNet 수준(100만 장 이상)이 필요하다. 내 데이터가 5,000장이면 모델이 데이터를 외워 버린다 → **overfitting (high variance)**.

### 아이디어: 이미 배운 "보는 능력"을 빌린다

```
앞쪽 층 (conv1, layer1)  → 선, 모서리, 색 변화            ← 어떤 사진에서든 쓸모 있음
중간 층 (layer2, layer3) → 질감, 무늬, 원, 격자            ← 대부분 쓸모 있음
뒤쪽 층 (layer4)         → 눈, 바퀴, 털, 날개 같은 부품      ← 꽤 쓸모 있음
마지막 층 (fc)           → "이 부품 조합이면 1000개 중 tabby" ← ImageNet 전용
```

특징을 뽑는 능력은 고양이든 비행기든 똑같이 필요하다. 바꿔야 하는 건 마지막에 "그래서 어떤 클래스냐"를 정하는 부분뿐.

- **transfer learning(전이 학습)**: 남이 큰 데이터로 학습한 모델을 가져와 내 문제에 맞게 일부만 바꿔 다시 학습하는 것
- **fine-tuning**: 그 "다시 학습하는" 과정

### 용어

```
         ┌──────────────── backbone ────────────────┐   ┌─ head ─┐
입력 → conv1 → layer1 → layer2 → layer3 → layer4 → avgpool → fc → 출력
         └──────── 특징 추출기 (512차원 벡터 생성) ──────┘   └ 분류기 ┘
```

| 용어 | 뜻 |
|---|---|
| **backbone** | 특징을 뽑는 몸통. pretrained 가중치 재사용 |
| **head** | 특징으로 최종 답을 내는 머리. 문제마다 새로 만든다 |
| **freeze** | 학습 중 가중치가 안 바뀌게 고정 |
| **pretrained** | 미리 학습된 가중치 |
| **from scratch** | 랜덤 초기화에서 처음부터 학습 |

### fine-tuning의 두 가지 강도

| 방식 | 학습하는 부분 | 언제 |
|---|---|---|
| **head만 학습** (lab02 Part 2) | fc만. backbone freeze. *linear probing* / *feature extraction*이라고도 함 | 데이터 적을 때, 빠르게 |
| **전체 fine-tuning** | backbone까지 전부. 대신 lr을 작게 해서 조금씩만 고침 | 데이터 많을 때, 내 데이터가 ImageNet과 많이 다를 때 (의료 영상 등) |

## 2. fc = Dense 층

```python
Dense(units=10, activation='linear')   # Keras
nn.Linear(512, 10)                      # PyTorch: 입력 512개 → 출력 10개
```

**fully connected = 모든 입력이 모든 출력과 연결.** 입력 3개, 출력 2개 예시:

```
입력        출력
 a1 ──┬──→ z1 = w11·a1 + w12·a2 + w13·a3 + b1
 a2 ──┼──→ z2 = w21·a1 + w22·a2 + w23·a3 + b2
 a3 ──┘
 (선 3×2 = 6개, 전부 연결)    W: (2, 3),  b: (2,)
```

ResNet의 fc:

```
Linear(in_features=512, out_features=1000, bias=True)
weight (1000, 512)   bias (1000,)
```

- 입력 512개 = backbone이 사진에서 뽑은 특징 숫자
- 출력 1000개 = ImageNet 클래스별 점수. `z_k = w_k · a + b_k` = "이 특징이면 k번 클래스일 점수"

```
사진 ──[ backbone: 특징 512개 뽑기 ]──→ 숫자 512개 ──[ fc: 점수 매기기 ]──→ 점수 1000개
```

## 3. "교체" = 속성에 새 값 대입

`model`은 파이썬 객체이고 각 층은 그 **속성**이다.

```python
model.conv1    # 첫 conv 층
model.layer1
model.fc       # 마지막 Linear 층
```

그래서 교체는 그냥:

```python
model.fc = nn.Linear(model.fc.in_features, 10)
```

`person.phone = new_phone`과 같은 문법. 대입 후 forward 마지막 단계에서 새 층이 쓰인다.

```
교체 후 Linear(in_features=512, out_features=10, bias=True)  weight (10, 512)
출력 shape (2, 10)     ← 사진 2장 → 점수 10개씩
```

- **왜 교체?** 기존 fc는 점수 1000개(ImageNet)를 내는데 CIFAR-10은 10개만 필요하다.
- **`model.fc.in_features`**: 기존 fc에게 "입력 몇 개 받았어?" → 512. 새 층도 backbone에서 512개를 받아야 하니 그대로 쓴다. ResNet-50(2048)으로 바꿔도 코드 수정이 필요 없다.
- 새 층의 가중치는 **랜덤**이라 학습해야 한다. (교체 전 가중치는 `[-0.018, -0.070, -0.052, ...]`처럼 학습된 값이었다.)

```
before:  ... → avgpool → (512) → fc: Linear(512, 1000) → (B, 1000)   ImageNet
after:   ... → avgpool → (512) → fc: Linear(512, 10)   → (B, 10)     CIFAR-10
                                   ↑ 새 층, 랜덤 초기화, 학습 대상
```

## 4. "동결" = requires_grad 스위치 끄기

가중치 텐서마다 **`requires_grad` on/off 스위치**가 있다.

- `True` (기본): `loss.backward()` 때 `.grad`(= `dj_dw`) 계산 → `opt.step()`에서 `w = w - α·dj_dw`로 업데이트
- `False`: gradient 계산 안 함 → 업데이트 안 됨 → **동결**

```python
for p in model.parameters():   # 모델 안의 가중치 텐서를 하나씩 꺼내서
    p.requires_grad = False    # 스위치를 끈다
```

실측:

```
freeze 후      conv1: False   fc: False   ← 전부 꺼짐
fc 교체 후     conv1: False   fc: True    ← 새 fc만 켜져 있음
```

새 fc가 `True`인 이유: freeze 루프는 **그때 있던** 가중치만 껐다. 새 fc는 그 뒤에 만들어져서 기본값 `True`.

## 5. 순서가 중요하다

```python
# ① 학습된 모델 통째로 가져오기
model = models.resnet18(weights=weights)
#    [conv1 ... layer4 ✅학습됨] → [fc 512→1000 ✅학습됨]

# ② 전부 동결
for p in model.parameters():
    p.requires_grad = False
#    [conv1 ... layer4 🔒] → [fc 512→1000 🔒]

# ③ 마지막 층만 새것으로 교체
model.fc = nn.Linear(model.fc.in_features, 10)
#    [conv1 ... layer4 🔒] → [fc 512→10 🆕랜덤, 학습 가능]

# ④ 모든 수정이 끝난 뒤 GPU로
model = model.to(device)

# ⑤ optimizer에도 fc만
opt = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
```

- **② → ③ 순서를 바꾸면**: 새 fc까지 얼어서 학습할 파라미터가 0개 → `loss.backward()`에서 에러.
- **④를 ③보다 먼저 하면**: 새 fc가 CPU에 만들어져 모델 일부는 GPU, 일부는 CPU → forward에서 장치 불일치 에러.
- **⑤ optimizer에 `model.fc.parameters()`만 넘기는 이유**: optimizer는 "업데이트할 목록"을 받는다. `requires_grad=False`만으로도 backbone은 안 바뀌지만, 이중으로 안전하고 Adam이 backbone용 상태(moment)를 메모리에 만들지 않는다.

## 6. Part 2 나머지 코드

### 데이터

```python
train = Subset(datasets.CIFAR10(root, train=True, download=True, transform=preprocess), range(5000))
test  = Subset(datasets.CIFAR10(root, train=False, download=True, transform=preprocess), range(1000))
```

- `datasets.CIFAR10`: 32×32 컬러, 10클래스. `train=True` 50,000장 / `False` 10,000장
- `transform=preprocess`: 사진을 **꺼낼 때마다** 적용 (미리 전부 변환해 두지 않음)
- `Subset(..., range(5000))`: 앞 5,000개 인덱스만 쓰는 얇은 포장지. 데이터 복사 없음

### 모델

```python
model = models.resnet18(weights=weights if PRETRAINED else None)
```

`A if 조건 else B` = 한 줄 if. `None`이면 **구조만 있고 가중치는 랜덤**.

### 파라미터 수

```python
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f"학습되는 파라미터 {n_train:,} / 전체 {n_total:,}")
```

- `p.numel()`: 원소 개수. `(10, 512)`면 5,120
- `sum(... for ... if ...)`: 조건 맞는 것만 더하기 (generator expression)
- `:,`: 천 단위 쉼표

### 학습 루프

```python
for epoch in range(3):
    model.eval()                       # ← Step 1은 train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()                # fc의 gradient만 계산됨
        opt.step()                     # fc만 업데이트
```

- Step 1의 5줄과 **똑같다.** 차이는 `train()` 대신 `eval()` 하나 → 이유는 [06 문서](06-train-eval-batchnorm.md)
- **속도**: `requires_grad=True`가 맨 끝 fc뿐이라 backward가 fc까지만 계산하고 멈춘다. backbone 쪽으로 역전파가 안 내려가서 빠르다.

### 평가

```python
correct += (model(x).argmax(1) == y).sum().item()
```

`(256, 10)` → `argmax(1)` 예측 번호 `(256,)` → `== y` True/False → `.sum().item()` 맞은 개수.
출력되는 `loss`는 **마지막 배치 하나**의 값이라 들쭉날쭉해도 정상.

## 7. Part 1 vs Part 2

| | Part 1 (inference) | Part 2 (fine-tuning) |
|---|---|---|
| 가중치 | ImageNet pretrained 그대로 | ImageNet pretrained + **새 fc** |
| 출력 | `(B, 1000)` | `(B, 10)` |
| 학습 | 없음 | fc만 |
| 루프 | forward만 | Step 1과 같은 5줄 |

## 기억할 것

- fc = Dense = `nn.Linear`. 뉴런 한 층.
- 교체 = `model.fc = 새 층` (속성 대입). 동결 = `requires_grad = False` (스위치 끄기).
- 순서: **불러오기 → 전부 freeze → head 교체 → `.to(device)` → optimizer에 head만.**
- ML 코스 비유: 집값 예측에서 feature(평수, 방 개수)는 주어지고 `w`, `b`만 학습했다. 여기서는 feature 512개를 pretrained ResNet이 뽑아 주고 마지막 `w`, `b`만 학습한다.
- 연구 코드에서 `backbone`, `self.head = nn.Linear(...)`, `requires_grad_(False)`, `freeze_backbone=True`가 보이면 이 구조다.
