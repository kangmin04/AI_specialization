# model(x)는 무엇이고, 추론 코드는 한 줄씩 뭘 하나

## 헷갈렸던 점

> lab02 Part 1(내 사진 분류)은 타이핑하면서도 어렵다. `unsqueeze`, `no_grad`, `softmax(1)[0]`, `topk`, `.item()`이 각각 뭔지.
> `model(x)`의 shape이 왜 `(1, 1000)`인가? 모델은 그냥 가중치를 받은 거 아닌가?

## 결론

- `model`은 가중치만 있는 숫자 덩어리가 아니라 **층 구조 + 가중치가 합쳐진 하나의 함수**다. `model(x)`는 x를 그 함수에 넣어 계산하라는 뜻이다.
- 출력 shape `(1, 1000)`에서 `1`은 **내가 넣은 사진 장수**, `1000`은 **마지막 층 `fc`의 출력 개수**(ImageNet 클래스 수)다.
- 추론 코드는 Step 1(MNIST)의 예측과 같은 구조다. 출력이 10개가 아니라 1000개이고, `argmax` 대신 `topk`로 상위 5개를 볼 뿐이다.

## 1. model = 구조 + 가중치

```python
model = models.resnet18(weights=weights)
```

1. `resnet18()`: 층을 어떤 순서로 쌓을지 정한 **구조**를 만든다 (이 시점 가중치는 랜덤). Step 1에서 `nn.Sequential(...)`로 직접 쌓았던 것과 같다.
2. `weights=`: 그 층들에 **학습된 숫자**를 채운다.

ML 코스 언어로:
- 구조 = 함수 모양 `f(x) = w·x + b`
- 가중치 = 학습으로 찾은 `w`, `b` 값
- `model(x)` = 그 `w`, `b`로 `f(x)`를 계산

가중치만으로는 계산할 수 없다. "어떤 순서로 곱하고 더할지"(구조)가 있어야 한다.

## 2. model(x)를 부르면 실제로 일어나는 일

ResNet-18의 층을 하나씩 꺼내 `(1, 3, 224, 224)`를 통과시킨 결과:

```
입력              (1,   3, 224, 224)
 conv1         -> (1,  64, 112, 112)   7×7 conv, stride 2 → 크기 절반
 bn1, relu     -> (1,  64, 112, 112)   shape 변화 없음
 maxpool       -> (1,  64,  56,  56)   크기 절반
 layer1        -> (1,  64,  56,  56)   residual block 묶음
 layer2        -> (1, 128,  28,  28)   채널 2배, 크기 절반
 layer3        -> (1, 256,  14,  14)
 layer4        -> (1, 512,   7,   7)
 avgpool       -> (1, 512,   1,   1)   7×7 칸을 평균 내서 1칸으로
 flatten       -> (1, 512)             숫자 512개짜리 벡터
 fc            -> (1, 1000)            ← 최종 출력
```

Step 1의 MNIST CNN과 같은 패턴이다.

```
Step 1:  (B,1,28,28)   → conv/pool        → (B,64,7,7)  → Flatten         → Linear(64*7*7, 10) → (B, 10)
ResNet:  (B,3,224,224) → conv/pool 여러 번 → (B,512,7,7) → avgpool+flatten → Linear(512, 1000)  → (B, 1000)
```

conv를 지나며 공간 크기(H, W)는 줄고 채널(특징 종류)은 늘어나고, 마지막에 벡터로 펴서 dense 층에 넣는다.

### 왜 (1, 1000)인가

```
fc.weight (1000, 512)
fc.bias   (1000,)
```

`fc = Linear(512, 1000)`: 512개를 받아 1000개를 내보낸다. `z = W·a + b` → `(1000×512)·(512) + (1000) = (1000)`.

- 사진 8장을 `(8, 3, 224, 224)`로 넣으면 출력은 `(8, 1000)`. **배치 차원은 모든 층을 그대로 통과한다.**

## 3. 추론 코드 전체 흐름

```
my_photo.jpg (4032×3024)
   │ Image.open().convert("RGB")
   ▼
PIL 이미지
   │ preprocess(img)
   ▼
(3, 224, 224)          이미지 1장
   │ .unsqueeze(0)
   ▼
(1, 3, 224, 224)       "1장짜리 배치"
   │ model(x)
   ▼
(1, 1000)              logits: 클래스별 점수
   │ .softmax(1)
   ▼
(1, 1000)              확률 (합 = 1)
   │ [0]
   ▼
(1000,)                배치 차원 제거
   │ .topk(5)
   ▼
values(확률 5개), indices(클래스 번호 5개)
   │ categories[i]
   ▼
"bobsled 0.253" ...
```

## 4. 한 줄씩

### 모델 불러오기

```python
model = models.resnet18(weights=weights).to(device).eval()
```

| 부분 | 하는 일 |
|---|---|
| `models.resnet18(...)` | 구조 생성 |
| `weights=weights` | 학습된 가중치 다운로드 & 채우기 (캐시 있으면 캐시에서) |
| `.to(device)` | 모든 파라미터를 MPS(맥 GPU)로 이동 |
| `.eval()` | 평가 모드 (BatchNorm이 고정 통계를 쓰게) → [06 문서](06-train-eval-batchnorm.md) |

`.eval()`은 모델 자기 자신을 반환해서 체이닝이 된다.

### 사진 열기

```python
img = Image.open(img_path).convert("RGB")
```

- `Image.open`: PIL 이미지 객체 (아직 텐서 아님)
- `.convert("RGB")`: **채널을 무조건 3개로.** PNG는 RGBA(4채널), 흑백은 L(1채널)일 수 있는데 첫 conv는 3채널만 받는다.

### 전처리 + 배치 차원

```python
x = preprocess(img).unsqueeze(0).to(device)
```

```
preprocess 후: (3, 224, 224)
unsqueeze 후:  (1, 3, 224, 224)
```

- 모델은 항상 **배치** `(B, C, H, W)`를 받는다. 1장은 축이 3개뿐이라 그대로 넣으면 에러.
- `unsqueeze(0)` = **0번 자리에 크기 1짜리 축 끼워 넣기.** numpy의 `x[np.newaxis, ...]`와 같다.
- `.to(device)`: **모델과 입력이 같은 장치에 있어야** 계산된다.

### 예측

```python
with torch.no_grad():
    probs = model(x).softmax(1)[0]
```

`torch.no_grad()`: PyTorch는 기본적으로 `backward()`를 위해 모든 연산을 기록한다 (`no_grad` 밖에서 `model(x).requires_grad`는 `True`였다). 추론에는 필요 없으니 꺼서 **메모리를 아끼고 빨라진다.**

| 코드 | shape | 내용 |
|---|---|---|
| `model(x)` | `(1, 1000)` | **logits**. 실측 앞 5개 `[-3.53, -2.31, -2.23, -3.01, -4.06]`. 음수도 있고 합도 1이 아님 |
| `.softmax(1)` | `(1, 1000)` | 확률로 변환. 0~1, **합 = 1** |
| `[0]` | `(1000,)` | 배치의 0번째(유일한) 사진 꺼내기 = unsqueeze의 반대 |

- 모델 안에 softmax가 없는 이유: 학습 때 `CrossEntropyLoss`가 softmax를 내부에서 처리하기 때문 (Keras `from_logits=True`와 같은 얘기). 사람이 확률을 보고 싶을 때만 직접 붙인다.
- **`softmax(1)`의 `1`은 축 번호.** `(1, 1000)`에서 0번 축 = 배치, 1번 축 = 클래스. "클래스 축을 따라 합이 1이 되게". `softmax(0)`이면 배치 축을 따라 계산해서 사진 1장일 때 전부 1.0이 되는 엉뚱한 결과가 나온다.

### 상위 5개

```python
top5 = probs.topk(5)
```

```
topk values:  tensor([0.2534, 0.0619, 0.0588, 0.0533, 0.0485])   ← 확률
topk indices: tensor([450, 767, 507, 795, 635])                  ← 클래스 번호
```

`argmax`는 1등 번호만, `topk(k)`는 상위 k개의 **값과 위치를 세트로** 준다. ImageNet 성능의 acc@1 / acc@5가 "1등이 정답인가 / 상위 5개 안에 정답이 있는가"다.

### 출력

```python
for p, i in zip(top5.values, top5.indices):
    print(f"{categories[i.item()]:>25s}  {p.item():.3f}")
```

- `zip`: 두 리스트를 짝지어 순회 → `(0.2534, 450)`, `(0.0619, 767)`, ...
- `i`는 숫자 하나 든 **텐서** `tensor(450)`. `.item()`으로 파이썬 정수로 바꿔야 리스트 인덱스로 쓸 수 있다.
- `:>25s`: 25칸 폭에 오른쪽 정렬, `:.3f`: 소수점 셋째 자리.

### 시각화

```python
plt.imshow(img)                                   # 전처리 전 원본 (normalize된 x는 음수라 색이 깨짐)
plt.title(categories[top5.indices[0].item()])     # top-1 이름
plt.axis("off")
plt.show()
```

## 기억할 것

| 헷갈리는 것 | 기억법 |
|---|---|
| `model` | 구조 + 가중치 = 함수. `model(x)` = forward 계산 |
| 출력 shape | `(내가 넣은 장수, 마지막 층 out_features)` |
| `unsqueeze(0)` | 모델은 항상 배치를 받는다. 1장이면 앞에 1을 붙인다 |
| `.eval()` vs `no_grad()` | eval = 층 동작 모드, no_grad = gradient 기록 끄기. 추론 땐 둘 다 |
| `softmax(1)` | 1 = 클래스 축. `(B, 1000)`의 1000 쪽 |
| `[0]` | 배치에서 첫 사진 꺼내기 |
| `.item()` | 원소 1개짜리 텐서 → 파이썬 숫자 |
| `topk` | `.values`(확률)와 `.indices`(번호)가 세트 |

shape이 헷갈리면 층마다 찍어 본다:

```python
h = x
for name, layer in model.named_children():
    if name == "fc":
        h = torch.flatten(h, 1)
    h = layer(h)
    print(f"{name:>8s} -> {tuple(h.shape)}")
```
