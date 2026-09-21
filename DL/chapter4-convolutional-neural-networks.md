# Course 4: Convolutional Neural Networks

> ML Specialization은 끝냈고 신경망/역전파/최적화(Adam, batch norm 등)는 안다고 가정하고 쓰는 노트. Deep Learning Specialization Course 4를 강의 대신 이 문서 하나로 훑을 수 있게 밀도 있게 정리한다. Andrew Ng 특유의 "작은 숫자로 손계산 → 직관 → 일반화" 흐름을 그대로 따라간다.

---

## Week1: Convolution 연산의 기초

### 왜 CNN인가 — fully-connected로는 이미지가 안 된다

1000x1000 컬러 이미지(=1000x1000x3)를 그냥 flatten해서 fully-connected(FC) layer에 넣으면 input feature 수가 300만 개다. 첫 hidden layer에 unit 1000개만 둬도 weight matrix가 30억 개짜리가 된다. 이러면 계산량도 문제고, 데이터도 그만큼 많아야 overfitting을 피하는데 현실적으로 불가능하다. 그래서 이미지 같은 입력은 "파라미터를 재사용"하는 구조, 즉 convolution이 필요하다.

### Edge detection — convolution의 가장 작은 예

가장 기본적인 예로 수직(vertical) edge를 찾는 필터를 보자. $6\times6$ 흑백 이미지에 $3\times3$ filter(=kernel)를 convolve하면 $4\times4$ 결과가 나온다.

```
필터(수직 edge detector):
 1  0 -1
 1  0 -1
 1  0 -1
```

이 필터를 이미지 위에서 한 칸씩 슬라이딩하면서, 겹치는 $3\times3$ 영역과 element-wise 곱한 뒤 다 더한다(이게 "convolution"이라 부르지만 사실 신호처리 정의상으로는 correlation임 — 강의에서도 이 점을 짚고 넘어간다. flip을 안 하니까). 왼쪽이 밝고 오른쪽이 어두운 이미지에 이 필터를 통과시키면, 밝기가 바뀌는 세로 경계 부분에서 값이 크게 나오고 나머지는 0에 가깝게 나온다 — 그게 곧 "edge를 찾았다"는 뜻이다.

- 수직 필터를 90도 돌리면 수평(horizontal) edge detector가 된다.
- $1,0,-1$ 대신 Sobel filter($1,0,-1 / 2,0,-2 / 1,0,-1$), Scharr filter($3,0,-3 / 10,0,-10 / 3,0,-3$) 같은 손으로 짠(hand-engineered) 필터들도 있다 — 기본형과 마찬가지로 열(column) 방향이 $+,0,-$ 패턴이라 여전히 수직 edge를 찾는 필터인데, 가운데 행에 더 큰 가중치를 줘서 노이즈에 좀 더 강건하게 만든 버전이다.
- 근데 핵심은: 이 9개 숫자를 사람이 정하지 않고, **학습으로 알아서 찾게** 하면 어떨까? 그게 CNN의 핵심 아이디어다. filter의 각 원소가 곧 학습 파라미터가 된다.

### Padding

문제 두 가지: (1) convolution을 거칠 때마다 이미지가 계속 작아진다 (2) 모서리/코너 픽셀은 한 번만 계산에 참여해서 정보 손실이 크다. 그래서 이미지 가장자리에 0을 둘러준다(zero-padding), padding 폭을 $p$라 한다.

- **Valid convolution**: padding 없음($p=0$). $n\times n$ 이미지에 $f\times f$필터 → $(n-f+1)\times(n-f+1)$.
- **Same convolution**: 출력 크기가 입력과 같도록 padding. $n+2p-f+1=n$ 을 풀면 $p=\dfrac{f-1}{2}$. 그래서 필터 크기 $f$는 관습적으로 홀수를 쓴다(1,3,5,7...) — 짝수면 padding이 비대칭이 되고, 홀수면 필터에 "중심 픽셀"이 딱 하나 생겨서 위치를 말하기 편하다.

### Stride

필터를 한 칸씩이 아니라 $s$칸씩 건너뛰며 슬라이딩하는 것. stride를 키우면 출력이 그만큼 작아진다.

### 출력 크기 공식 (꼭 외우기)

$$n \times n$$ 입력, $$f \times f$$ 필터, padding $p$, stride $s$ 일 때 출력 크기는

$$\left\lfloor \frac{n+2p-f}{s}+1 \right\rfloor \times \left\lfloor \frac{n+2p-f}{s}+1 \right\rfloor$$

나눗셈이 딱 안 떨어지면 **floor**(내림) 처리한다 — 필터가 이미지 밖으로 완전히 못 나가면 그 위치는 그냥 계산 안 한다는 뜻. 예: $n=7, f=3, p=0, s=2$ → $\lfloor (7-3)/2+1 \rfloor = \lfloor 3 \rfloor = 3$, 그래서 $3\times3$ 출력.

### 3D(볼륨) convolution — 컬러 이미지, 여러 필터

컬러 이미지는 채널이 3개(RGB)라서 $n\times n\times n_c$ 형태다($n_c$ = number of channels = depth). 이때 필터도 $f\times f\times n_c$ 로 채널 수를 맞춰야 한다. 즉 $6\times6\times3$ 이미지에 $3\times3\times3$ 필터를 convolve하면, 27개 숫자(=$3\times3\times3$)를 다 곱해서 더한 **스칼라 하나**가 나온다. 결과는 $4\times4\times1$ — 채널 차원이 없어진다(정확히는 1로 줄어든다).

여기서 필터를 여러 개(예: 수직 edge용 1개 + 수평 edge용 1개, 총 2개) 쓰면, 각 필터의 결과를 쌓아서 $4\times4\times2$가 된다. 즉:

> **다음 층의 채널 수 = 이번 층에서 쓴 필터의 개수**

이게 CNN에서 채널 수가 점점 늘어나는 이유다(반대로 $n_H, n_W$는 점점 줄어든다).

### 한 Conv Layer의 표기법 (layer $l$)

강의 표기 그대로 정리:

- $f^{[l]}$ = filter size
- $p^{[l]}$ = padding
- $s^{[l]}$ = stride
- $n_c^{[l]}$ = 이번 layer에서 쓰는 필터 개수(=출력 채널 수)
- 입력: $n_H^{[l-1]} \times n_W^{[l-1]} \times n_c^{[l-1]}$
- 필터 한 개의 크기: $f^{[l]} \times f^{[l]} \times n_c^{[l-1]}$ (입력 채널 수와 항상 일치해야 함)
- 출력: $n_H^{[l]} \times n_W^{[l]} \times n_c^{[l]}$, 여기서 $n_H^{[l]} = \left\lfloor \dfrac{n_H^{[l-1]}+2p^{[l]}-f^{[l]}}{s^{[l]}}+1 \right\rfloor$ ($n_W$도 동일한 식)
- weight 개수: 필터 1개당 $f^{[l]}\times f^{[l]}\times n_c^{[l-1]}$ 개, 필터가 $n_c^{[l]}$개니까 총 $f^{[l]}\times f^{[l]}\times n_c^{[l-1]}\times n_c^{[l]}$
- bias 개수: 필터당 1개 → $n_c^{[l]}$개
- **파라미터 총합** = $(f^{[l]}\times f^{[l]}\times n_c^{[l-1]}+1)\times n_c^{[l]}$

이미지 개수(batch size) $m$을 포함하면 activation 텐서 shape는 $m \times n_H^{[l]} \times n_W^{[l]} \times n_c^{[l]}$ (TensorFlow 기본 convention).

### 간단한 CNN 예시로 차원 추적 연습

$39\times39\times3$ 입력을 예로 손으로 따라가보자.

| Layer | $f$ | $s$ | $n_c$(필터 개수) | 출력 shape |
|---|---|---|---|---|
| Layer1 (CONV) | 3 | 1 | 10 | $37\times37\times10$ |
| Layer2 (CONV) | 5 | 2 | 20 | $17\times17\times20$ |
| Layer3 (CONV) | 5 | 2 | 40 | $7\times7\times40$ |
| Flatten | - | - | - | $1960$ (=$7\times7\times40$) |

이렇게 쭉 따라가다가 마지막엔 flatten해서 logistic/softmax unit에 넣는다. 이게 습관이 되어야 함 — 새 아키텍처를 볼 때마다 층마다 shape을 손으로 계산해보는 것.

### Pooling — 학습 파라미터가 없는 층

- **Max pooling**: 영역 안에서 가장 큰 값만 남긴다. 직관: "이 영역 어딘가에 특정 feature가 있었는지"만 살아남으면 되니까 max를 쓴다.
- **Average pooling**: 영역 평균. 최근엔 아주 깊은 네트워크 마지막에 (7x7x1000 같은 걸 1x1x1000으로 줄일 때) 가끔 쓰이는 정도.
- 보통 $f=2, s=2$를 많이 써서 $n_H, n_W$를 절반으로 줄인다.
- **핵심**: pooling은 학습되는 파라미터가 0개다(그냥 고정된 연산). 그래서 파라미터 개수 표에서 pooling layer 줄은 항상 0.
- 채널마다 독립적으로 적용되므로 $n_c$는 pooling 전후로 안 바뀐다.

### LeNet-5 스타일 예시 — activation shape/size/파라미터 표

$32\times32\times3$ 입력 기준으로 강의에서 쓰는 표 형태를 재현하면:

| Layer | Activation shape | Activation size | # parameters |
|---|---|---|---|
| Input | $(32,32,3)$ | 3,072 | 0 |
| CONV1 ($f=5,s=1$, 필터 8개) | $(28,28,8)$ | 6,272 | $(5\times5\times3+1)\times8=608$ |
| POOL1 ($f=2,s=2$) | $(14,14,8)$ | 1,568 | 0 |
| CONV2 ($f=5,s=1$, 필터 16개) | $(10,10,16)$ | 1,600 | $(5\times5\times8+1)\times16=3216$ |
| POOL2 ($f=2,s=2$) | $(5,5,16)$ | 400 | 0 |
| FC3 (unit 120) | $(120,1)$ | 120 | $400\times120+120=48120$ |
| FC4 (unit 84) | $(84,1)$ | 84 | $120\times84+84=10164$ |
| Softmax (10 class) | $(10,1)$ | 10 | $84\times10+10=850$ |

관찰 포인트: 층이 깊어질수록 activation size는 대체로 서서히 줄어들고(너무 급격히 줄면 정보 손실이 큼), 파라미터 개수는 **conv layer보다 FC layer에 훨씬 많이 몰려있다.** conv 자체는 파라미터가 의외로 적다는 게 포인트.

### 왜 Convolution인가 (FC 대비 두 가지 장점)

1. **Parameter sharing**: 이미지 왼쪽 위 모서리에서 유용한 필터(예: 수직 edge detector)는 오른쪽 아래에서도 똑같이 유용하다. 그래서 같은 필터(=같은 weight 세트)를 이미지 전체 위치에 재사용한다. FC라면 위치마다 다른 weight를 새로 배워야 하는데, conv는 "한 번 배운 필터"를 이미지 전체에 그대로 슬라이딩만 시킨다.
2. **Sparsity of connections**: 출력의 각 값은 입력의 아주 일부(필터 크기만큼)에만 의존한다. 예를 들어 출력의 왼쪽 위 픽셀은 입력 오른쪽 아래 픽셀과 아무 관계가 없다. FC라면 모든 출력이 모든 입력과 연결되는 것과 대조적이다.

이 두 가지 덕분에 파라미터 수가 확 줄어서 (1) 학습 데이터가 적어도 overfitting이 덜 되고 (2) translation invariance(이미지가 몇 픽셀 이동해도 같은 특징을 잡아냄)를 자연스럽게 얻는다.

---

## Week2: 유명 아키텍처들

### Classic Networks

- **LeNet-5** (1998): 손글씨 숫자 인식용. 위 표처럼 conv-pool을 반복하다가 FC로 마무리. 당시엔 ReLU가 없어서 sigmoid/tanh를 썼고, 지금 기준으로는 작은 네트워크.
- **AlexNet** (2012): LeNet과 구조는 비슷한 철학이지만 훨씬 크다(파라미터 6천만 개 수준). ReLU를 쓰고, 딥러닝이 이미지 분야에서 폭발적으로 뜨는 계기가 된 논문. Local Response Normalization(LRN)을 썼는데 요즘은 잘 안 씀.
- **VGG-16**: "16"은 학습 가능한 layer 수(conv+FC). 특징이 아주 단순 — conv는 전부 $3\times3, s=1,$ same padding, pool은 전부 $2\times2, s=2$ max pooling. 필터 개수가 64→128→256→512로 매 블록마다 정확히 2배씩 늘어난다. 구조가 규칙적이라 이해하기 쉽지만 파라미터가 약 1억 3800만 개로 매우 큼.

### ResNet — Residual Network

**문제**: 이론적으로는 layer를 계속 쌓으면 training error가 계속 줄어들어야 하는데, 실제로는 아주 깊은 plain network(residual 없는 network)는 어느 순간부터 training error가 오히려 다시 올라간다(vanishing/exploding gradient 때문). ResNet은 이걸 해결한다.

**Residual block**: 원래 forward propagation은

$$z^{[l+1]} = W^{[l+1]}a^{[l]}+b^{[l+1]}, \quad a^{[l+1]} = g(z^{[l+1]})$$
$$z^{[l+2]} = W^{[l+2]}a^{[l+1]}+b^{[l+2]}, \quad a^{[l+2]} = g(z^{[l+2]})$$

인데, residual block은 $a^{[l]}$을 **두 layer 건너뛰어서** 더해준다("skip connection" 또는 "shortcut"):

$$a^{[l+2]} = g\left(z^{[l+2]} + a^{[l]}\right)$$

이런 block을 계속 쌓은 게 ResNet(예: ResNet-50, ResNet-101, ResNet-152).

**왜 작동하나 (직관)**: $W^{[l+2]}, b^{[l+2]}$가 학습을 통해 0에 가깝게 수렴한다고 해보자(L2 regularization 등으로). 그러면 $a^{[l+2]} = g(a^{[l]}) = a^{[l]}$ (ReLU이고 $a^{[l]}\ge0$이라면). 즉 **identity function을 배우는 게 너무 쉬워진다.** 그래서 최소한 "이 block을 추가해도 성능이 나빠지지는 않는다"가 보장되고, 운이 좋으면 추가 block이 뭔가 더 유용한 걸 배워서 성능이 더 좋아진다. plain network는 layer를 추가하면 identity조차 배우기 어려워서(weight를 아주 정교하게 맞춰야 함) 깊어질수록 오히려 손해를 볼 수 있다.

*주의*: $z^{[l+2]}+a^{[l]}$를 더하려면 두 텐서의 차원이 같아야 한다. 그래서 ResNet 안에서는 same-padding conv를 많이 써서 차원을 유지하고, 차원이 다를 수밖에 없는 지점(예: 채널 수가 바뀌는 지점)에는 $a^{[l]}$에 $W_s$라는 별도의 행렬을 곱해서 차원을 맞춰준다: $a^{[l+2]} = g(z^{[l+2]} + W_s a^{[l]})$. $W_s$는 학습되거나, 그냥 zero-padding으로 채널만 맞추는 고정된 방식일 수도 있다.

### 1x1 Convolution (Network in Network)

$1\times1$ 필터가 $n_H\times n_W\times n_c$ 짜리 볼륨을 볼 때, 공간적으로는 아무것도 안 섞고(한 픽셀만 보니까) **채널 방향으로만** FC layer 하나를 통과시키는 것과 같다. 그래서 두 가지 용도로 쓴다.

1. **채널 수 줄이기(bottleneck)**: $28\times28\times192$ 를 $1\times1, s=1$ 필터 32개로 convolve하면 $28\times28\times32$가 된다 — 공간 크기는 그대로, 채널만 확 줄인다. 뒤에 오는 비싼 연산(예: $5\times5$ conv)의 계산량을 크게 줄여준다.
2. 채널 수를 유지하거나 늘리는 데도 쓸 수 있다 — 결국 "채널 차원의 비선형 조합"을 배우는 매우 저렴한 layer.

### Inception Network

**동기**: conv layer 설계할 때 filter size($1\times1$? $3\times3$? $5\times5$?)나 pooling을 넣을지 매번 골라야 하는데, Inception은 "다 해보고 결과를 이어붙이자(concatenate)"는 아이디어다. 한 Inception module 안에서 $1\times1, 3\times3, 5\times5$ conv와 max pooling을 **병렬로** 다 계산해서 채널 방향으로 쌓는다.

**문제**: $5\times5$ conv를 채널이 많은 볼륨에 그대로 쓰면 연산량이 어마어마하다. 예: $28\times28\times192$ 입력에 $5\times5$ same conv, 필터 32개 → 곱셈 연산이 약 1억 2천만 번.

**해결(bottleneck)**: $5\times5$ conv 앞에 $1\times1$ conv를 끼워서 채널을 먼저 줄인다. $192 \to 16$(1x1) $\to 32$(5x5). 이러면 연산량이 약 1240만 번으로 **10분의 1** 수준까지 줄어드는데, 성능은 거의 그대로 유지된다(bottleneck layer가 너무 작지만 않으면). 이런 bottleneck을 여러 크기의 conv 앞마다 넣은 게 하나의 Inception module이고, 이걸 여러 개 쌓은 게 GoogLeNet/Inception network.

### MobileNet — Depthwise Separable Convolution

모바일/임베디드처럼 연산 자원이 부족한 환경을 위한 구조. 핵심은 일반 convolution을 두 단계로 쪼개는 것.

- **일반 convolution**: $n\times n\times n_c$ 입력에 $f\times f\times n_c$ 필터 $n_c'$개 → 연산량이 대략 (출력 크기) $\times$ (필터 크기) $\times$ (입력 채널) $\times$ (필터 개수) 만큼.
- **Depthwise convolution**: 필터를 채널별로 따로 적용한다(입력 채널당 필터 1개, $f\times f\times1$짜리를 $n_c$개). 채널끼리 안 섞고 공간 정보만 필터링. 출력 채널 수 = 입력 채널 수(그대로).
- **Pointwise convolution**: 그 결과에 $1\times1\times n_c$ 필터 $n_c'$개를 적용해서 채널을 원하는 개수로 섞어 조합한다.

두 개를 합쳐 depthwise separable convolution이라 하는데, 일반 conv 대비 연산량이 대략

$$\frac{1}{n_c'} + \frac{1}{f^2}$$

배로 줄어든다. 예를 들어 $f=3, n_c'=512$면 대략 $\frac{1}{512}+\frac{1}{9} \approx 0.113$, 즉 약 10배 가까이 저렴해진다. MobileNet v2는 여기에 residual connection과 "expansion layer"(bottleneck을 먼저 넓혔다가 depthwise 하고 다시 줄이는 구조)를 추가한 버전.

### EfficientNet (개념만)

네트워크의 성능은 resolution(입력 해상도), depth(층 개수), width(채널 개수) 세 가지를 키우면 좋아지는데, 셋 중 하나만 무작정 키우면 곧 한계에 부딪힌다. EfficientNet은 계산 자원 예산이 주어졌을 때 이 세 축을 **같이, 균형있게** 얼마나 키울지 정하는 compound scaling 규칙을 제안한 것 — 자원이 넉넉하면 EfficientNet-B7처럼 크게, 부족하면 B0처럼 작게 같은 계열 안에서 고를 수 있다.

### 오픈소스 구현 / Transfer Learning 실무 조언

> **처음부터 아키텍처를 새로 설계하려 하지 말고, 검증된 오픈소스 구현+pretrained weight를 가져다 쓰는 게 실무에서 훨씬 합리적이다.** (ResNet, MobileNet, Inception 등 GitHub에 코드+weight가 다 있음)

전이학습(transfer learning) 시 데이터 양에 따라 freeze 전략이 달라진다.

- **데이터가 아주 적을 때**: 마지막 softmax(출력) layer만 새로 만들고, 나머지 전체를 freeze. 사실상 pretrained network를 고정된 feature extractor로 쓰는 것. (팁: freeze된 부분의 activation을 미리 한 번 계산해서 디스크에 캐싱해두면, 그 이후엔 그 feature 위에서 shallow model만 빠르게 여러 번 학습할 수 있다.)
- **데이터가 어느 정도 있을 때**: 앞쪽(저수준 feature: edge, 색깔, texture 등 범용적인 것)은 freeze하고, 뒤쪽 layer 몇 개 + 새 출력 layer를 직접 학습(fine-tuning).
- **데이터가 아주 많을 때**: pretrained weight를 초기값(initialization)으로만 쓰고 네트워크 전체를 fine-tuning.

### Data Augmentation

이미지 데이터는 거의 항상 데이터가 "충분히 많다"고 느껴지지 않아서 augmentation이 사실상 기본값이다.

- Mirroring(좌우 반전), Random cropping, Rotation, Shearing, Local warping
- Color shifting: RGB 채널 값에 작은 값을 랜덤하게 더하거나 빼서 조명/색감 변화에 강건하게 만듦. 세 채널에 완전히 무작위로 더하는 대신, 그 이미지에서 실제로 자주 나타나는 색 변화 방향(픽셀들의 RGB 값에 PCA를 돌려서 구한 주성분) 위주로 왜곡을 주는 정교한 버전을 PCA color augmentation이라 부른다(AlexNet 논문에서 제안).
- 실무에서는 augmentation을 별도 CPU 스레드에서 미리 만들어두고, GPU는 그동안 학습만 계속 돌리는 파이프라인(hard disk → distortion → mini-batch → training)을 많이 씀.

---

## Week3: Object Detection

### Localization vs Detection vs Classification

- **Image classification**: "이 이미지가 뭐냐" 한 개 레이블만 출력.
- **Classification with localization**: 객체가 1개라고 가정하고, 클래스 + bounding box(위치)까지 출력.
- **Detection**: 이미지 안에 객체가 여러 개(개수도 모름) 있을 수 있고, 각각의 클래스+위치를 다 찾아야 함.

### Bounding box를 위한 출력 벡터 $y$

Classification with localization 문제의 출력 $y$를 예로 들면(클래스 3개: 보행자/자동차/오토바이 + 배경):

$$y = \begin{bmatrix} p_c \\ b_x \\ b_y \\ b_h \\ b_w \\ c_1 \\ c_2 \\ c_3 \end{bmatrix}$$

- $p_c$: 객체가 존재하는지(1) 배경인지(0)
- $b_x, b_y$: bounding box 중심의 좌표(이미지 크기로 정규화, 보통 0~1)
- $b_h, b_w$: bounding box의 높이/너비(마찬가지로 이미지 크기 기준으로 정규화). 객체가 이미지 안에 통째로 들어있다고 가정하는 이 기본 설정에서는 보통 0~1 사이 값이다. *(주의: 뒤에 나올 YOLO처럼 grid cell 크기 기준으로 정규화하면 객체가 cell보다 커서 1을 넘는 경우가 생기는데, 그건 해당 부분에서 따로 짚는다.)*
- $c_1,c_2,c_3$: 각 클래스일 확률(원-핫)

$p_c=0$이면($=$배경) 나머지 값은 뭐가 나오든 loss 계산에서 무시(don't care)한다.

### Landmark Detection

bounding box 대신 "특정 점들의 좌표"를 직접 출력하게 할 수도 있다. 예: 얼굴 인식에서 눈꼬리 위치, 표정 인식에서 입/눈 주변 landmark 64개 등. 출력이 $(l_{1x}, l_{1y}), (l_{2x}, l_{2y}), \dots$ 형태가 될 뿐 나머지 아이디어(레이블링을 일관되게 하는 게 중요함)는 동일. Snapchat 필터 같은 게 이 방식을 씀.

### Sliding Window Detection과 Convolutional 구현

**원시적인 방법**: 작은 window를 이미지 위에서 이동시키며 매번 ConvNet에 넣어서 "이 window 안에 객체가 있냐"를 판단. window 크기를 여러 개 시도해야 하고, stride를 촘촘히 하면 정확하지만 계산량이 엄청나게 커진다(느림).

**Convolutional 구현(OverFeat 아이디어)**: FC layer를 $1\times1$ conv로 바꿔서 표현하면, sliding window의 각 위치별로 따로따로 forward pass를 돌릴 필요 없이 **큰 이미지 하나를 통째로 한 번만** ConvNet에 통과시켜도 된다. 그러면 출력 볼륨의 각 위치가 원본 이미지의 각 window 위치에 대응하는 결과가 되어서, 겹치는 계산을 자연스럽게 공유하게 된다 — 훨씬 빠름.

### YOLO (You Only Look Once)

**Grid 나누기**: 이미지를 $S\times S$ (예: $3\times3$ 또는 실제론 $19\times19$) grid cell로 나누고, 각 cell마다 위에서 본 $y$ 벡터($p_c,b_x,b_y,b_h,b_w,c_1,\dots$)를 출력하게 학습시킨다. 객체의 **중심점**이 속한 cell 하나만 그 객체에 대한 책임을 진다(라벨링 규칙). 최종 출력 텐서 shape은 $S\times S\times(5+\text{클래스수})$ — 여기에 anchor box까지 쓰면 채널이 더 늘어난다.

- $b_x,b_y$는 해당 cell 내부 기준 상대 좌표(0~1), $b_h,b_w$는 cell 크기 기준 비율(1보다 커질 수 있음 — 객체가 cell보다 클 수 있으니까).
- 이 방식의 장점: bounding box 좌표가 신경망의 "출력"이라서(sliding window처럼 classifier를 여러 번 돌리는 게 아님) 좀 더 유연하고 빠르다(실시간 가능해서 이름이 You Only Look **Once**).

**IoU (Intersection over Union)**: 두 bounding box가 얼마나 겹치는지 측정하는 지표.

$$\text{IoU} = \frac{\text{교집합 넓이}}{\text{합집합 넓이}}$$

보통 IoU $\ge 0.5$면 "맞다(correct)"고 판단하는 관례를 쓴다(더 엄격하게 하려면 0.6, 0.7도 씀).

**Non-max Suppression (NMS)**: 객체 하나에 대해 여러 grid cell/anchor가 겹쳐서 비슷한 box를 여러 개 예측하는 문제를 정리하는 후처리.
1. $p_c$(또는 $p_c\times$클래스확률)가 threshold보다 낮은 box는 다 버림.
2. 남은 box 중 $p_c$가 가장 높은 것을 하나 선택하고 출력에 포함.
3. 그 box와 IoU가 threshold(예: 0.5) 이상인 나머지 box들은 다 버림(같은 객체를 가리키는 중복이라고 판단).
4. 남은 box가 없어질 때까지 2~3 반복. 클래스가 여러 개면 클래스별로 독립적으로 NMS를 돌린다.

**Anchor Box**: 한 grid cell 안에 중심이 겹치는 객체가 2개 이상 있으면(예: 사람과 자동차가 겹쳐 있음) box 하나로는 표현이 안 된다. 그래서 미리 정해둔 모양(세로로 긴 anchor, 가로로 긴 anchor 등) 여러 개를 각 cell에 배정하고, 각 실제 객체는 자기 실제 box와 **IoU가 가장 높은 anchor**에 할당된다. 그러면 $y$ 벡터가 anchor 개수만큼 이어붙여지고(anchor 2개면 $y$ 길이가 2배), 출력 텐서는 $S\times S\times(\text{anchor 개수}\times(5+\text{클래스수}))$가 된다. anchor box shape은 보통 k-means 같은 걸로 학습 데이터의 box 모양들을 클러스터링해서 정한다(YOLOv2 이후).

### Region Proposal (R-CNN 계열, 간단히)

Sliding window는 명백히 배경인 영역까지 다 검사한다는 낭비가 있다. R-CNN(Regions with CNN)은 먼저 "객체가 있을 법한 후보 영역"만 segmentation 알고리즘(예: selective search — 색/텍스처/크기가 비슷한 인접 영역끼리 점점 합쳐나가며 물체 후보 blob들을 만들어내는 고전적인(딥러닝 이전의) 이미지 분할 알고리즘)으로 추려낸 뒤, 그 영역들에만 CNN을 돌린다.

- **R-CNN**: region propose → 영역마다 각각 classify. 느림(영역 하나하나 forward pass).
- **Fast R-CNN**: propose는 그대로 하되, 전체 이미지에 conv를 한 번만 돌리고 그 feature map 위에서 영역별 classify(convolutional 구현). 훨씬 빠름.
- **Faster R-CNN**: region proposal 자체도 별도 알고리즘이 아니라 conv network로 하게 만듦 — 그래도 YOLO보다는 보통 느림.

실무적으로는 요즘 1-stage(YOLO 계열)가 속도 때문에 더 많이 쓰이지만, R-CNN 계열의 "propose 후 정제" 아이디어 자체는 알아둘 가치가 있다.

### Semantic Segmentation과 U-Net

Object detection은 "박스"로 위치를 표시하지만, semantic segmentation은 **픽셀 하나하나**에 클래스를 매긴다(예: 의료 영상에서 장기 경계를 픽셀 단위로 구분). 출력이 입력과 같은 $n_H\times n_W$ 크기를 가져야 하는데, 일반 conv+pool은 계속 크기를 줄이는 쪽으로만 동작하므로 **다시 키우는 연산**이 필요하다.

**Transpose Convolution**: 일반 conv의 반대 방향 연산. 작은 입력에 필터를 대응시켜 큰 출력을 만든다(입력의 한 픽셀 값을 필터 크기만큼 "펼쳐서" 출력에 곱해 넣고, 겹치는 부분은 더함). 이걸 이용해서 작아진 feature map을 다시 원래 해상도로 upsampling한다.

**U-Net 구조**: 이름처럼 U자 모양.
- 앞부분(encoder, contracting path): 일반 conv+pooling으로 점점 작고 채널이 많은 feature map으로 압축(공간 정보는 줄고 "무엇이 있는지"에 대한 정보는 늘어남).
- 뒷부분(decoder, expanding path): transpose convolution으로 다시 키워서 원래 해상도로 복원.
- **Skip connection**: encoder의 각 단계에서 나온 feature map을 decoder의 대응하는 단계에 직접 연결(concatenate)한다. 이유: decoder만으로 upsampling하면 "무엇이 있는지"는 알아도 "정확히 어디에 있었는지"(고해상도 공간 정보)가 이미 pooling으로 손실된 상태라서, encoder의 고해상도 feature를 다시 가져다 붙여줘야 픽셀 단위로 정밀한 경계를 그릴 수 있다. ResNet의 skip connection과 목적은 다르지만("gradient 잘 흐르게" vs "고해상도 정보 보존") 아이디어 형태는 비슷하다.

---

## Week4: Face Recognition & Neural Style Transfer

### Verification vs Recognition

- **Face Verification**: "이 사람이 A가 맞습니까?" — 1:1 매칭 문제. 입력 이미지+주장하는 신원 → yes/no.
- **Face Recognition**: "이 사람이 DB에 있는 K명 중 누구입니까?" — 1:K 매칭이라서 verification보다 훨씬 어렵다(오류 가능성이 K배).

### One-shot Learning 문제

회사 직원 얼굴 인식 같은 경우, 사람마다 학습 이미지가 **딱 1장**밖에 없는 경우가 흔하다. 이런 상황에서 일반적인 "클래스별로 softmax 학습" 방식은 안 맞는다(데이터가 너무 적고, 새 사람이 추가될 때마다 네트워크를 다시 학습해야 함). 그래서 "클래스를 분류"하는 대신 **"두 이미지 사이의 유사도 함수 $d(\text{img1}, \text{img2})$를 학습"**하는 방향으로 문제를 바꾼다. $d$가 threshold $\tau$보다 작으면 "같은 사람"이라고 판단.

### Siamese Network

같은 CNN(같은 weight를 공유)에 이미지 두 장을 각각 통과시켜서 각각 embedding 벡터 $f(x^{(1)}), f(x^{(2)})$를 얻고, 그 사이 거리를 유사도로 쓴다.

$$d(x^{(1)}, x^{(2)}) = \lVert f(x^{(1)}) - f(x^{(2)}) \rVert_2^2$$

목표: 같은 사람 사진 쌍은 $d$가 작게, 다른 사람 사진 쌍은 $d$가 크게 나오도록 $f$(=embedding 함수, network의 파라미터)를 학습하는 것.

### Triplet Loss

**Anchor(A)**, **Positive(P, 같은 사람)**, **Negative(N, 다른 사람)** 세 장을 한 세트(triplet)로 학습한다. 원하는 조건:

$$\lVert f(A)-f(P) \rVert^2 + \alpha \le \lVert f(A)-f(N) \rVert^2$$

$\alpha$는 **margin**으로, 이게 없으면 $f$가 항상 0을 출력하는 등 trivial한 답으로 loss를 0으로 만들어버릴 수 있어서, "A-N 거리가 A-P 거리보다 최소 $\alpha$만큼은 더 커야 한다"고 강제하는 역할이다. 이걸 손실함수로 만들면:

$$\mathcal{L}(A,P,N) = \max\left(\lVert f(A)-f(P) \rVert^2 - \lVert f(A)-f(N) \rVert^2 + \alpha,\ 0 \right)$$

$\max(\cdot, 0)$을 쓰는 이유는 "조건을 이미 만족했으면(음수가 되면) loss를 더 줄일 필요 없이 그냥 0"으로 처리하기 위함이다(SVM에서 쓰는 hinge loss, 즉 $\max(\text{마진 위반량}, 0)$ 형태와 같은 꼴).

**Hard Triplet 선택이 중요한 이유**: triplet을 완전히 무작위로 뽑으면 $d(A,P)$가 이미 $d(A,N)$보다 훨씬 작은, 너무 쉬운 조합이 대부분이라서(A,P가 애초에 다르게 생긴 사람도 아니고 랜덤 N은 대체로 확연히 다르게 생김) gradient descent가 별로 배울 게 없다. 그래서 **hard negative**(다른 사람인데 $d(A,N)$이 $d(A,P)$에 가깝게 나오는, 헷갈리는 조합)를 의도적으로 골라 학습에 사용해야 학습이 효율적으로 진행된다.

### Neural Style Transfer

Content 이미지 $C$의 내용을 유지하면서 Style 이미지 $S$의 화풍을 입힌 Generated 이미지 $G$를 만드는 문제. $G$의 픽셀 값 자체를 gradient descent로 최적화한다(네트워크의 weight가 아니라 이미지 픽셀이 학습 대상이라는 게 특이한 점).

전체 cost function:

$$J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S,G)$$

**Content Cost**: pretrained ConvNet(보통 VGG)의 **중간 정도 깊이**의 layer $l$을 하나 골라서, $C$와 $G$를 각각 통과시켰을 때 나오는 activation $a^{[l](C)}, a^{[l](G)}$가 비슷하면 내용이 비슷하다고 본다.

$$J_{content}(C,G) = \frac{1}{2}\lVert a^{[l](C)} - a^{[l](G)} \rVert^2$$

너무 얕은 layer를 쓰면 픽셀 값 자체를 그대로 베끼라는 압박이 되고, 너무 깊은 layer를 쓰면 "이게 개다/고양이다" 같은 추상적 내용만 맞으면 되니까 layer 선택이 중간 정도가 적당하다.

**Style Cost와 Gram Matrix**: "style"이란 것을, 어떤 layer의 여러 채널(=서로 다른 필터가 감지한 feature)들이 **서로 얼마나 함께(correlate) 나타나는지**로 정의한다. 이걸 수치화한 게 Gram matrix다. layer $l$의 activation을 채널 $c$, 위치 $(i,j)$에 대해 $a_{i,j,c}^{[l]}$라 하면,

$$G_{kk'}^{[l]} = \sum_{i=1}^{n_H}\sum_{j=1}^{n_W} a_{i,j,k}^{[l]}\, a_{i,j,k'}^{[l]}$$

이 $n_c^{[l]}\times n_c^{[l]}$ 짜리 행렬이 채널 $k$와 $k'$이 같이 활성화되는 경향(비대각 원소)과 각 채널 자체의 활성화 정도(대각 원소, 강도)를 담는다. $S$와 $G$ 각각에서 이 Gram matrix를 구해서($G^{[l](S)}, G^{[l](G)}$) 그 차이를 줄이는 게 style cost:

$$J_{style}^{[l]}(S,G) = \frac{1}{(2n_H n_W n_c)^2}\sum_{k,k'}\left(G_{kk'}^{[l](S)}-G_{kk'}^{[l](G)}\right)^2$$

보통 layer 하나가 아니라 여러 layer의 style cost를 가중합해서 쓴다(얕은 layer는 색감/질감 같은 저수준 style, 깊은 layer는 더 큰 패턴의 style을 잡아내므로 섞어 쓰는 게 결과가 좋음):

$$J_{style}(S,G) = \sum_l \lambda^{[l]} J_{style}^{[l]}(S,G)$$

**깊은 layer가 뭘 학습하는지 시각화**: ConvNet의 각 unit을 최대로 활성화시키는 입력 패치를 찾아보면(gradient ascent로 이미지를 최적화), 얕은 layer의 unit들은 edge/색깔/간단한 texture에 반응하고, 층이 깊어질수록 점점 더 복잡한 패턴(질감 조합 → 사물의 부분 → 개/자동차 같은 전체 개념)에 반응하는 걸 관찰할 수 있다. 이게 "content는 중간~깊은 layer, style은 얕은 layer 위주로 여러 층을 섞어서" 쓰는 이유의 근거가 된다.

### Face Verification을 Binary Classification으로 풀기

Triplet loss 말고 다른 접근도 있다: 두 이미지를 같은 Siamese network에 통과시켜 embedding $f(x^{(i)}), f(x^{(j)})$을 얻은 뒤, 이 두 embedding을 원소별로 비교한 값(예: $|f(x^{(i)})_k - f(x^{(j)})_k|$)을 입력으로 로지스틱 회귀 unit 하나에 넣어서 "같은 사람(1)/다른 사람(0)"을 곧바로 이진분류로 학습시키는 방식이다. Triplet처럼 (A,P,N) 세 장을 한 세트로 묶을 필요 없이 (이미지 쌍, 레이블) 데이터만 있으면 되고, 실전 배포할 때는 등록된 직원들의 embedding을 미리 한 번 계산해서 저장(precompute)해두면, 새 이미지가 들어올 때마다 그 저장된 embedding과 비교만 하면 되므로 편하다(DeepFace 논문에서 쓴 방식).

### Convolution의 1D/3D 일반화

지금까지 다룬 건 전부 2D 이미지($n_H\times n_W\times n_c$)에 대한 convolution인데, 같은 아이디어가 차원을 바꿔도 그대로 적용된다.

- **1D convolution**: EKG(심전도) 신호처럼 시간축으로 값 하나가 쭉 이어지는 데이터($n\times n_c$ 형태)에도 필터를 슬라이딩시켜 convolution을 적용할 수 있다(이런 시계열 데이터는 뒤 course에서 배울 RNN으로도 다룰 수 있다).
- **3D convolution**: CT 촬영 영상처럼 가로/세로에 (물리적인) 깊이 축까지 있는 volume 데이터($n_H\times n_W\times n_D\times n_c$)에는 3D 필터($f\times f\times f\times n_c$)를 슬라이딩시켜 적용한다.
- 차원이 늘어나도 출력 크기 공식은 각 축에 대해 똑같이 $\lfloor (n+2p-f)/s+1 \rfloor$ 로 계산하면 된다.

---

## 핵심 요약

- Convolution은 **작은 필터를 이미지 전체에 재사용**함으로써 FC 대비 파라미터를 극적으로 줄인다 (parameter sharing + sparsity of connections).
- 출력 크기는 항상 $\lfloor (n+2p-f)/s+1 \rfloor$로 손계산 가능해야 하고, 필터 개수가 다음 layer의 채널 수가 된다.
- Pooling은 학습 파라미터가 없는 고정 연산(max/average)이고, 공간 크기만 줄인다.
- ResNet의 skip connection은 "identity를 배우기 쉽게" 만들어서 아주 깊은 네트워크도 학습이 안정적으로 되게 해준다.
- $1\times1$ conv는 채널 방향 FC이자 bottleneck(연산량 절감) 도구로 Inception/MobileNet 등 여러 아키텍처의 핵심 부품이다.
- 실무에서는 새 아키텍처를 밑바닥부터 설계하기보다 **오픈소스+pretrained weight로 transfer learning**하는 게 기본값이고, freeze 범위는 보유 데이터 양에 반비례한다.
- Detection 계열: sliding window의 비효율을 conv 공유(OverFeat)와 grid 기반 단일 forward pass(YOLO)로 해결. IoU/NMS/anchor box는 겹치는 예측을 정리하는 장치들.
- Segmentation은 encoder(축소)-decoder(transpose conv로 확대) + skip connection(U-Net)으로 픽셀 단위 출력을 만든다.
- Face recognition은 classification이 아니라 **embedding 거리 학습** 문제로 재정의(Siamese network, triplet loss, 혹은 embedding 차이를 넣는 binary classification)해서 one-shot 상황에 대응한다.
- Style transfer는 이미지의 weight가 아니라 **픽셀 자체를 최적화**하며, content는 activation 유사도로, style은 Gram matrix(채널 간 상관관계) 유사도로 정의한다.

## 헷갈리기 쉬운 점

- **"Convolution" vs 실제 신호처리의 convolution**: 딥러닝에서 말하는 convolution은 필터를 뒤집지(flip) 않는 cross-correlation이다. 이름만 convolution.
- **Valid vs Same**: valid는 padding 없음($p=0$, 출력이 작아짐), same은 입력=출력 크기가 되도록 padding($p=(f-1)/2$). "same"이 padding=0이라고 착각하기 쉬운데 반대다.
- **필터 개수 = 다음 layer의 채널 수**이지, 필터 크기($f$)가 채널 수를 결정하는 게 아니다. 필터 자체의 깊이(채널 차원)는 항상 **입력**의 채널 수와 같아야 한다(그래서 필터 shape에 $n_c^{[l-1]}$이 들어감).
- **Pooling은 파라미터가 0개**라서 "파라미터 개수 표"를 만들 때 pooling layer 줄을 빼먹거나 반대로 값을 넣는 실수를 하기 쉽다.
- **ResNet의 skip connection vs U-Net의 skip connection**: 둘 다 "이전 layer 결과를 더 나중 layer로 직접 전달"한다는 점은 같지만, ResNet은 **더하기(add)**이고 목적은 gradient 흐름/identity 학습이다. U-Net은 **이어붙이기(concatenate)**이고 목적은 encoder의 고해상도 공간 정보를 decoder에 보존하는 것. 연산 자체도 다르다.
- **IoU threshold와 NMS threshold는 별개의 값**일 수 있다(하나는 "이 예측이 맞다고 볼 기준", 하나는 "두 예측이 같은 객체를 가리킨다고 볼 기준"). 같은 0.5를 쓰는 경우가 많아서 같은 개념으로 착각하기 쉽다.
- **Anchor box는 "미리 정해둔 모양의 틀"**이지 실제 예측된 box가 아니다. 네트워크는 각 anchor에 대한 **보정값**(offset)을 출력하고, 학습 시 각 실제 객체는 IoU가 가장 큰 anchor 하나에만 할당된다(한 cell에 anchor가 여러 개 있어도 객체 하나당 보통 anchor 하나만 책임짐).
- **Triplet loss의 margin $\alpha$는 하이퍼파라미터**이지 학습되는 파라미터가 아니다. margin이 없으면 embedding이 전부 같은 점으로 collapse해도 loss가 0이 될 수 있다는 걸 막는 장치라는 걸 기억할 것.
- **Style transfer에서 최적화 대상은 이미지 $G$의 픽셀 값**이지 네트워크의 weight가 아니다. VGG 등 ConvNet의 weight는 학습 내내 고정(frozen)이고, activation을 뽑아내는 용도로만 쓰인다.
- **Gram matrix의 대각 원소**는 채널 하나 자체의 활성화 강도(그 feature가 이미지에 얼마나 강하게 등장하는지)이고, **비대각 원소**가 두 채널이 "함께" 나타나는 정도(style의 핵심)다. 둘을 혼동해서 대각 원소만 style이라고 오해하기 쉽다.

---

## Lab 예제 — 직접 짜보기

> 프레임워크 없이 numpy로 CNN의 핵심 연산을 직접 짠다. 뼈대의 `TODO`를 채우고 **체크포인트**와 비교해보자. 정답은 접혀있다.

### Lab 1. Convolution forward를 for문으로 직접

**목표**: `zero_pad` → `conv_single_step` → `conv_forward` 순서로 conv layer의 forward를 구현한다. 강의의 6×6 세로 edge 예제를 재현하고, padding/stride에 따른 출력 shape이 공식 $\lfloor\frac{n+2p-f}{s}\rfloor+1$과 맞는지 확인한다.

```python
import numpy as np
import matplotlib.pyplot as plt

def zero_pad(X, pad):
    # X: (m, n_H, n_W, n_C) — 높이/너비 축만 pad
    return None   # TODO: np.pad

def conv_single_step(a_slice, W, b):
    # a_slice, W: (f, f, n_C_prev), b: (1, 1, 1)
    return None   # TODO: 원소별 곱의 합 + b

def conv_forward(A_prev, W, b, stride=1, pad=0):
    # A_prev: (m, n_H_prev, n_W_prev, n_C_prev), W: (f, f, n_C_prev, n_C), b: (1, 1, 1, n_C)
    m, n_H_prev, n_W_prev, _ = A_prev.shape
    f, _, _, n_C = W.shape
    n_H = None    # TODO: 출력 크기 공식
    n_W = None    # TODO
    Z = np.zeros((m, n_H, n_W, n_C))
    A_pad = zero_pad(A_prev, pad)
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                # TODO: stride를 고려해 f x f 창(a_slice)을 잘라내고
                #       필터 c마다 conv_single_step(a_slice, W[..., c], b[..., c])
                pass
    return Z

# 강의 예제: 왼쪽 밝고(10) 오른쪽 어두운(0) 6x6 이미지
img = np.zeros((6, 6)); img[:, :3] = 10
vertical = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
horizontal = vertical.T
Wk = np.stack([vertical, horizontal], axis=-1)[:, :, None, :]   # (3,3,1,2): 필터 2개
out = conv_forward(img[None, :, :, None], Wk, np.zeros((1, 1, 1, 2)))
print("세로 edge 필터 결과:\n", out[0, :, :, 0])
print("가로 edge 필터 결과:\n", out[0, :, :, 1])

np.random.seed(1)
A_prev = np.random.randn(2, 5, 7, 4)
W = np.random.randn(3, 3, 4, 8); b = np.random.randn(1, 1, 1, 8)
for pad, stride in [(0, 1), (1, 1), (1, 2)]:
    print(f"pad={pad}, stride={stride} -> Z.shape = {conv_forward(A_prev, W, b, stride, pad).shape}")

# 원 이미지에 Sobel 필터
yy, xx = np.mgrid[0:64, 0:64]
circle = ((xx - 32) ** 2 + (yy - 32) ** 2 < 20 ** 2).astype(float)
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
edges = conv_forward(circle[None, :, :, None], sobel_x[:, :, None, None], np.zeros((1, 1, 1, 1)), pad=1)
fig, ax = plt.subplots(1, 2)
ax[0].imshow(circle, cmap="gray"); ax[0].set_title("input")
ax[1].imshow(edges[0, :, :, 0], cmap="gray"); ax[1].set_title("Sobel vertical edge")
plt.show()
```

**체크포인트**
- 세로 필터: 가운데 두 열이 30인 `4x4` (`[0, 30, 30, 0]` × 4) — 강의 그림 그대로. 가로 필터는 전부 0 (가로 방향 edge가 없으니까).
- shape: `(2, 3, 5, 8)`, `(2, 5, 7, 8)`(same padding), `(2, 3, 4, 8)`.
- Sobel 결과: 원의 왼쪽 테두리(어둡→밝)는 음수라 검게, 오른쪽 테두리(밝→어둡)는 양수라 희게 — 필터 부호가 edge의 "방향"(밝→어둡 vs 어둡→밝)까지 구분한다.

<details>
<summary>정답 보기</summary>

```python
def zero_pad(X, pad):
    return np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=0)

def conv_single_step(a_slice, W, b):
    return np.sum(a_slice * W) + b.item()

def conv_forward(A_prev, W, b, stride=1, pad=0):
    m, n_H_prev, n_W_prev, _ = A_prev.shape
    f, _, _, n_C = W.shape
    n_H = (n_H_prev + 2 * pad - f) // stride + 1
    n_W = (n_W_prev + 2 * pad - f) // stride + 1
    Z = np.zeros((m, n_H, n_W, n_C))
    A_pad = zero_pad(A_prev, pad)
    for i in range(m):
        for h in range(n_H):
            vs = h * stride
            for w in range(n_W):
                hs = w * stride
                a_slice = A_pad[i, vs:vs + f, hs:hs + f, :]
                for c in range(n_C):
                    Z[i, h, w, c] = conv_single_step(a_slice, W[..., c], b[..., c])
    return Z
```

</details>

### Lab 2. Pooling + LeNet-5 shape/파라미터 계산기

**목표**: max/avg pooling을 구현하고, 레이어 목록만 넣으면 강의의 "activation shape / size / 파라미터 수" 표를 자동으로 뽑아주는 함수를 만든다. conv가 FC보다 파라미터가 얼마나 적은지(parameter sharing) 숫자로 체감한다.

```python
import numpy as np

def pool_forward(A_prev, f, stride, mode="max"):
    m, n_H_prev, n_W_prev, n_C = A_prev.shape
    n_H = (n_H_prev - f) // stride + 1
    n_W = (n_W_prev - f) // stride + 1
    A = np.zeros((m, n_H, n_W, n_C))
    for h in range(n_H):
        for w in range(n_W):
            window = None   # TODO: (m, f, f, n_C) 창
            A[:, h, w, :] = None   # TODO: 높이/너비 축(1,2)으로 max 또는 mean. 채널은 따로!
    return A

x = np.array([[1, 3, 2, 1], [2, 9, 1, 1], [1, 3, 2, 3], [5, 6, 1, 2]], dtype=float)
print("max pool:\n", pool_forward(x[None, :, :, None], 2, 2)[0, :, :, 0])
print("avg pool:\n", pool_forward(x[None, :, :, None], 2, 2, "avg")[0, :, :, 0])

def summarize(input_shape, layers):
    h, w, c = input_shape
    print(f"{'layer':>10} | {'shape':>14} | {'size':>6} | params")
    print(f"{'input':>10} | {str((h, w, c)):>14} | {h*w*c:>6} | 0")
    total = 0
    for layer in layers:
        kind = layer[0]
        if kind == "conv":
            _, f, s, p, n_c = layer
            params = None   # TODO: (f*f*이전채널 + 1) * 필터 수
            h, w, c = None, None, None   # TODO
        elif kind == "pool":
            _, f, s = layer
            params = 0      # pooling은 학습 파라미터 없음
            h, w = None, None            # TODO
        elif kind == "fc":
            params = None   # TODO: (입력 unit 수 + 1) * 출력 unit 수
            h, w, c = 1, 1, layer[1]
        total += params
        print(f"{kind:>10} | {str((h, w, c)):>14} | {h*w*c:>6} | {params}")
    print("총 파라미터:", total)

# LeNet-5 스타일: (conv f, s, p, 필터수) / (pool f, s) / (fc units)
summarize((32, 32, 3), [("conv", 5, 1, 0, 6), ("pool", 2, 2), ("conv", 5, 1, 0, 16), ("pool", 2, 2),
                        ("fc", 120), ("fc", 84), ("fc", 10)])
print("같은 32x32x3 -> 28x28x6 을 FC로 만들면 파라미터:", 32 * 32 * 3 * 28 * 28 * 6)
```

**체크포인트**
- max pool `[[9, 2], [6, 3]]`, avg pool `[[3.75, 1.25], [3.75, 2]]`.
- 표: conv1 `(28,28,6)` 456개 → pool `(14,14,6)` → conv2 `(10,10,16)` 2416개 → pool `(5,5,16)` → fc 48120 / 10164 / 850, 총 62006개.
- 첫 conv 레이어 456개 vs 같은 입출력을 FC로 하면 **약 1445만 개**. 파라미터 대부분이 뒤쪽 FC에 몰려있고, activation size는 점점 줄어든다는 것도 표에서 보인다.

<details>
<summary>정답 보기</summary>

```python
# pool_forward
            window = A_prev[:, h * stride:h * stride + f, w * stride:w * stride + f, :]
            A[:, h, w, :] = window.max(axis=(1, 2)) if mode == "max" else window.mean(axis=(1, 2))

# summarize
        if kind == "conv":
            _, f, s, p, n_c = layer
            params = (f * f * c + 1) * n_c
            h, w, c = (h + 2 * p - f) // s + 1, (w + 2 * p - f) // s + 1, n_c
        elif kind == "pool":
            _, f, s = layer
            params = 0
            h, w = (h - f) // s + 1, (w - f) // s + 1
        elif kind == "fc":
            params = (h * w * c + 1) * layer[1]
            h, w, c = 1, 1, layer[1]
```

</details>

### Lab 3. Residual block이 왜 깊어져도 괜찮은가 + 연산량 계산

**목표**: (1) W가 0이어도 residual block은 identity를 그대로 통과시킨다는 것, (2) 작은 가중치로 블록을 50개 쌓으면 plain network는 신호가 사라지지만 residual은 유지된다는 걸 확인한다. (3) 1×1 conv bottleneck(Inception)과 depthwise separable conv(MobileNet)의 곱셈 횟수를 직접 계산한다.

```python
import numpy as np

def relu(z): return np.maximum(0, z)

def plain_block(a, W1, W2):
    return relu(W2 @ relu(W1 @ a))

def residual_block(a, W1, W2):
    return None   # TODO: skip connection — 두 번째 relu "전에" a를 더한다

np.random.seed(0)
a = np.abs(np.random.randn(8, 1))
Z0 = np.zeros((8, 8))
print("W=0일 때 plain block 출력:", plain_block(a, Z0, Z0).ravel()[:4], "...")
print("W=0일 때 residual block 출력 == 입력?", np.allclose(residual_block(a, Z0, Z0), a))

def deep_forward(a, n_blocks, residual, scale=0.1):
    rng = np.random.RandomState(0)
    for _ in range(n_blocks):
        W1, W2 = rng.randn(8, 8) * scale, rng.randn(8, 8) * scale
        a = residual_block(a, W1, W2) if residual else plain_block(a, W1, W2)
    return np.linalg.norm(a)

for n in [1, 5, 20, 50]:
    print(f"블록 {n:>2}개 - plain |a|: {deep_forward(a, n, False):.2e}, residual |a|: {deep_forward(a, n, True):.2e}")

def conv_cost(n_out, f, n_c_in, n_c_out):
    return None   # TODO: 출력 원소 수(n_out*n_out*n_c_out) x 원소 하나당 곱셈 수(f*f*n_c_in)

def depthwise_separable_cost(n_out, f, n_c_in, n_c_out):
    depthwise = None   # TODO: 채널마다 f x f 필터 하나씩
    pointwise = None   # TODO: 1x1 x n_c_in 필터를 n_c_out개
    return depthwise + pointwise

normal, sep = conv_cost(4, 3, 3, 5), depthwise_separable_cost(4, 3, 3, 5)
print(f"강의 예시(6x6x3 -> 4x4x5): normal {normal}, separable {sep}, 비율 {sep/normal:.3f} (= 1/n_c' + 1/f^2 = {1/5 + 1/9:.3f})")
normal, sep = conv_cost(28, 3, 256, 512), depthwise_separable_cost(28, 3, 256, 512)
print(f"큰 레이어: normal {normal:,}, separable {sep:,}, 비율 {sep/normal:.3f}")
bottleneck = conv_cost(28, 1, 192, 16) + conv_cost(28, 5, 16, 32)
print(f"Inception 1x1 bottleneck: 5x5 직접 {conv_cost(28, 5, 192, 32):,} vs 1x1 거쳐서 {bottleneck:,}")
```

**체크포인트**
- W=0: plain은 전부 0, residual은 입력 그대로(`True`). "identity 함수를 배우기 쉽다" = 최악이어도 블록을 건너뛴 것과 같다는 뜻.
- 블록 50개: plain `|a|` ≈ `1e-76` (신호 소멸), residual ≈ `4.4` (유지).
- 곱셈 수: 강의 예시 2160 vs 672(비율 0.311), 큰 레이어는 비율 0.113 — 채널이 많을수록 MobileNet의 이득이 커진다. Inception: 약 1.2억 vs 1244만, 약 10배 절약.

<details>
<summary>정답 보기</summary>

```python
def residual_block(a, W1, W2):
    return relu(W2 @ relu(W1 @ a) + a)

def conv_cost(n_out, f, n_c_in, n_c_out):
    return n_out * n_out * n_c_out * f * f * n_c_in

def depthwise_separable_cost(n_out, f, n_c_in, n_c_out):
    depthwise = n_out * n_out * n_c_in * f * f
    pointwise = n_out * n_out * n_c_out * n_c_in
    return depthwise + pointwise
```

</details>

### Lab 4. IoU + Non-max Suppression (YOLO 후처리)

**목표**: 두 박스의 IoU를 계산하고, 같은 물체에 겹쳐 나온 박스들 중 가장 확신 높은 것만 남기는 NMS를 구현한다.

```python
import numpy as np
import matplotlib.pyplot as plt

def iou(box1, box2):
    # box = (x1, y1, x2, y2): 왼쪽 위, 오른쪽 아래 좌표
    inter = None   # TODO: 교집합 영역 (안 겹치면 0 — max(0, ...) 조심)
    union = None   # TODO: 넓이1 + 넓이2 - 교집합
    return inter / union

print("IoU 겹침:", iou((2, 1, 4, 3), (1, 2, 3, 4)))
print("IoU 안 겹침:", iou((1, 1, 2, 2), (3, 3, 4, 4)))
print("IoU 동일:", iou((1, 1, 3, 3), (1, 1, 3, 3)))

def non_max_suppression(boxes, scores, score_threshold=0.6, iou_threshold=0.5):
    # TODO:
    # 1) score < score_threshold 인 박스는 버린다
    # 2) 남은 것 중 score 최고인 박스를 keep에 넣고
    # 3) 그 박스와 IoU >= iou_threshold 인 박스들은 제거
    # 4) 남은 박스가 없을 때까지 2~3 반복
    keep = []
    return keep

boxes = np.array([[50, 50, 150, 150], [55, 45, 155, 145], [60, 60, 160, 170],
                  [200, 80, 280, 200], [205, 85, 285, 195], [100, 200, 130, 230]], dtype=float)
scores = np.array([0.9, 0.75, 0.8, 0.7, 0.85, 0.4])
keep = non_max_suppression(boxes, scores)
print("NMS 후 남은 박스:", keep)

fig, ax = plt.subplots()
for i, (x1, y1, x2, y2) in enumerate(boxes):
    color = "red" if i in keep else "lightgray"
    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=color, lw=2))
    ax.text(x1, y1 - 3, f"{scores[i]:.2f}", color=color)
ax.set_xlim(0, 320); ax.set_ylim(260, 0); ax.set_title("Non-max suppression (red = kept)")
plt.show()
```

**체크포인트**
- IoU: `0.1429`(=1/7), `0.0`, `1.0`.
- NMS 결과 `[0, 4]` — 왼쪽 물체는 0.9짜리, 오른쪽 물체는 0.85짜리만 남고, 0.4짜리 작은 박스는 score threshold에서 먼저 탈락.
- 실제 YOLO에서는 클래스마다 따로 NMS를 돌린다는 것도 기억.

<details>
<summary>정답 보기</summary>

```python
def iou(box1, box2):
    xi1, yi1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    xi2, yi2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union

def non_max_suppression(boxes, scores, score_threshold=0.6, iou_threshold=0.5):
    idx = [i for i in np.argsort(-scores) if scores[i] >= score_threshold]
    keep = []
    while idx:
        best = idx.pop(0)
        keep.append(best)
        idx = [i for i in idx if iou(boxes[best], boxes[i]) < iou_threshold]
    return keep
```

</details>

### Lab 5. Triplet loss와 Style transfer의 Gram matrix

**목표**: face recognition의 triplet loss를 구현해서 easy/hard negative에서 loss가 어떻게 다른지 보고, 거리 threshold로 verification을 해본다. 이어서 Gram matrix와 style cost를 구현하고, "style은 위치와 무관한 채널 간 상관관계"라는 말을 실험으로 확인한다.

```python
import numpy as np

def triplet_loss(f_A, f_P, f_N, alpha=0.2):
    # f_*: (m, 128) 임베딩
    pos_dist = None   # TODO: ||f(A) - f(P)||^2  (샘플별)
    neg_dist = None   # TODO: ||f(A) - f(N)||^2
    return None       # TODO: sum(max(pos - neg + α, 0))

def l2_normalize(x): return x / np.linalg.norm(x, axis=1, keepdims=True)
np.random.seed(0)
person = l2_normalize(np.random.randn(3, 128))                       # 3명의 "진짜 얼굴" 임베딩
f_A = l2_normalize(person + 0.03 * np.random.randn(3, 128))          # anchor
f_P = l2_normalize(person + 0.03 * np.random.randn(3, 128))          # 같은 사람 다른 사진
f_N_easy = l2_normalize(np.random.randn(3, 128))                     # 전혀 다른 사람
f_N_hard = l2_normalize(person + 0.05 * np.random.randn(3, 128))     # 닮은 사람
print("triplet loss (easy negative):", triplet_loss(f_A, f_P, f_N_easy))
print("triplet loss (hard negative):", triplet_loss(f_A, f_P, f_N_hard))

def verify(f_img, f_db, threshold=0.7):
    d = np.linalg.norm(f_img - f_db)
    return d, d < threshold
print("같은 사람:", verify(f_A[0], f_P[0]), " 다른 사람:", verify(f_A[0], f_N_easy[0]))

def gram_matrix(A):
    # A: (n_H, n_W, n_C) -> G: (n_C, n_C), G[k, k'] = 채널 k와 k'가 같은 위치에서 같이 켜지는 정도
    return None   # TODO: (n_C, n_H*n_W)로 펴서 A A^T

def style_cost_layer(a_S, a_G):
    n_H, n_W, n_C = a_S.shape
    return None   # TODO: ||G_S - G_G||_F^2 / (2 n_H n_W n_C)^2

def content_cost(a_C, a_G):
    n_H, n_W, n_C = a_C.shape
    return np.sum((a_C - a_G) ** 2) / (4 * n_H * n_W * n_C)

np.random.seed(1)
a_S = np.random.rand(8, 8, 4)
a_S[..., 1] = a_S[..., 0] * 0.9           # 채널 0과 1이 항상 같이 켜지는 "스타일"
a_shuffled = a_S.reshape(-1, 4)[np.random.permutation(64)].reshape(8, 8, 4)   # 픽셀 위치만 섞음
print("Gram matrix of style:\n", np.round(gram_matrix(a_S), 2))
print("위치만 섞은 이미지 - style cost:", style_cost_layer(a_S, a_shuffled), ", content cost:", round(content_cost(a_S, a_shuffled), 4))
print("완전 다른 이미지 - style cost:", round(style_cost_layer(a_S, np.random.rand(8, 8, 4)), 6))
```

**체크포인트**
- easy negative loss = 0 (이미 margin 밖이라 학습에 기여 X), hard negative loss ≈ 0.17. 그래서 학습 때 **hard triplet**을 골라야 한다는 것.
- verify: 같은 사람 거리 ≈ 0.42 → True, 다른 사람 ≈ 1.25 → False.
- Gram matrix에서 `G[0,1]`이 크다(채널 0, 1이 같이 켜짐). 위치만 섞은 이미지는 style cost ≈ 0 (1e-33)인데 content cost는 0이 아님 — Gram matrix는 "어디에"는 버리고 "무엇과 무엇이 같이"만 본다.

<details>
<summary>정답 보기</summary>

```python
def triplet_loss(f_A, f_P, f_N, alpha=0.2):
    pos_dist = np.sum((f_A - f_P) ** 2, axis=1)
    neg_dist = np.sum((f_A - f_N) ** 2, axis=1)
    return np.sum(np.maximum(pos_dist - neg_dist + alpha, 0))

def gram_matrix(A):
    n_C = A.shape[-1]
    A_unrolled = A.reshape(-1, n_C).T      # (n_C, n_H*n_W)
    return A_unrolled @ A_unrolled.T

def style_cost_layer(a_S, a_G):
    n_H, n_W, n_C = a_S.shape
    G_S, G_G = gram_matrix(a_S), gram_matrix(a_G)
    return np.sum((G_S - G_G) ** 2) / (2 * n_H * n_W * n_C) ** 2
```

</details>
