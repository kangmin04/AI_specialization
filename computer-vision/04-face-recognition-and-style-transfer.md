# Face Recognition & Neural Style Transfer

Deep Learning Specialization Course 4 (Convolutional Neural Networks)의 마지막 챕터인 "Special Applications: Face recognition & Neural style transfer" 파트를 정리한다. 지금까지 배운 CNN을 그대로 응용해서 실제로 재밌는 응용 문제 두 개(얼굴 인식, 스타일 변환)를 어떻게 푸는지 보는 챕터라 흥미로웠다.

---

# Part 1. Face Recognition

## 1. Face Verification vs Face Recognition

먼저 용어부터 구분하고 가야 한다. 둘이 비슷해 보이는데 난이도가 꽤 다르다.

- **Face Verification (1:1 문제)**: "이 사람이 진짜 이 사람이 맞는가?"를 확인하는 문제. 입력으로 이미지 하나와 주장하는 이름(ID)이 같이 들어오고, 출력은 맞다/아니다(binary) 하나면 된다. 예를 들면 회사 출입 시스템에서 사원증 찍고 얼굴 스캔해서 "이 사람이 그 사원증 주인이 맞는지" 확인하는 것.
- **Face Recognition (1:K 문제)**: 데이터베이스에 K명의 얼굴이 등록되어 있고, 입력 이미지 하나가 주어졌을 때 "이 사람이 K명 중 누구인가" (혹은 아무도 아닌가)를 판별하는 문제.

여기서 중요한 포인트는 **Recognition이 Verification보다 훨씬 어렵다**는 것. 왜냐하면 Verification은 정확도가 예를 들어 99%만 되어도 "100번에 1번 틀림" 정도로 나쁘지 않은데, Recognition은 이 verification을 K번 반복하는 셈이라 K가 커질수록 에러가 누적된다. K=100이면 정확도 99%짜리 verifier로는 실용적으로 못 쓸 만큼 틀릴 확률이 커진다. 그래서 face recognition 시스템을 만들려면 verification 단계에서 요구되는 정확도 자체가 훨씬 높아야 한다 (99.9% 이상 등).

## 2. One-Shot Learning 문제

회사 출입 시스템 같은 걸 만든다고 하면, 직원 한 명당 보통 얼굴 사진이 딱 1장밖에 없는 경우가 많다 (사원증 등록 사진 1장). 이런 상황을 **one-shot learning**이라고 부른다 - 한 클래스(사람)당 학습 샘플이 1개뿐인 상황에서 학습해야 하는 문제.

여기서 일반적인 방식, 즉 사람 얼굴 이미지를 입력으로 받아서 softmax로 "직원 A, B, C, ... 중 누구" 를 분류하는 CNN을 그대로 쓰면 왜 안 될까?

1. 학습 데이터가 사람당 1장이라 CNN을 처음부터 학습시키기엔 데이터가 턱없이 부족하다 (deep learning은 데이터가 많아야 잘 되는데).
2. 더 근본적인 문제: 새로운 직원이 입사하면 output class 개수(K)가 바뀐다. 그러면 softmax output layer 자체를 다시 설계하고 전체 네트워크를 재학습시켜야 한다. 직원이 늘어날 때마다 이 짓을 반복할 수는 없다.

그래서 방향을 완전히 바꾼다. "이 사람이 누구인지 직접 분류하자"가 아니라, **"두 이미지가 얼마나 비슷한지(similarity/difference)를 학습하는 함수를 만들자"**로 문제를 재정의한다. 이 함수를 d(img1, img2) = 두 이미지 사이의 차이 정도(degree of difference) 라고 하면:

- d(img1, img2)가 threshold(예: tau)보다 작으면 → 같은 사람
- d(img1, img2)가 threshold보다 크면 → 다른 사람

이렇게 similarity function을 한 번 잘 학습해두면, 새 직원이 추가되어도 재학습 없이 그냥 새 직원 사진과 입력 이미지를 d 함수에 넣어서 비교만 하면 된다. one-shot 문제를 similarity learning 문제로 바꾼 것이 핵심 아이디어.

## 3. Siamese Network

그럼 이 d(img1, img2) 함수는 어떻게 학습시킬까? 여기서 **Siamese Network** 구조가 등장한다.

구조 자체는 단순하다. 이미지를 받아서 벡터(embedding)를 출력하는 CNN을 하나 만든다 (마지막에 softmax 대신 FC layer로 128차원 같은 벡터를 뽑아내도록). 이 CNN을 encoding function이라고 부르고 f(x)로 표기한다. 즉 f(x^(1))은 이미지 x^(1)을 128차원 벡터로 인코딩한 결과다.

**같은 CNN(같은 파라미터)을 이미지 두 장에 각각 적용**해서 f(x^(1)), f(x^(2)) 두 개의 embedding을 얻는다. 이게 "Siamese"(샴 쌍둥이)라는 이름이 붙은 이유 - 두 네트워크가 똑같이 생겼고 weight도 공유한다.

그 다음 두 embedding 사이의 거리를 d(x^(1), x^(2)) = ||f(x^(1)) - f(x^(2))||^2 (L2 distance)로 정의한다.

**학습 목표**는:
- x^(1), x^(2)가 같은 사람 사진이면 → ||f(x^(1)) - f(x^(2))||^2 가 작아지도록
- x^(1), x^(2)가 다른 사람 사진이면 → ||f(x^(1)) - f(x^(2))||^2 가 커지도록

즉 CNN의 파라미터(weight)를 학습시키는 목표 자체가 "좋은 embedding을 만드는 것"으로 바뀐 것. 이제 이 목표를 구체적인 loss function으로 어떻게 표현하느냐가 다음 문제다.

## 4. Triplet Loss

Siamese network를 학습시키는 대표적인 방법이 **Triplet Loss** (FaceNet 논문에서 제안).

이름 그대로 이미지 3장을 한 세트(triplet)로 구성해서 학습한다:
- **Anchor (A)**: 기준이 되는 이미지
- **Positive (P)**: Anchor와 같은 사람 이미지
- **Negative (N)**: Anchor와 다른 사람 이미지

우리가 원하는 것은 d(A,P)는 작고, d(A,N)은 커지는 것. 즉:

```
||f(A) - f(P)||^2 <= ||f(A) - f(N)||^2
```

근데 이 식을 그대로 두면 문제가 있다. f를 그냥 항상 0벡터로만 만들어버리는(zero function, 모든 이미지를 같은 점으로 매핑) trivial한 해도 이 부등식을 만족시켜버린다 (양변이 둘 다 0이 되니까). 이런 의미없는 해를 막기 위해 **margin(alpha)**이라는 여유값을 넣는다:

```
||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + alpha <= 0
```

margin alpha는 "d(A,P)가 d(A,N)보다 최소한 alpha만큼은 작아야 한다"는 걸 강제해서, positive/negative 사이에 일정 간격을 두도록 만든다. 이게 없으면 두 거리가 거의 같아지는 것도 "만족"으로 쳐버려서 구분력이 떨어진다.

이걸 loss function 형태로 만들면 (음수가 되어도 상관없으니 0으로 클리핑):

```
L(A, P, N) = max( ||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + alpha, 0 )
```

이게 **Triplet Loss**. 전체 cost function은 이 L을 학습셋의 모든 triplet에 대해 합산한 것.

- 이미 부등식을 만족하는 경우(즉 d(A,P) - d(A,N) + alpha <= 0) → loss = 0 (더 배울 게 없음)
- 만족 못 하는 경우 → loss가 양수로 나와서 gradient descent가 f(A), f(P)는 가깝게, f(A), f(N)은 멀게 밀어붙임

**Hard triplet 선택이 왜 중요한가**: 학습 데이터에서 triplet (A, P, N)을 그냥 랜덤하게 뽑으면, 대부분의 경우 이미 d(A,P) + alpha <= d(A,N) 조건을 (별 노력 없이도) 만족해버린다. 왜냐하면 서로 다른 사람의 랜덤한 두 사진은 애초에 많이 다르게 생겼을 확률이 높기 때문. 이러면 gradient descent가 계속 loss=0인 (이미 쉬운) 예제만 보게 되어서 실질적으로 학습이 거의 일어나지 않는다.

그래서 **hard triplet**, 즉 d(A,P)가 d(A,N)에 최대한 가까운(구분하기 어려운) triplet들을 의도적으로 골라서 학습시켜야 네트워크가 진짜로 미세한 차이를 구분하도록 압박받는다. 예를 들면 비슷하게 생긴 다른 사람 두 명을 negative pair로 잡거나, 조명/각도가 달라서 어려운 같은 사람 사진을 positive pair로 잡는 식. 이런 hard example mining이 triplet loss 학습의 실질적인 성능을 좌우하는 핵심 노하우다.

## 5. Binary Classification으로 푸는 방법

Triplet loss 말고 face verification 문제를 그냥 **binary classification**으로 바꿔서 푸는 방법도 있다 (DeepFace, FaceNet 계열에서 쓰는 또다른 방식).

아이디어는: 두 이미지를 각각 Siamese network(같은 CNN)에 통과시켜서 embedding f(x^(1)), f(x^(2))를 얻은 다음, 이 두 embedding을 조합해서 logistic regression 하나에 넣어 "같은 사람 (1)" / "다른 사람 (0)" 을 예측하게 학습시키는 것.

두 embedding을 조합하는 방법으로는 각 차원별 차이의 절댓값을 쓰거나, chi-square 유사도 형태를 쓰는 등 여러 변형이 있다:

```
y_hat = sigmoid( sum_k w_k * |f(x^(1))_k - f(x^(2))_k| + b )
```

이렇게 하면 결국 "같은 사람인지 다른 사람인지"를 곧바로 supervised binary label(1/0)로 학습시키는 구조가 되고, 학습이 끝나면 새로운 사람이 추가되어도 그 사람의 이미지 1장을 미리 인코딩(embedding 계산)해서 데이터베이스에 저장해두고, 나중에 입력 이미지가 들어오면 그것도 인코딩해서 저장된 것과 비교만 하면 되니까 매번 전체 forward pass 두 번을 새로 돌릴 필요도 없다 (embedding을 미리 캐싱). Triplet loss 방식과 binary classification 방식 둘 다 실제로 잘 동작하는 접근법이라고 한다.

## 6. 실무 팁: Pretrained Embedding 쓰기

Face recognition 시스템을 처음부터 직접 학습시키려면 사람 수백만 명 * 여러 장의 데이터셋이 필요하다 (실제 FaceNet, DeepFace 논문들도 대규모 데이터셋으로 학습). 이런 규모의 데이터셋을 개인이나 작은 회사가 모으기는 현실적으로 매우 어렵다.

그래서 실무에서는 **다른 사람이 이미 대규모 데이터로 학습시켜놓은 pretrained face embedding 모델(및 weight)을 가져다 쓰는 것이 일반적**이다. 이런 pretrained encoding function f를 그대로 가져오면, 우리는 처음부터 학습시킬 필요 없이 각 사람의 사진 1장을 f에 통과시켜 embedding만 뽑아서 데이터베이스에 저장해두고, 이후 verification/recognition은 그 embedding들 사이의 거리 비교만으로 처리할 수 있다. "학습은 이미 잘 된 모델을 빌려쓰고, 우리는 그 위에서 응용만 한다"는 transfer learning적인 접근이 face recognition 분야에서는 거의 표준적인 방식이라고 봐도 될 듯하다.

---

# Part 2. Neural Style Transfer

## 1. Neural Style Transfer가 뭔가

**Neural Style Transfer**는 두 장의 이미지를 입력으로 받는다:
- **Content image (C)**: 그림의 "내용"(구도, 형태, 어떤 객체가 있는지)을 가져올 이미지
- **Style image (S)**: 그림의 "화풍/질감"(색감, 붓터치, 패턴 등)을 가져올 이미지

이 둘을 합쳐서 **Generated image (G)**를 만드는데, G는 C의 내용을 유지하면서 S의 스타일로 그려진 이미지가 되는 게 목표다. 예를 들면 사진 한 장(content)을 고흐 그림(style) 화풍으로 다시 그리는 것.

## 2. Deep ConvNet이 학습하는 Feature들의 의미

Neural style transfer가 어떻게 가능한지 이해하려면, 먼저 CNN의 각 레이어가 이미지에서 어떤 걸 "보고" 있는지에 대한 직관이 필요하다.

CNN의 각 레이어에서 특정 hidden unit을 최대로 활성화시키는 이미지 패치들을 시각화해보면 (여러 이미지에 대해 그 unit의 activation이 가장 큰 패치들을 모아보는 방식):

- **얕은 레이어(shallow layer)**: 상대적으로 단순한 저수준(low-level) feature를 감지한다. 예를 들면 edge(경계선), 특정 방향의 선, 특정 색깔, 간단한 texture(질감) 패턴 같은 것들.
- **깊은 레이어(deep layer)**: 앞선 레이어들의 정보가 누적되면서 점점 더 복잡하고 추상적인 패턴을 감지하게 된다. 얼굴의 일부, 바퀴 모양, 특정 동물의 털 패턴, 심지어 특정 종류의 객체 전체(개, 자동차 등) 같은 고수준(high-level) 개념까지 인식하게 된다.

즉 네트워크가 깊어질수록 receptive field가 넓어지고 표현하는 정보의 추상화 수준도 높아진다는 것. 이 직관이 있어야 "content는 어느 레이어의 activation을 보고 비교해야 하는지", "style은 어떻게 정의해야 하는지"에 대한 다음 내용이 자연스럽게 이해된다.

## 3. Cost Function 구성

Neural style transfer의 핵심은, generated image G가 얼마나 "좋은지"를 평가하는 cost function J(G)를 잘 정의하는 것이다. 이 cost는 두 부분의 가중합으로 구성된다:

```
J(G) = alpha * J_content(C, G) + beta * J_style(S, G)
```

- J_content: G가 C와 내용적으로 얼마나 비슷한가
- J_style: G가 S와 스타일적으로 얼마나 비슷한가
- alpha, beta: 두 cost 사이의 상대적 중요도를 조절하는 hyperparameter

### Content Cost Function

미리 학습된(pretrained) CNN (보통 VGG 같은 네트워크)을 가져와서, 그 중간의 특정 hidden layer l을 하나 고른다. 이 layer는 너무 얕지도 깊지도 않은 중간 정도를 주로 쓴다 (너무 얕으면 픽셀 값 자체를 거의 그대로 요구하게 되고, 너무 깊으면 "내용"이라고 부르기엔 너무 추상화된 고수준 개념만 남기 때문).

C와 G를 각각 이 CNN에 통과시켜서 layer l에서의 activation을 뽑는다: a^[l](C), a^[l](G). Content cost는 이 두 activation 사이의 L2 distance(제곱 차이의 합)로 정의한다:

```
J_content(C, G) = (1/2) * ||a^[l](C) - a^[l](G)||^2
```

즉 "layer l이 뽑아낸 특징이 C와 G에서 비슷하면 두 이미지의 내용이 비슷하다고 보자"는 아이디어. activation 벡터 값 자체가 비슷해지도록 G를 최적화하는 것이 content를 보존하는 방법이 된다.

### Style Cost Function

Style은 조금 더 창의적인 정의가 필요하다. 여기서 등장하는 게 **Gram Matrix**.

특정 layer l의 activation은 (n_H, n_W, n_C) 형태 (높이, 너비, 채널 수)를 가진다. 여기서 채널(channel)은 각각 서로 다른 종류의 feature(예: 특정 방향의 edge, 특정 texture 등)를 감지한다고 볼 수 있다.

**Style을 "채널들 사이의 상관관계(correlation)"로 정의**하는 게 핵심 아이디어다. 왜 상관관계인가 하면: 예를 들어 어떤 채널이 "세로줄 무늬"를 감지하고 다른 채널이 "황금빛 색조"를 감지한다고 하면, 이 두 채널이 이미지의 같은 위치에서 자주 같이 활성화되는지(상관관계가 높은지) 아닌지가 그 이미지의 화풍(예: 반 고흐 특유의 세로 붓터치 + 노란 색감의 조합)을 나타낸다고 볼 수 있다는 것. 즉 "어떤 스타일적 요소들이 함께 나타나는 경향이 있는가"가 style의 정의가 된다.

이 채널 간 상관관계를 행렬로 표현한 것이 Gram Matrix G^[l] (n_C x n_C 크기)이다. layer l의 activation을 채널별로 펼친 벡터라고 할 때:

```
G^[l]_kk' = sum over all (i,j) of  a^[l]_ijk * a^[l]_ijk'
```

(k, k'는 채널 인덱스, i,j는 공간 위치). 즉 채널 k와 채널 k'의 activation을 모든 위치에서 곱해서 더한 것 - 이게 두 채널이 얼마나 "같이 활성화되는지"를 나타내는 값이다.

Style image S와 generated image G 각각에 대해 이 Gram matrix를 layer l에서 구하고 (G^[l](S), G^[l](G)), 둘 사이의 차이(Frobenius norm 제곱)를 style cost로 정의한다:

```
J_style^[l](S, G) = (normalization constant) * ||G^[l](S) - G^[l](G)||^2_F
```

**여러 레이어의 style cost를 가중합하는 이유**: 하나의 레이어만 골라서 style을 정의하면 그 레이어가 포착하는 수준의 스타일만 반영된다 (얕은 레이어면 저수준 색감/edge, 깊은 레이어면 고수준 패턴). 근데 우리가 원하는 "스타일"이라는 건 사실 저수준부터 고수준까지 다양한 레벨의 패턴이 섞인 것이기 때문에, 여러 레이어(얕은 것부터 깊은 것까지)의 style cost를 각각 계산해서 가중합하면 훨씬 더 풍부하고 시각적으로 만족스러운 결과가 나온다:

```
J_style(S, G) = sum over l of  lambda^[l] * J_style^[l](S, G)
```

여기서 lambda^[l]은 각 레이어의 기여도를 조절하는 hyperparameter.

## 4. Generated Image 자체를 최적화한다는 점

Neural style transfer에서 가장 신기하면서도 헷갈렸던 부분인데, 일반적인 딥러닝 학습과 다르게 여기서는 **네트워크의 weight를 학습시키는 게 아니라 generated image G의 픽셀 값 자체가 학습 대상(파라미터)**이 된다.

과정을 정리하면:
1. G를 랜덤 노이즈 이미지로 초기화한다 (완전 랜덤 픽셀값으로 시작).
2. Pretrained CNN(VGG 등)의 weight는 고정(freeze)시켜놓는다 - 이건 그냥 feature extractor로만 쓰는 것.
3. J(G) = alpha * J_content(C,G) + beta * J_style(S,G) 를 계산한다.
4. G의 각 픽셀에 대해 gradient descent를 수행해서 G를 업데이트한다: G := G - learning_rate * dJ(G)/dG
5. 이걸 반복하면 G가 점점 C의 내용은 유지하면서 S의 스타일을 닮아가는 방향으로 픽셀 값이 조금씩 바뀌어간다.

즉 보통의 supervised learning에서는 입력(x)은 고정이고 네트워크 weight가 학습 대상인데, 여기서는 정반대로 네트워크(feature extractor)는 고정하고 "입력 이미지 자체"를 최적화 변수로 놓고 gradient descent를 돌리는 것. 이 발상 전환이 이 알고리즘의 핵심이라고 생각한다.

## 5. 확장: 1D and 3D Convolutions

이 강의 챕터 마지막에 나오는 내용으로, 지금까지 다룬 convolution은 전부 2D 이미지에 대한 것이었는데 이 개념이 다른 차원의 데이터에도 똑같이 일반화될 수 있다는 걸 짚어준다.

- **1D Convolution**: 시계열(time-series) 데이터, 예를 들면 EKG(심전도) 신호 같은 데이터에 적용할 수 있다. 2D convolution에서 filter가 (f, f) 크기로 이미지 위를 슬라이딩하며 지역적 패턴을 뽑아내듯이, 1D에서는 filter가 (f,) 크기로 시계열 축 위를 슬라이딩하면서 지역적인 패턴(파형의 특징 등)을 뽑아낸다. RNN 대신 1D conv로 시계열을 처리하는 접근도 있다고 한다.
- **3D Convolution**: 의료 영상(CT scan 같은 volumetric data - 3차원 공간 정보를 가진 데이터)이나 동영상(video, 시간 축이 추가된 데이터)에 적용된다. 필터도 (f, f, f) 형태의 3차원 큐브가 되어서, 3차원 공간(혹은 공간+시간)을 슬라이딩하며 지역적인 3D 패턴을 감지한다.

결국 convolution의 본질은 "국소적인(local) 영역에서 패턴을 감지하고, 그 필터를 전체 입력에 대해 파라미터 공유(parameter sharing)하며 슬라이딩시킨다"는 아이디어이고, 이건 입력 데이터가 몇 차원이든 상관없이(1D, 2D, 3D) 똑같이 적용 가능한 일반적인 원리라는 게 이 섹션에서 얻어가야 할 결론인 것 같다.
