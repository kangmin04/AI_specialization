# CNN 기초 (Foundations of CNNs)

Deep Learning Specialization 4번째 코스 "Convolutional Neural Networks" 첫 주차 정리. 컴퓨터 비전 문제를 딥러닝으로 풀 때 왜 fully-connected network 대신 CNN을 쓰는지부터 시작해서, convolution 연산 자체의 수학적 정의, padding/stride, 여러 채널을 다루는 방법, 마지막에 pooling까지 쭉 이어서 정리한다.

## 1. 왜 Fully-Connected Network가 아니라 CNN인가

이미지를 다루는 문제(예: 고양이 vs 아닌지 분류)를 지금까지 배운 일반 neural network로 풀려고 하면 바로 문제가 생긴다.

- 예를 들어 64x64x3 크기의 작은 이미지라고 해도 flatten하면 input feature 개수가 64*64*3 = 12,288개다. 이 정도는 그럭저럭 감당이 되는데, 만약 해상도가 좀 큰 1000x1000x3 이미지라면 flatten한 벡터 크기가 3,000,000이 된다.
- 그런데 첫 번째 hidden layer에 예를 들어 unit 1000개만 둬도, 그 layer의 weight matrix 크기는 1000 x 3,000,000 = 30억 개짜리 파라미터가 된다. 데이터도 이 정도 파라미터를 감당할 만큼 충분히 못 모으고, 메모리/연산량도 감당이 안 된다. 이게 흔히 말하는 "파라미터 폭발" 문제다.
- 근데 사실 이미지에서 어떤 걸 인식할 때는 이미지 전체 픽셀을 다 봐야 하는 게 아니라, 국소적인 패턴(local feature) - 예를 들면 눈, 코, 귀의 edge나 texture - 을 보고 그걸 조합해서 판단하는 거다. 즉 "이 근처 픽셀들끼리의 관계"만 보는 filter를 이미지 전체에 재사용(공유)하면 파라미터를 엄청 줄일 수 있다.
- 게다가 어떤 edge detector가 이미지 왼쪽 위에서 유용하면, 오른쪽 아래에서도 똑같이 유용할 가능성이 높다 (translation invariance, 위치 불변성). 고양이가 사진 왼쪽에 있든 오른쪽에 있든 "고양이 특징"은 똑같이 감지할 수 있어야 하니까, 같은 filter를 이미지 전체에 슬라이딩하면서 재사용하는 게 자연스럽다.

이 두 가지 아이디어 - "국소적인 부분만 보는 필터"와 "그 필터를 이미지 전체에 공유해서 쓰는 것" - 가 바로 convolution 연산이고, 이게 CNN의 핵심 동기다. (뒤에 11번 섹션에서 parameter sharing / sparsity of connections로 다시 정리한다.)

## 2. Edge Detection으로 보는 Convolution의 직관

Convolution이 뭘 하는 연산인지 감을 잡기 제일 좋은 예제가 edge detection이다.

이미지에서 세로 방향 edge(vertical edge)를 찾고 싶다고 하면, 아래와 같은 3x3 filter(=kernel)를 이미지 위에 슬라이딩시키면서 각 위치마다 필터와 겹치는 부분을 element-wise 곱하고 다 더하는 연산을 한다.

```
Vertical edge detector 예시:
 1  0 -1
 1  0 -1
 1  0 -1
```

이 필터를 밝은 영역 -> 어두운 영역으로 바뀌는 이미지 위에 슬라이딩하면, 밝기가 급격히 바뀌는 세로 경계선 위치에서 출력값이 크게 나오고, 평평한 영역에서는 출력이 0에 가깝게 나온다. 즉 "왼쪽은 밝고 오른쪽은 어두운" 패턴을 감지하는 필터가 되는 거다. 이 필터를 90도 돌리면(transpose) horizontal edge detector가 된다.

이런 필터는 사람이 직접 값을 정해서 쓸 수도 있는데 (고전 컴퓨터 비전에서 쓰던 게 아래 두 개다):

```
Sobel filter (vertical):        Scharr filter (vertical):
 1  0 -1                         3   0  -3
 2  0 -2                        10   0 -10
 1  0 -1                         3   0  -3
```

Sobel/Scharr는 가운데 행에 가중치를 더 줘서 edge 검출을 좀 더 안정적(robust)으로 만든 버전이다. 근데 여기서 CNN의 핵심 아이디어가 나온다 - "이 9개의 필터 값을 사람이 직접 정하지 말고, 그냥 학습 가능한 파라미터로 두고 backpropagation으로 데이터에서 학습시키면 어떨까?"라는 거다. 그러면 vertical/horizontal edge뿐 아니라 45도 각도의 edge든, 아니면 사람이 이름 붙이기 애매한 어떤 low-level feature든 네트워크가 알아서 가장 유용한 필터를 찾아낸다. 이게 바로 convolutional layer의 필터(=weight)가 하는 역할이다.

## 3. Convolution의 수학적 정의 (+ Cross-correlation과의 차이)

Convolution 연산 `*` 은, input 이미지 위에 filter(kernel)를 슬라이딩시키면서 겹치는 영역끼리 element-wise 곱을 한 뒤 전부 합산해서 하나의 출력 값을 만드는 걸 매 위치마다 반복하는 연산이다.

수식으로 쓰면 (2D, 정수 인덱스 기준):

```
(I * K)[i, j] = sum over m, n of I[i+m, j+n] * K[m, n]
```

딥러닝 프레임워크(TensorFlow, PyTorch 등)에서 "convolution"이라고 부르는 연산은 사실 신호처리 교과서에서 정의하는 "진짜 convolution"이 아니라 **cross-correlation**이다.

- 신호처리에서 진짜 convolution은 연산 전에 커널을 상하좌우로 뒤집는(flip, 180도 회전) 과정이 들어간다. `flip(K)`를 곱해서 합산하는 게 원래 정의다.
- 근데 딥러닝에서는 이 flip 과정을 생략하고 그냥 cross-correlation을 하면서 이름만 convolution이라고 부른다. 왜 상관없냐면, 어차피 필터 값 자체를 학습으로 찾아낼 거라서 - 필터가 flip 되어 있든 안 되어 있든, 그냥 학습 과정에서 알아서 그에 맞는 값으로 수렴하기 때문이다. flip을 하고 안 하고는 수학적 관례의 차이일 뿐, 학습 가능한 필터를 쓰는 CNN 입장에서는 결과적으로 상관없는 문제다.
- 그래서 이 노트에서도 관례를 따라 "convolution"이라고 부르지만, 실제로 구현되는 연산은 cross-correlation이라는 점만 기억해두면 된다.

## 4. Padding

필터 없이 그냥 convolution만 반복하면 두 가지 문제가 생긴다.

1. **Output이 계속 줄어든다 (shrinking output).** n x n 이미지에 f x f 필터를 convolution하면 결과는 (n-f+1) x (n-f+1) 크기가 된다. 예를 들어 6x6 이미지에 3x3 필터를 쓰면 4x4가 나온다. 레이어를 여러 개 쌓으면 이미지가 점점 작아지다가 결국 너무 작아져버린다.
2. **모서리/가장자리 픽셀 정보가 버려진다 (throwing away border information).** 이미지 가장자리에 있는 픽셀은 필터가 슬라이딩할 때 한두 번 정도밖에 안 겹치는데, 가운데 픽셀은 훨씬 여러 번 겹친다. 즉 가장자리 정보가 상대적으로 결과에 덜 반영되는 문제가 생긴다.

이걸 해결하려고 이미지 테두리에 픽셀을 추가하는 게 **padding**이다. 보통 값 0으로 채우는 zero-padding을 쓴다. 패딩을 p만큼 추가하면 output 크기 계산식이 바뀐다 (자세한 공식은 5번 섹션에서).

Padding 방식은 크게 두 가지로 나눠서 부른다.

- **Valid convolution**: padding을 아예 안 하는 것 (p=0). Output이 계속 줄어든다.
- **Same convolution**: output 크기가 input 크기와 똑같이 나오도록 padding을 주는 것. n x n 입력에 f x f 필터를 쓸 때, output도 n x n이 되게 하려면

```
p = (f - 1) / 2
```

만큼 패딩을 준다. 이 공식이 성립하려면 f가 홀수여야 딱 나누어떨어진다 - 이게 컨벌루션 필터 크기를 관례적으로 3x3, 5x5, 7x7처럼 홀수로 쓰는 이유이기도 하다 (홀수 필터는 "중심 픽셀"이 정확히 하나 존재해서 필터의 위치를 직관적으로 얘기하기도 편하다).

## 5. Strided Convolution과 출력 크기 공식

지금까지는 필터를 한 칸씩(step=1) 옮기면서 슬라이딩했는데, 몇 칸씩 건너뛰면서 슬라이딩할 수도 있다. 이 건너뛰는 칸 수를 **stride (s)**라고 한다. Stride를 크게 주면 필터가 이미지를 더 듬성듬성 훑고 지나가서, output 크기가 더 작아지고 연산량도 줄어든다 (downsampling 효과).

Padding p, stride s, 필터 크기 f, 입력 크기 n일 때 출력 크기 공식은 다음과 같다.

```
n_out = floor((n + 2p - f) / s) + 1
```

- floor(내림)를 쓰는 이유는, 필터가 이미지 밖으로 나가버리면 그 위치에서는 convolution을 안 하기 때문이다 (딱 나누어 떨어지지 않는 경우, 남는 부분은 그냥 무시).
- 예시: n=7, f=3, p=0, s=2 라면 `(7 + 0 - 3)/2 + 1 = 2 + 1 = 3` 이 되어 3x3 출력이 나온다.
- 참고로 padding/stride를 가로/세로 다르게 줄 수도 있지만(n_H, n_W를 따로 계산), 보통 실습에서는 정사각형 이미지에 동일한 값을 쓰는 경우가 많아서 이 노트에서도 편의상 정사각형 기준으로 정리했다.

## 6. Convolutions over Volume (다채널 입력)

실제 이미지는 흑백 2D가 아니라 RGB 3채널짜리 3D 볼륨(height x width x channel)이다. 이런 입력에 convolution을 적용하려면 필터도 입력과 채널 수가 똑같아야 한다.

- 입력이 6x6x3 (height x width x channel=3) 이면, 필터도 3x3x3처럼 마지막 채널 차원이 3이어야 한다.
- 연산은 필터의 27개(3x3x3) 값 전부를 입력의 대응 위치와 element-wise 곱해서 다 더한 뒤 "숫자 하나"를 만드는 거다. 즉 3D 필터를 슬라이딩해도 출력은 2D (channel이 사라진 형태)가 된다.
- 만약 필터를 여러 개(예를 들어 filter 1은 vertical edge detector, filter 2는 horizontal edge detector) 동시에 쓰면, 각 필터마다 하나씩 2D output이 나오고 이걸 쌓아서(stack) output의 채널 차원을 만든다.

즉 **output의 channel 개수는 그 레이어에서 사용한 필터의 개수와 같다.** 이게 "input의 채널 수는 필터의 채널 수와 같아야 하고, output의 채널 수는 필터의 개수와 같다"는 CNN의 기본 규칙이다.

## 7. One Layer of a Convolutional Network

Convolution layer 한 층이 실제로 하는 일은, 여러 개의 필터로 convolution을 한 뒤 각 필터마다 bias를 더하고 activation function(보통 ReLU)을 적용하는 것까지 포함한다. Fully-connected layer에서 `z = Wx + b`, `a = g(z)` 하던 것과 완전히 같은 패턴이고, 그냥 `Wx` 대신 convolution이 들어간 것뿐이다.

과정을 정리하면:

1. 입력 볼륨에 필터 1을 convolution -> 2D output 하나
2. 여기에 bias b1 (스칼라 하나)을 더한다
3. ReLU 같은 activation을 적용
4. 필터 2, 3, ... 도 똑같이 반복
5. 각 필터의 결과(2D)를 쌓아서 3D output 볼륨을 만든다 (output channel 수 = 필터 개수)

**파라미터 개수 계산 예시**: 입력이 6x6x3이고, 3x3x3 필터를 10개 쓴다고 하면

- 필터 1개당 파라미터 수 = 3*3*3 (weight) + 1 (bias) = 28개
- 필터가 10개니까 총 파라미터 수 = 28 * 10 = 280개

여기서 중요한 포인트는, 입력 이미지 크기가 6x6이든 600x600이든 파라미터 개수는 **여전히 280개로 동일**하다는 거다 (필터 크기와 개수만으로 결정됨). 이게 1번 섹션에서 얘기한 파라미터 폭발 문제를 CNN이 해결하는 방식이다.

## 8. CNN 표기법 정리

레이어가 깊어지면 표기가 헷갈리기 쉬워서, l번째 layer에 대한 표기를 정리해둔다.

| 기호 | 의미 |
|---|---|
| f^[l] | l번째 레이어 필터 크기 (f x f) |
| p^[l] | l번째 레이어 padding |
| s^[l] | l번째 레이어 stride |
| n_C^[l] | l번째 레이어에서 사용하는 필터 개수 (= output channel 수) |
| 필터 크기 | f^[l] x f^[l] x n_C^[l-1] (input channel 수는 이전 레이어의 output channel 수와 같아야 함) |
| Input 크기 | n_H^[l-1] x n_W^[l-1] x n_C^[l-1] |
| Output 크기 | n_H^[l] x n_W^[l] x n_C^[l] |
| Activation A^[l] 크기 | n_H^[l] x n_W^[l] x n_C^[l] (배치 처리 시 맨 앞에 batch size m 차원 추가) |
| Weight 크기 | f^[l] x f^[l] x n_C^[l-1] x n_C^[l] |
| Bias 크기 | n_C^[l] (필터 하나당 스칼라 bias 1개) |

Output의 height/width는 6번 공식을 그대로 적용:

```
n_H^[l] = floor((n_H^[l-1] + 2*p^[l] - f^[l]) / s^[l]) + 1
n_W^[l] = floor((n_W^[l-1] + 2*p^[l] - f^[l]) / s^[l]) + 1
```

## 9. Pooling Layers

Pooling은 convolution layer들 사이에 끼워 넣어서 feature map의 가로/세로 크기를 줄이는(downsampling) 역할을 하는 레이어다. 대표적으로 두 종류가 있다.

- **Max pooling**: f x f 크기 영역을 stride s만큼 슬라이딩하면서, 그 영역 안에서 제일 큰 값 하나만 뽑아서 출력으로 남긴다. 어떤 영역 안에 특정 feature(예: edge, texture)가 "존재하기만 하면" 그 강한 신호를 그대로 살리고 나머지는 버리는 방식이라, 직관적으로 "이 부분에 이 feature가 있었다"는 정보를 보존하는 데 유리하다. 실무에서 max pooling이 average pooling보다 훨씬 많이 쓰인다.
- **Average pooling**: 마찬가지로 f x f 영역을 슬라이딩하면서 이번엔 평균값을 출력으로 남긴다. 아주 깊은 네트워크의 마지막 부분에서 전체 spatial 정보를 하나로 뭉갤 때(예: 7x7x1000을 1x1x1000으로 만드는 용도) 종종 쓰인다.

Pooling에서 흔히 헷갈리는 부분들:

- **Pooling은 학습되는 파라미터가 없다.** Convolution 필터처럼 weight/bias가 있는 게 아니라, 그냥 "이 영역에서 최댓값/평균값을 계산해라"는 고정된 규칙(hyperparameter인 f, s만 있음)이기 때문이다. 그래서 backpropagation으로 업데이트할 대상 자체가 없다.
- **왜 필요한가?** representation의 크기를 줄여서 이후 레이어의 연산량과 파라미터 수를 줄여주고, 어느 정도 위치가 조금 달라져도 (약간의 이동/왜곡에 대해) 특징을 안정적으로 잡아내는 효과(작은 translation에 대한 invariance)도 있다.
- Pooling은 채널마다 독립적으로 적용된다. 즉 input이 n_H x n_W x n_C 이면, pooling 후에도 채널 수 n_C는 그대로 유지되고 height/width만 줄어든다. (channel을 섞지 않음.)
- Output 크기 공식은 convolution과 동일한 식을 쓰되, pooling에는 보통 padding을 주지 않는다 (p=0):

```
n_out = floor((n - f) / s) + 1
```

## 10. 간단한 ConvNet 예시로 보는 Layer별 Activation Shape 변화

예를 들어 32x32x3 (RGB, 32x32) 이미지를 입력으로 받아서 숫자 분류를 하는 아주 간단한 LeNet-5 스타일 ConvNet을 만든다고 하면, 대략 이런 구조가 된다.

가정: Conv1 (f=5, s=1, padding=valid, 필터 6개) -> Pool1 (f=2, s=2, max) -> Conv2 (f=5, s=1, padding=valid, 필터 16개) -> Pool2 (f=2, s=2, max) -> Flatten -> FC1 (120) -> FC2 (84) -> Softmax(10)

| Layer | 연산 | Output shape (H x W x C) | Activation size | 파라미터 수 |
|---|---|---|---|---|
| Input | - | 32x32x3 | 3,072 | 0 |
| Conv1 | f=5,s=1,valid, 필터 6개 | 28x28x6 | 4,704 | (5*5*3+1)*6 = 456 |
| Pool1 | max, f=2,s=2 | 14x14x6 | 1,176 | 0 |
| Conv2 | f=5,s=1,valid, 필터 16개 | 10x10x16 | 1,600 | (5*5*6+1)*16 = 2,416 |
| Pool2 | max, f=2,s=2 | 5x5x16 | 400 | 0 |
| Flatten | - | 400 (1D) | 400 | 0 |
| FC1 | Dense | 120 | 120 | 400*120+120 = 48,120 |
| FC2 | Dense | 84 | 84 | 120*84+84 = 10,164 |
| Softmax | Dense | 10 | 10 | 84*10+10 = 850 |

여기서 얻을 수 있는 인사이트:

- Conv/Pool 레이어를 지나면서 height, width는 계속 줄어들고(32 -> 28 -> 14 -> 10 -> 5), channel 수는 오히려 늘어난다(3 -> 6 -> 16). "이미지가 옆으로 넓게 퍼진 정보를 점점 깊이(채널) 방향으로 압축해나간다"는 CNN의 전형적인 패턴이다.
- Activation size(=n_H * n_W * n_C)는 레이어를 지날수록 대체로 줄어드는 추세다. 너무 급격하게 줄어들면 정보 손실이 심할 수 있어서, 보통 서서히 줄여나가는 걸 권장한다.
- 파라미터 수를 보면 Pooling 레이어는 항상 0이고 (9번 섹션에서 얘기한 대로), 오히려 맨 뒤의 FC layer들이 conv layer보다 훨씬 많은 파라미터를 차지하는 경우가 흔하다 (FC1의 48,120개 vs Conv 레이어들 합쳐도 몇 천 개 수준). Conv layer는 필터를 재사용하기 때문에 파라미터 대비 표현력이 훨씬 효율적이라는 걸 숫자로도 확인할 수 있다.

## 11. CNN이 잘 동작하는 이유: Parameter Sharing과 Sparsity of Connections

1번 섹션에서 던졌던 질문 - "왜 CNN이 이미지에 대해 fully-connected보다 훨씬 적은 파라미터로도 잘 동작하는가" - 에 대한 답은 크게 두 가지다.

1. **Parameter sharing (파라미터 공유)**: 하나의 필터(예: vertical edge detector)는 이미지의 특정 한 위치에서만 유용한 게 아니라, 이미지 전체 어디서든 똑같이 유용할 가능성이 높다. 그래서 같은 필터 값(weight)을 이미지의 모든 위치에 대해 재사용(공유)한다. 그 결과 학습해야 할 파라미터 수가 이미지 크기와 무관하게 필터 크기 * 필터 개수 수준으로 고정된다 (7번 섹션 예시에서 6x6 이미지든 600x600 이미지든 파라미터가 280개로 동일했던 이유).
2. **Sparsity of connections (연결의 희소성)**: Fully-connected layer에서는 output의 각 원소가 input의 모든 원소와 연결되어 있는데, convolution에서는 output의 각 원소가 input의 아주 일부(필터 크기만큼, 예를 들면 3x3=9개)에만 의존한다. 즉 대부분의 input-output 쌍 사이에는 연결 자체가 없다(=0으로 취급). 이렇게 연결을 국소적으로만 만드는 게, 애초에 이미지에서 의미 있는 패턴들이 대체로 "국소적"이라는 사전 지식(prior)을 모델 구조에 반영한 거라고 볼 수 있다.

이 두 가지 특성 덕분에 CNN은:

- 파라미터 수가 훨씬 적어서 overfitting 위험이 줄고, 상대적으로 적은 학습 데이터로도 학습이 가능하다.
- Translation invariance(이미지 안에서 물체 위치가 조금 바뀌어도 같은 필터가 똑같이 반응)를 자연스럽게 얻을 수 있다.

결국 CNN은 "이미지의 특징은 국소적(local)이고, 위치에 무관하게 재사용 가능하다"는 이미지 데이터 고유의 성질을 네트워크 구조 자체에 새겨 넣은 아키텍처라고 정리할 수 있다.
