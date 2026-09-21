# 대표적인 CNN 아키텍처들

CNN 기본 개념(convolution, pooling 등)을 배웠으니, 이제 실제로 사람들이 "이렇게 쌓았더니 잘 되더라"하고 검증한 유명한 아키텍처들을 시간순으로 훑어보려고 한다. 신기한 건 이 아키텍처들 하나하나가 그 시대의 한계를 넘어서려는 시도였다는 점이다. 그냥 구조를 외우기보다 "왜 이런 구조가 필요했는지"를 따라가면서 정리해본다.

## Classic Networks

### LeNet-5 (1998, Yann LeCun)

- 논문: LeCun et al., "Gradient-Based Learning Applied to Document Recognition" (1998)
- 목적: 손글씨 숫자(우편번호, 수표 등) 인식. 지금 보면 아주 작은 문제처럼 보이지만 당시엔 이걸 신경망으로 푼다는 것 자체가 도전이었다.
- 구조: `INPUT(32x32x1) -> CONV -> AVG POOL -> CONV -> AVG POOL -> FC -> FC -> OUTPUT`
  - 지금 기준으로 엄청 작다. 파라미터 수 약 6만개 수준.
  - 당시엔 ReLU가 없어서 activation으로 sigmoid/tanh를 썼고, pooling도 max pooling이 아니라 average pooling을 썼다.
  - 채널이 깊어질수록(레이어를 통과할수록) height/width는 줄어들고 채널 수는 늘어나는 패턴 - 이게 이후 모든 CNN의 기본 패턴이 된다. LeNet이 이 패턴을 처음 보여준 셈.
- 의의: "이미지를 그대로 넣고, conv+pooling을 반복하고, 마지막에 FC로 분류한다"는 CNN의 기본 골격을 최초로 성공시킨 사례. 이후 나오는 모든 아키텍처가 이 뼈대 위에서 발전한다.

### AlexNet (2012, Krizhevsky, Sutskever, Hinton)

- 논문: "ImageNet Classification with Deep Convolutional Neural Networks" (NeurIPS 2012)
- 배경: ImageNet 대회(ILSVRC)에서 압도적인 차이로 우승하면서 딥러닝이 컴퓨터 비전을 평정하기 시작한 계기가 된 모델. 이 전까지는 hand-crafted feature(SIFT, HOG 등) + 전통 ML(SVM 등) 조합이 주류였는데, AlexNet 이후로 "그냥 CNN에 데이터 때려넣고 학습시키는 게 이긴다"는 인식이 퍼졌다.
- 구조: LeNet과 기본 골격은 비슷한데(conv-pool 반복 후 FC) 훨씬 크다. 파라미터 수 약 6천만개 수준 - LeNet 대비 1000배.
- LeNet 대비 핵심 변화:
  - **ReLU 사용**: sigmoid/tanh는 입력이 크거나 작을 때 gradient가 거의 0이 되는(saturate) 문제가 있는데, ReLU는 양수 구간에서 기울기가 항상 1이라 학습이 훨씬 빨라진다. 이게 딥러닝이 "딥"해질 수 있었던 이유 중 하나.
  - **여러 GPU에 나눠서 학습**: 당시 GPU 메모리가 부족해서 네트워크를 두 개의 GPU에 나눠 학습시켰다. (지금은 하드웨어가 좋아져서 크게 신경쓸 부분은 아니지만, 당시엔 이런 엔지니어링적 트릭이 중요했다)
  - **Local Response Normalization(LRN)** 사용 - 근데 이건 이후 연구에서 큰 효과가 없다고 밝혀져서 요즘 아키텍처엔 거의 안 쓰인다.
- 의의: "딥러닝 붐의 시작점"이라고 불리는 이유는, 이 논문이 나온 뒤로 컴퓨터 비전 학계 전체가 CNN 기반으로 넘어갔기 때문이다. 논문 자체도 비교적 읽기 쉬운 편이라 처음 딥러닝 논문 읽어볼 때 추천되는 논문.

### VGG-16 (2014, Simonyan & Zisserman)

- 논문: "Very Deep Convolutional Networks for Large-Scale Image Recognition"
- 핵심 아이디어: AlexNet은 필터 크기(11x11, 5x5, 3x3 등)와 stride가 레이어마다 다 제각각이라 좀 복잡한데, VGG는 "그냥 3x3 conv(stride 1, same padding)와 2x2 max pool(stride 2)만 계속 반복하자"는 매우 단순하고 일관된 규칙으로 설계했다.
- 구조 패턴: `CONV3x3 반복 -> POOL` 블록을 쌓아가면서 필터 개수를 64 -> 128 -> 256 -> 512로 두 배씩 늘려간다. "16"은 weight를 가진 레이어(conv+FC)가 16개라는 뜻(VGG-19도 있음, 19개 레이어).
- 왜 3x3 conv를 반복하는 방식이 괜찮은가: 3x3 conv 두 번을 쌓으면 receptive field가 5x5 conv 한 번과 비슷해지는데, 파라미터 수는 더 적고 non-linearity(ReLU)를 한 번 더 통과하니 표현력은 오히려 좋아진다. → 작은 필터를 깊게 쌓는 게 큰 필터를 얕게 쌓는 것보다 효율적이라는 통찰.
- 깊이가 주는 효과: 레이어가 깊어질수록(=receptive field가 커질수록) 더 추상적이고 복잡한 패턴(가장자리 -> 질감 -> 부분 -> 전체 객체)을 잡아낼 수 있어서 성능이 좋아진다.
- 단점: 구조는 단순하지만 파라미터 수가 어마어마하다 (약 1억 3천8백만개, 대부분 FC 레이어에 몰려있다). 그만큼 메모리/연산량 부담이 크다는 게 VGG의 가장 큰 단점으로 지적된다.

### Classic Networks 비교표

| 모델 | 연도 | 핵심 특징 | 파라미터 수(대략) | 깊이(weight layer) |
|---|---|---|---|---|
| LeNet-5 | 1998 | sigmoid/tanh, avg pooling, 최초의 CNN 골격 | ~6만 | 5~7 |
| AlexNet | 2012 | ReLU 도입, GPU 병렬학습, ImageNet 우승 | ~6천만 | 8 |
| VGG-16 | 2014 | 3x3 conv만 반복하는 단순/일관된 구조 | ~1억 3천8백만 | 16 |

## ResNet (Residual Networks, 2015)

- 논문: He et al., "Deep Residual Learning for Image Recognition" (CVPR 2015)
- 배경 문제의식: VGG 이후 "네트워크를 더 깊게 쌓으면 더 잘 되지 않을까?"라는 자연스러운 기대가 있었는데, 실제로 plain network(그냥 conv 레이어를 계속 이어붙인 네트워크)를 너무 깊게 쌓으면 오히려 **training error가 늘어나는** 현상이 관찰됐다. 이게 이론적으로 이상한 부분인데, overfitting 때문이 아니라 아예 training set에 대한 에러 자체가 나빠지는 것이었다. 이를 **degradation problem**이라고 부른다.
  - 왜 이런 일이 생기냐면, 네트워크가 깊어질수록 forward/backward propagation 과정에서 gradient가 점점 작아지거나(vanishing) 커지는(exploding) 문제가 심해지기 때문이다. 특히 vanishing gradient가 심해지면 앞쪽 레이어들의 weight가 거의 업데이트되지 않아서 최적화 자체가 잘 안 되는 상황이 생긴다.
  - 직관적으로 생각하면 "깊은 네트워크는 최소한 얕은 네트워크만큼은 해야 한다"(뒤쪽 레이어들이 identity function만 학습해도 얕은 네트워크와 같아지니까)는 게 상식인데, 실제로는 최적화가 잘 안 돼서 그 identity function조차 제대로 학습을 못 하는 게 문제였다.
- 핵심 아이디어 - Residual block과 skip connection:
  - 일반적인 레이어는 `a[l+2] = g(z[l+2])`처럼 이전 레이어의 출력을 다음 레이어에 순차적으로 통과시키는데, residual block은 **skip connection(shortcut)**을 추가해서 `a[l]`을 몇 개 레이어를 건너뛰어(예: 2개 레이어 건너뜀) 뒤쪽에 그대로 더해준다.
  - 수식으로 보면 `a[l+2] = g(z[l+2] + a[l])` - 즉 레이어가 학습해야 하는 건 원래 목표 함수 `H(x)` 자체가 아니라 **잔차(residual)** `F(x) = H(x) - x`가 된다.
  - 이렇게 하면 왜 학습이 쉬워지냐면, 만약 그 블록이 별로 도움이 안 되는 경우라도 weight를 0에 가깝게만 학습하면 `a[l+2] ≈ g(a[l]) = a[l]`(ReLU 쓰면 identity)이 되어 최소한 성능이 나빠지지는 않는다. 즉 **identity mapping을 학습하는 게 매우 쉬워진다** - "아무것도 안 하는 것"을 기본값으로 깔아주는 구조라서, 추가된 레이어가 최소한 손해는 안 보고 잘하면 이득을 보는 구조가 되는 것이다.
  - 이게 gradient 관점에서도 도움이 되는데, skip connection이 gradient가 곧바로 앞쪽 레이어로 흘러갈 수 있는 "지름길"을 만들어줘서 vanishing gradient 문제를 완화시켜준다.
- ResNet의 구조: 이런 residual block을 수십~수백 개 쌓는다 (ResNet-50, ResNet-101, ResNet-152 등 숫자는 레이어 개수). 참고로 ResNet에서는 각 conv 뒤에 Batch Normalization을 붙이는 게 표준.
- Plain network vs ResNet 성능 비교:
  - Plain network는 레이어를 계속 늘리면 어느 순간부터 training error가 다시 증가하는 U자 형태(혹은 그냥 계속 나빠지는 형태)를 보인다.
  - ResNet은 레이어를 100개, 심지어 1000개 넘게 쌓아도 training error가 꾸준히 감소한다 - 이론이 실제로 증명된 사례. 덕분에 "깊이가 곧 성능"이라는 걸 실제로 실현 가능하게 만든 아키텍처.

## 1x1 Convolution (Network in Network)

- 처음 봤을 땐 "1x1 conv가 뭔 의미가 있지? 그냥 픽셀 하나에 숫자 하나 곱하는 거 아닌가" 싶었는데, 채널이 여러 개일 때를 생각하면 이야기가 달라진다.
- 입력이 `(H, W, C)`일 때 1x1 conv 필터 하나는 사실 그 위치의 C개 채널 값들을 다 모아서 **가중합(weighted sum)**을 구하는 것과 같다. 즉 공간적으로는 아무것도 안 하지만, **채널 방향으로는 FC layer 하나를 통과시키는 것**과 동일한 효과를 낸다.
- 이게 왜 유용한가:
  1. **채널 수 조절**: 필터 개수를 조절하면 채널 수를 늘리거나 줄일 수 있다. 특히 채널 수를 줄이는 용도로 쓰면 이후 3x3, 5x5 같은 비싼 conv 연산의 입력 채널 수를 줄여서 **연산량을 확 줄이는 bottleneck 구조**를 만들 수 있다. (반대로 pooling은 height/width는 줄일 수 있지만 채널 수는 못 줄인다는 한계가 있는데, 1x1 conv가 그 빈자리를 채워준다)
  2. **non-linearity 추가**: 1x1 conv 뒤에도 ReLU 같은 activation을 붙이니까, 네트워크에 non-linearity를 하나 더 추가하는 효과가 있다. 그러니까 단순히 채널을 줄이는 것 뿐 아니라 그 자체로 하나의 표현력 있는 레이어 역할도 한다.
- 이 아이디어는 "Network in Network" 논문(Lin et al., 2013)에서 처음 나왔고, 이후 Inception, ResNet의 bottleneck block, MobileNet 등 거의 모든 현대 아키텍처에서 필수 요소로 쓰인다.

## Inception Network (GoogLeNet, 2014)

- 논문: Szegedy et al., "Going Deeper with Convolutions" (CVPR 2015, 2014 ImageNet 대회 우승)
- 배경 문제의식: CNN 설계할 때 "이 레이어에 3x3을 쓸지 5x5를 쓸지 pooling을 쓸지" 매번 고민하고 하나를 선택해야 하는데, Inception의 아이디어는 아예 **"고민하지 말고 다 해보고 결과를 이어붙이자(concatenate)"**는 것이다.
- Inception module 구조: 같은 입력에 대해 병렬로
  - 1x1 conv
  - 3x3 conv
  - 5x5 conv
  - 3x3 max pooling (+ 1x1 conv로 채널 맞추기)
  
  를 각각 적용한 뒤, 출력들을 채널 방향으로 concat해서 다음 레이어로 넘긴다. 각 필터가 어떤 크기의 패턴을 더 잘 잡아낼지 네트워크가 학습 과정에서 알아서 조합하도록 맡기는 셈.
- Computational cost 문제: 그냥 이렇게 다 이어붙이면 특히 5x5 conv 쪽에서 연산량이 폭발적으로 늘어난다. (예를 들어 입력 채널이 크면 5x5 conv 하나의 연산량이 억 단위로 커질 수 있다)
  - 해결책이 바로 위에서 다룬 **1x1 conv를 이용한 bottleneck**: 5x5(또는 3x3) conv를 적용하기 전에 1x1 conv로 채널 수를 확 줄여놓고(bottleneck layer), 그다음에 원래 conv를 적용한다. 예를 들어 `192채널 -> 1x1 conv로 16채널 -> 5x5 conv로 32채널` 순서로 하면, 중간에 채널을 줄였다가 다시 늘리는 구조라 전체 연산량이 10배 가까이 줄어들면서도 성능 손실은 거의 없다는 게 실험적으로 확인됐다.
  - 이 bottleneck 구조 덕분에 "다양한 필터를 다 써보자"는 아이디어를 연산량 부담 없이 실현할 수 있게 된 것.
- GoogLeNet은 이런 Inception module을 여러 개 쌓아서 만든 네트워크. 참고로 중간중간에 auxiliary classifier(보조 출력)를 달아서 중간 레이어에서도 gradient가 잘 전달되도록 하는 트릭도 쓰였다 (지금 기준으로는 덜 중요하게 다뤄지는 디테일).

## 실용적인 조언들

### Transfer Learning

- 컴퓨터 비전에서는 처음부터(random initialization) 네트워크를 학습시키는 경우보다, ImageNet 같은 대규모 데이터셋으로 **미리 학습된(pretrained) 가중치**를 가져다 쓰는 경우가 훨씬 많다. 이유는 간단한데, 이미지의 저수준 특징(가장자리, 곡선, 색 패턴 등)은 어떤 task든 거의 공통적으로 필요하기 때문에, 남이 이미 큰 데이터셋+오랜 시간으로 학습해놓은 걸 재활용하는 게 훨씬 효율적이다. 나처럼 GPU도 부족하고 데이터도 적은 상황에서는 거의 필수적인 전략인 것 같다.
- Freeze할 레이어를 정하는 기준은 **내가 가진 데이터 양**에 달려있다:
  - **데이터가 매우 적을 때**: 앞쪽의 거의 모든 conv 레이어를 freeze(가중치 고정, 학습 안 함)하고, 마지막 FC(softmax) 레이어만 내 task에 맞게 새로 학습한다. 이 경우 마지막 레이어 직전까지의 출력을 미리 계산해서 디스크에 저장해두면(feature caching), 매 epoch마다 앞쪽 레이어를 다시 계산할 필요가 없어서 학습이 훨씬 빨라진다.
  - **데이터가 어느 정도 있을 때**: 앞쪽 일부 레이어는 freeze하고, 뒤쪽 몇 개 레이어 + 새 출력층을 학습시킨다(fine-tuning). 데이터가 많아질수록 freeze하는 레이어 수를 줄여간다.
  - **데이터가 아주 많을 때**: pretrained weight는 random initialization 대신 "좋은 초기값"으로만 쓰고, 네트워크 전체를 fine-tuning한다.
  - 규칙을 요약하면: **데이터가 적을수록 더 많이 freeze, 데이터가 많을수록 더 많이 학습**.

### Data Augmentation

- 컴퓨터 비전은 다른 분야보다 "데이터가 항상 부족하다"고 느껴지는 분야라서(이미지 하나하나가 정보량이 크고 라벨링 비용도 크니까) data augmentation이 거의 표준적으로 쓰인다.
- 대표적인 기법들:
  - **Mirroring(좌우 반전)**: 대부분의 이미지 인식 task에서 좌우 반전해도 라벨이 바뀌지 않으니 (고양이를 좌우로 뒤집어도 여전히 고양이) 가장 안전하고 흔하게 쓰는 기법.
  - **Random Cropping**: 이미지의 일부를 무작위로 잘라내서 사용. 완벽하진 않지만(가끔 애매하게 잘릴 수 있음) 실용적으로 잘 작동한다. 객체가 이미지 어디에 있든, 크기가 어떻든 잘 인식하게 만드는 효과.
  - **Color Shifting**: R, G, B 채널 값에 약간씩 랜덤한 값을 더하거나 빼서 색감을 변형시킨다. 조명 조건이 다른 환경에서도 강건하게 만들어주는 목적. (참고로 AlexNet 논문에서는 PCA 기반으로 색을 왜곡시키는 "PCA color augmentation" 기법도 소개됨)
  - 그 외에도 rotation, shearing, local warping 등 다양하게 있지만 위 세 가지가 가장 기본적으로 많이 쓰인다.
- 목적을 한 문장으로 정리하면: **모델이 학습 데이터의 사소한 변형(위치, 각도, 조명, 크기)에 흔들리지 않고 진짜 중요한 특징만 학습하도록 만들어서 일반화 성능을 높이는 것**.
- 실무에서는 큰 학습 데이터셋을 디스크/네트워크에서 읽어오는 스레드와, augmentation을 실시간으로 적용하는 스레드, 그리고 실제 학습(forward/backward)을 하는 프로세스를 병렬로 돌리는 구조가 흔하다 (CPU에서 augmentation하고 GPU에서 학습하는 식으로).

### MobileNet

- 배경: 지금까지 본 네트워크들(VGG, ResNet, Inception 등)은 성능은 좋지만 연산량이 매우 커서 **모바일이나 임베디드 기기처럼 연산 자원이 제한된 환경**에서 돌리기엔 부담스럽다. MobileNet은 이런 저자원 환경에서도 쓸 수 있는 가벼운 CNN을 만들자는 목표에서 나왔다.
- 핵심 아이디어 - **Depthwise Separable Convolution**: 일반적인 conv 연산을 두 단계로 쪼갠다.
  1. **Depthwise Convolution**: 입력의 각 채널마다 독립적으로 하나의 필터를 적용한다(채널을 섞지 않음). 즉 `n_c`개 채널이면 `n_c`개의 2D 필터가 각자 자기 채널에만 conv를 수행.
  2. **Pointwise Convolution**: 그 결과에 1x1 conv를 적용해서 채널들을 섞고(조합하고) 원하는 출력 채널 수로 맞춘다.
  
  일반 conv가 "공간 방향 필터링 + 채널 방향 조합"을 한 번에 처리하는 것과 달리, 이 둘을 분리해서 순차적으로 처리하는 것.
- 연산량 절감 효과: 일반 conv의 연산 비용을 대략 `(출력 크기) x (필터 크기^2) x (입력 채널) x (출력 채널)`이라 하면, depthwise separable conv는 이걸 depthwise 단계와 pointwise 단계로 나눠서 계산하기 때문에 전체 연산량이 대략 `1/출력채널 + 1/필터크기^2` 비율로 줄어든다. 필터가 3x3이라고 하면 대략 **8~9배 정도 연산량이 줄어드는 효과**가 있다 (논문 기준). 성능 손실은 크지 않으면서 연산량은 크게 줄어드니, 모바일/엣지 환경처럼 배터리와 연산 자원이 제한된 곳에서 실시간으로 돌려야 하는 경우에 특히 중요하다.
- **MobileNet v1**: 위 depthwise separable conv 블록을 기본 단위로 쌓은 구조.
- **MobileNet v2**: v1 대비 두 가지가 추가됐다.
  - **Bottleneck block (Inverted Residual)**: 일반적인 residual block은 채널을 줄였다가(bottleneck) 다시 늘리는 구조인데, MobileNet v2는 반대로 **채널을 늘렸다가(expansion) 줄이는** 구조를 쓴다. 순서는 `1x1 conv로 채널 확장 -> depthwise conv -> 1x1 conv로 채널 축소` + skip connection. 채널을 확장한 상태에서 depthwise conv를 하면 표현력이 더 풍부해지면서도, 최종적으로는 얇은 채널로 압축해서 저장/전달하니 메모리 효율이 좋다는 아이디어.
  - Skip connection도 ResNet처럼 추가해서 gradient 흐름과 학습 안정성을 개선했다.

### EfficientNet

- 배경 문제의식: 네트워크 성능을 올리고 싶을 때 보통 셋 중 하나를 키운다 - **depth(레이어 수를 늘림)**, **width(채널/필터 수를 늘림)**, **resolution(입력 이미지 해상도를 키움)**. 근데 기존 연구들은 대부분 이 셋 중 하나만 조절하거나 감으로 대충 같이 조절했다.
- 핵심 아이디어 - **Compound Scaling**: depth, width, resolution 셋을 **동시에, 정해진 비율로** 함께 스케일링하는 게 하나만 키우는 것보다 훨씬 효율적이라는 걸 보이고, 이 셋의 최적 조합 비율을 찾는 방법을 제시했다. 대략 `depth ~ alpha^phi`, `width ~ beta^phi`, `resolution ~ gamma^phi` 같은 식으로 하나의 계수 `phi`로 셋을 함께 조절하는 방식.
- 결과적으로 EfficientNet-B0부터 B7까지, 사용 가능한 연산 자원(모바일부터 클라우드까지)에 맞춰 셋을 비례해서 키운 여러 버전의 모델을 만들 수 있게 됐다. "내 디바이스 연산력이 이 정도면 어떤 조합의 네트워크가 최적일까"라는 질문에 체계적인 답을 준 셈.
- 지금 당장 구조를 세세히 구현할 일은 없을 것 같지만, "네트워크 크기를 키울 때 depth/width/resolution을 어떤 비율로 같이 키워야 효율적인가"라는 관점 자체가 이후 아키텍처 설계에 큰 영향을 준 아이디어라 개념만 잘 잡아두면 될 듯하다.

## 아키텍처 비교 정리

| 모델 | 연도 | 핵심 아이디어 | 주로 노리는 것 |
|---|---|---|---|
| LeNet-5 | 1998 | conv-pool 반복 + FC라는 기본 골격 | 최초의 동작하는 CNN |
| AlexNet | 2012 | ReLU, 대규모 데이터 + 깊은 네트워크 | 정확도(딥러닝 붐 시작) |
| VGG-16 | 2014 | 3x3 conv만 일관되게 반복 | 단순함/일관성, 깊이 |
| GoogLeNet(Inception) | 2014 | 여러 필터 크기를 병렬로 다 쓰고 concat, 1x1 bottleneck | 정확도 + 연산 효율 |
| ResNet | 2015 | skip connection으로 identity mapping을 쉽게 학습 | 매우 깊은 네트워크의 학습 가능성 |
| MobileNet (v1/v2) | 2017/2018 | depthwise separable conv, inverted residual | 모바일/엣지 환경의 경량화 |
| EfficientNet | 2019 | depth/width/resolution compound scaling | 자원 대비 최적 성능 |

## Computer Vision에서 데이터 양에 따른 실무 전략 (State of Computer Vision)

강의에서 재밌었던 부분인데, 머신러닝 문제를 "얼마나 많은 데이터가 있는가"라는 축으로 놓고 보면 두 축의 스펙트럼이 있다는 얘기였다.

- **데이터가 아주 많을 때** (예: 일부 음성 인식, 일부 이미지 분류처럼 데이터가 풍부한 경우): 굳이 사람이 이것저것 손으로 설계(hand-engineering)할 필요 없이, 비교적 단순한 알고리즘 + 큰 네트워크에 데이터를 많이 넣어주는 것만으로 좋은 성능이 나온다. "데이터가 문제를 해결해준다"는 쪽에 가깝다.
- **데이터가 적을 때** (컴퓨터 비전의 상당수 문제, 특히 object detection처럼 라벨링 비용이 큰 task): 데이터가 부족한 걸 메꾸기 위해 사람이 더 많이 개입해야 한다.
  - **Hand-engineering**: 네트워크 구조를 더 정교하게 설계하거나(예: 위에서 본 여러 아키텍처들의 디테일한 트릭들), 손으로 만든 feature를 쓰거나, 하이퍼파라미터를 세밀하게 튜닝하는 등 사람의 지식/노력을 더 많이 투입한다.
  - **Transfer learning**: 위에서 다룬 것처럼 다른 대규모 데이터셋에서 학습한 가중치를 가져다 쓴다.
  - **Data augmentation**을 적극적으로 활용해서 데이터 양을 사실상 늘린다.
- 즉 컴퓨터 비전 분야는 예나 지금이나 "데이터냐 hand-engineering이냐"의 트레이드오프가 계속 존재하는데, 최근 추세는 데이터셋이 커질수록 hand-engineering 비중이 줄고 있지만, 여전히 완전히 없앨 순 없는 영역이라는 인상이다.
- 벤치마크/대회에서 잘 통하지만 실무 프로덕션에서는 잘 안 쓰는 기법들도 언급됐는데:
  - **Ensembling**: 여러 개의 네트워크를 독립적으로 학습시킨 뒤 출력을 평균내는 방법. 성능은 1~2%p 정도 올라갈 수 있지만, 추론 시 여러 모델을 다 돌려야 해서 실제 서비스에는 부담.
  - **Multi-crop at test time**: 테스트 이미지를 여러 방식으로 crop해서 각각 예측한 뒤 평균내는 방법(10-crop 등). 역시 정확도는 오르지만 추론 속도가 느려져서 실무에서는 잘 안 쓴다.
  - 결론적으로 이런 기법들은 "논문/대회 성능 극대화용"이지 실제 서비스 배포용은 아니라는 걸 구분해서 받아들이면 될 것 같다.
- 마지막 팁으로 강의에서 강조한 것: 오픈소스로 공개된 코드/pretrained model을 최대한 활용하라는 것. 위에서 다룬 아키텍처들 대부분 공식 구현체나 pretrained weight가 공개되어 있으니, 처음부터 새로 구현하기보다는 이런 걸 가져다 쓰는 게 훨씬 실용적이다.
