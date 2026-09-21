# Computer Vision 공부 전에 알아야 할 Deep Learning 선수 개념

이 노트는 `01~04` CV 노트를 읽기 전에 필요한 배경 지식 중, **Machine Learning Specialization에는 없고 Deep Learning Specialization(Course 1~3: Neural Networks and Deep Learning / Improving Deep Neural Networks / Structuring ML Projects)에서 새로 나오는 개념**만 추려서 정리한 것이다.

ML Specialization에서 이미 다룬 내용 - linear/logistic regression, 기본적인 neural network 구조와 TensorFlow 사용법, activation function(ReLU/softmax) 개념, regularization(L2)의 기본 아이디어, k-means 같은 것들은 여기서 다시 설명하지 않는다. 여기 나오는 건 전부 "이게 있어야 CV 노트에 나온 표현/개념(예: ResNet 문서의 vanishing gradient, Inception 구조에 흔히 들어가는 batch norm 등)을 제대로 이해할 수 있다"는 기준으로 고른 것들이다.

## 1. Deep Neural Network의 벡터화 표기법 (Course 1)

CV 노트에서 CNN의 레이어를 `l`로 표기하고 `f[l], p[l], s[l], n_C[l]` 같은 식으로 쓰는 표기법 자체가, 사실 일반 deep neural network에서 쓰던 표기법을 그대로 가져온 거다. 이 표기법에 익숙하지 않으면 CV 노트가 갑자기 어려워 보일 수 있다.

- `L`: 전체 레이어 수, `l`: 몇 번째 레이어인지 (l = 1 ... L)
- 레이어 `l`에서: `Z[l] = W[l] A[l-1] + b[l]`, `A[l] = g[l](Z[l])` (`g[l]`은 그 레이어의 activation function)
- `A[0]`은 그냥 input `X`
- 이걸 m개의 학습 샘플에 대해 한 번에 계산하려고 column 방향으로 쌓아서 행렬곱 하나로 처리하는 게 vectorization. for문으로 샘플 하나하나 도는 것보다 훨씬 빠르다.
- Forward propagation은 l=1부터 L까지 순서대로 위 식을 계산해나가는 것, backpropagation은 반대로 L부터 1까지 chain rule로 `dZ[l], dW[l], db[l]`을 구해나가는 것.
- **Parameter(W, b) vs Hyperparameter(learning rate, layer 수, hidden unit 수, activation 종류 등)** 구분: hyperparameter는 학습으로 정해지는 게 아니라 사람이 골라야 하는 값이라는 점. CV 아키텍처 문서에서 "filter 크기를 몇으로 할지, stride를 몇으로 할지"를 사람이 실험적으로 정하는 것도 결국 hyperparameter tuning의 연장선이다.

## 2. Weight Initialization과 Vanishing/Exploding Gradient (Course 1~2)

`02-classic-and-modern-architectures.md`에서 ResNet을 설명할 때 "네트워크를 깊게 쌓으면 오히려 training error가 나빠지는 degradation problem"과 vanishing/exploding gradient를 언급했는데, 그 배경이 되는 개념이다.

- **왜 weight를 전부 0으로 초기화하면 안 되는가**: 모든 뉴런이 완전히 똑같은 계산을 하게 돼서(symmetry), 아무리 학습을 시켜도 뉴런들이 서로 다른 feature를 학습하지 못한다. 그래서 weight는 랜덤한 작은 값으로 초기화해야 한다.
- **Vanishing/Exploding gradient**: 네트워크가 깊어질수록, 각 레이어를 통과할 때마다 activation 값(또는 gradient)이 계속 곱해지면서 레이어 수에 지수적으로 비례해 값이 아주 작아지거나(vanishing) 아주 커지는(exploding) 현상이 생긴다. 레이어가 10개, 20개만 넘어가도 심각해질 수 있다.
- **He initialization / Xavier initialization**: weight 초기값의 분산을 이전 레이어의 unit 개수에 반비례하게 스케일링해서(예: ReLU를 쓸 땐 분산을 `2/n[l-1]`로) 이 문제를 어느 정도 완화하는 초기화 방법. "그냥 랜덤"이 아니라 "레이어 크기에 맞춰 스케일링된 랜덤"이라는 게 포인트.
- 이 개념이 있어야 "ResNet이 skip connection으로 gradient가 더 잘 흐르게 해서 이 문제를 우회한다"는 CV 노트의 설명이 왜 중요한 발상인지 와닿는다.

## 3. Regularization - Dropout (Course 2)

ML Specialization에서 배운 L2 regularization 외에, CNN 계열 아키텍처 논문(AlexNet 등)에서 흔히 같이 언급되는 기법이다.

- **Dropout**: 학습할 때마다 각 레이어의 뉴런을 랜덤한 확률(keep_prob)로 꺼버리고(0으로 만들고) 학습하는 방법. 매 iteration마다 다른 "축소된 네트워크"를 학습시키는 셈이라, 특정 뉴런 하나에 과하게 의존하지 못하게 만들어서 overfitting을 줄인다.
- **Inverted dropout**: 꺼지지 않은 뉴런의 출력값을 `keep_prob`으로 나눠서 activation의 기댓값 스케일을 맞춰주는 실제 구현 방식. 이렇게 해야 test time에는 dropout 없이 그냥 forward pass만 하면 된다.
- **Early stopping**: train/dev error를 iteration마다 그려서 dev error가 다시 올라가기 시작하는 지점에서 학습을 멈추는 방법도 regularization의 일종으로 다룬다.

## 4. Mini-batch와 최적화 알고리즘 (Course 2)

ML Specialization의 gradient descent는 보통 전체 데이터를 한 번에 쓰는 batch gradient descent 위주였는데, 딥러닝에서는 데이터가 크기 때문에 다른 접근이 필요하다.

- **Mini-batch gradient descent**: 전체 데이터를 작은 batch(예: 64, 128, 256개)로 쪼개서, 한 mini-batch씩 볼 때마다 파라미터를 업데이트한다. Batch gradient descent보다 훨씬 빠르게 iteration을 돌 수 있고, Stochastic gradient descent(batch size=1)보다는 노이즈가 적어서 안정적이다.
- **Exponentially Weighted Average**: 최근 값에 더 큰 가중치를 주는 이동평균 방식. `v_t = beta * v_{t-1} + (1-beta) * theta_t`. 아래 최적화 알고리즘들의 기반이 되는 아이디어다.
- **Momentum**: gradient의 지수가중평균을 이용해서 업데이트 방향을 부드럽게 만든다. 골짜기에서 진동하지 않고 목표 방향으로 더 곧장 나아가게 해준다.
- **RMSprop**: gradient 제곱의 지수가중평균으로 각 파라미터별 학습률을 자동으로 조절한다 (변화가 큰 방향은 스텝을 줄이고, 작은 방향은 스텝을 키움).
- **Adam (Adaptive Moment Estimation)**: Momentum + RMSprop을 합친 방식으로, 실무에서 가장 널리 쓰이는 기본 optimizer다. CV 아키텍처들(ResNet, Inception 등)을 실제로 학습시킬 때도 대부분 Adam이나 그 변형을 쓴다.
- **Learning rate decay**: 학습이 진행될수록 learning rate를 점점 줄여서, 초반엔 빠르게 수렴하고 후반엔 최적점 근처에서 미세조정하게 만드는 기법.

## 5. Batch Normalization (Course 2)

CV 아키텍처 노트에는 직접 등장하지 않지만, 실제로 ResNet/Inception 등 거의 모든 현대 CNN 구조 내부에 기본적으로 들어가는 매우 중요한 구성 요소라 따로 짚어둔다.

- **아이디어**: input feature를 정규화(평균 0, 분산 1)하면 학습이 빨라진다는 건 이미 알려진 사실인데, 이걸 input layer뿐 아니라 **네트워크 중간의 모든 hidden layer의 activation(정확히는 활성화 함수 통과 전 Z값)에도 적용**하자는 게 batch normalization이다.
- 각 mini-batch마다 `Z`의 평균/분산을 구해서 정규화한 뒤, 학습 가능한 파라미터 `gamma`, `beta`로 다시 스케일/이동시킨다 (정규화를 강제하면서도 네트워크가 필요하면 원래 분포로 되돌릴 수 있는 유연성을 준다).
- **효과**: 레이어 간 입력 분포가 학습 중에 계속 바뀌는 문제(covariate shift)를 줄여서 학습을 안정화하고 빠르게 만들며, 약간의 regularization 효과도 있다. 더 깊은 네트워크를 안정적으로 학습시킬 수 있게 해주는 핵심 도구 중 하나.

## 6. Bias/Variance와 ML Strategy (Course 3)

`Structuring Machine Learning Projects` 코스는 "모델을 어떻게 개선해나갈지"에 대한 방법론을 다루는데, CV 노트의 transfer learning/data augmentation 관련 실무 팁들이 이 프레임 위에 있는 내용이다.

- **Orthogonalization**: "train set 성능이 안 좋다", "dev set 성능이 안 좋다", "test set 성능이 안 좋다", "실제 서비스에서 성능이 안 좋다"는 각각 원인과 해결책이 다른 독립적인 문제이니, 한 번에 하나씩 진단하고 고쳐야 한다는 원칙.
- **Bias(High bias = underfitting) vs Variance(High variance = overfitting)**: train error와 dev error를 human-level performance와 비교해서 진단한다.
  - High bias(train error 자체가 높음) → 더 큰 네트워크, 더 오래 학습
  - High variance(train은 좋은데 dev가 나쁨) → 데이터 추가, regularization(dropout, data augmentation)
  - CV 노트의 "데이터가 적으면 transfer learning/data augmentation을 쓴다"는 실무 팁이 바로 이 variance 문제에 대한 해결책 중 하나다.
- **Human-level performance / Avoidable bias**: 사람 수준의 성능을 bias의 기준선(Bayes error의 근사치)으로 삼고, train error와의 차이(avoidable bias)와 train-dev error 차이(variance)를 나눠서 어디를 개선해야 할지 판단한다.
- **Error analysis**: 틀린 예측들을 직접 훑어보면서 어떤 유형의 오류가 몇 %를 차지하는지 세어보고, 가장 비중이 큰 오류 유형부터 고치는 우선순위를 정하는 방법.
- **Train/dev/test set 분포**: 데이터가 많아진 딥러닝 시대에는 굳이 dev/test를 60/20/20으로 나눌 필요 없이 dev/test는 신뢰할 만큼만(예: 각 수천 개) 떼어놓고 나머지는 다 train에 쓴다는 것, 그리고 dev/test는 반드시 같은 분포에서 뽑아야 한다는 원칙.
- **Transfer learning / Multi-task learning / End-to-end deep learning을 언제 쓰는가**: CV 노트에서 "pretrained model을 가져다 쓴다"고 나온 부분이, 사실 "task A의 데이터가 아주 많고 task B의 데이터가 적을 때, A로 학습한 표현을 B에 재사용한다"는 일반 원칙의 한 사례라는 걸 알아두면 좋다.

## 요약: CV 노트와의 연결 지점

| 이 노트의 개념 | CV 노트에서 등장하는 곳 |
|---|---|
| Deep network 표기법(`W[l]`, `Z[l]` 등) | `01-cnn-foundations.md`의 레이어 표기법 전체 |
| Weight initialization, vanishing/exploding gradient | `02-...architectures.md`의 ResNet(degradation problem) |
| Batch Normalization | `02-...architectures.md`의 Inception/ResNet 구조 (문서에 명시적으로 안 나오지만 실제 구현엔 항상 들어감) |
| Mini-batch, Adam 등 optimizer | 모든 CNN 아키텍처를 실제로 학습시킬 때 |
| Bias/Variance, transfer learning 원칙 | `02-...architectures.md`의 transfer learning/data augmentation 실무 팁 |
| Dropout | AlexNet 등 classic architecture의 overfitting 방지 기법 |
