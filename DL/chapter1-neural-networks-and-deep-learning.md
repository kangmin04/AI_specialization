# Course 1: Neural Networks and Deep Learning

> ML Specialization을 이미 들은 사람 기준 노트다. 선형/로지스틱 회귀, 기본 신경망 개념, TensorFlow 기초, 결정 트리, K-means는 안다고 가정하고, 딥러닝 코스에서 "새로 각 잡고 다시 배우는" 부분 — 특히 표기법(notation)과 벡터화(vectorization), 역전파(backpropagation)를 수식으로 유도하는 부분 — 위주로 정리한다. 미적분을 몰라도 괜찮다. Andrew Ng도 강의에서 계속 얘기하지만, 미분의 정확한 증명보다 "이게 왜 이렇게 움직이는지"에 대한 직관이 훨씬 중요하다.

---

## Week 1. 딥러닝이 왜 지금 뜨는가

### 왜 갑자기 딥러닝인가

신경망(neural network) 자체는 옛날부터 있던 아이디어다. 근데 왜 하필 지금 와서 폭발적으로 쓰이게 됐을까. Andrew Ng이 강의에서 그리는 유명한 그래프가 하나 있는데, x축은 데이터 양(amount of data), y축은 성능(performance)이다.

- 전통적인 알고리즘(SVM, logistic regression 같은 것들)은 데이터를 늘려도 어느 순간부터 성능이 **정체(plateau)**된다.
- 신경망은 데이터가 많아질수록, 그리고 네트워크를 더 크게(scale up) 만들수록 계속 성능이 올라간다. 특히 "큰 신경망 + 많은 데이터" 조합에서 이 우상향이 두드러진다.

즉 스케일(scale)이 성능을 이끈다는 게 핵심 관찰이다. 이걸 가능하게 만든 세 가지 축이 있다.

1. **데이터(data)** — 인터넷, 모바일, IoT로 디지털 데이터가 폭발적으로 쌓임.
2. **연산(computation)** — GPU 등 하드웨어 발전으로 큰 네트워크를 실제로 학습시킬 수 있게 됨.
3. **알고리즘(algorithms)** — 예를 들어 activation function을 sigmoid에서 ReLU로 바꾼 것 하나만으로도 gradient descent가 훨씬 빨리 도는 걸 알게 됨 (sigmoid는 양 끝에서 기울기가 0에 가까워져서 학습이 느려지는데, ReLU는 그 문제가 덜하다 — 이건 Week 3에서 다시 자세히 본다).

여기서 중요한 실무적 통찰: **작은 학습 데이터셋(training set)에서는 어떤 알고리즘이 1등인지 순위가 자주 뒤바뀐다.** 근데 데이터가 커지면 커질수록 큰 신경망이 거의 항상 이긴다. 그래서 "데이터가 많다 → 신경망을 크게 만들어라"가 실무 감각으로 자리잡았다.

### Supervised Learning 응용

ML Specialization에서 이미 supervised learning(입력 x → 출력 y를 매핑하는 학습)은 다뤘으니 개념 자체는 새롭지 않다. Course 1에서 강조하는 건 딥러닝이 이 supervised learning을 어떤 데이터 타입에 적용하는지 넓어졌다는 점이다.

| 입력 x | 출력 y | 응용 | 네트워크 타입 |
|---|---|---|---|
| 집 특징 | 가격 | 부동산 | Standard NN |
| 광고 + 유저 정보 | 클릭 여부(0/1) | 온라인 광고 | Standard NN |
| 이미지 | 객체 클래스(1,...,1000) | 사진 태깅 | CNN |
| 오디오 | 텍스트 | 음성 인식 | RNN |
| 영어 문장 | 중국어 문장 | 기계 번역 | RNN |
| 이미지 + 레이더 정보 | 다른 차의 위치 | 자율주행 | Custom/Hybrid |

정형 데이터(structured data, 표 형태 — 이건 ML Specialization의 결정 트리/회귀에서 다루던 데이터랑 비슷하다)와 비정형 데이터(unstructured data, 이미지·오디오·텍스트)를 구분하는 게 포인트다. 사람은 원래 비정형 데이터를 잘 다루는데(눈으로 보고, 귀로 듣고), 신경망이 최근 몇 년간 비약적으로 잘하게 된 영역이 바로 이 비정형 데이터 쪽이다. 그래서 뉴스에 나오는 화려한 딥러닝 성과(이미지 인식, 음성 인식 등)가 다 이쪽이다.

---

## Week 2. 신경망 관점의 Logistic Regression

로지스틱 회귀 자체는 ML Specialization에서 이미 알고 있을 거다. 여기서 새로 익혀야 하는 건 **딥러닝 코스 표기법(notation)**이다. 이 표기법을 확실히 잡고 가야 Week 3, 4의 수식이 안 헷갈린다.

### 표기법(Notation)

- 학습 샘플 하나: $(x, y)$, $x \in \mathbb{R}^{n_x}$, $y \in \{0, 1\}$
- 학습 샘플 $m$개: $\{(x^{(1)}, y^{(1)}), \dots, (x^{(m)}, y^{(m)})\}$ — 위첨자 소괄호 $(i)$는 "$i$번째 샘플"을 뜻한다. 나중에 나올 위첨자 대괄호 $[l]$(레이어 번호)이랑 헷갈리지 않게 지금부터 구분해서 보는 습관을 들이자.
- **가장 중요한 습관**: 입력 데이터를 행렬로 쌓을 때, ML Specialization에서 흔히 보던 "행 = 샘플" 방식이 아니라 **열(column) 단위로 샘플을 쌓는다.**

$$
X = \begin{bmatrix} | & | & & | \\ x^{(1)} & x^{(2)} & \cdots & x^{(m)} \\ | & | & & | \end{bmatrix} \in \mathbb{R}^{n_x \times m}
$$

즉 $X$의 shape은 `(n_x, m)`이다. `X.shape`을 찍었을 때 `(m, n_x)`가 나오면 뭔가 뒤집힌 거다 — 이 습관 하나만 잘 들여도 나중에 행렬곱 차원 에러를 절반은 피할 수 있다. $Y$도 마찬가지로 $Y \in \mathbb{R}^{1 \times m}$, 즉 행 벡터로 쌓는다.

> **Andrew Ng이 강조하는 습관**: 신경망 코드를 짤 때 각 변수의 shape을 항상 주석으로 달아놓고, 실제로 shape이 맞는지 의심하는 습관을 들이라고 한다. 이건 이 코스 전체를 관통하는 조언이다.

### Logistic Regression을 신경망 언어로

로지스틱 회귀 자체 수식은 익숙할 거다.

$$
\hat{y} = a = \sigma(w^Tx + b), \quad \sigma(z) = \frac{1}{1+e^{-z}}
$$

여기서 $\hat{y}$ 대신 $a$(activation)라는 표기를 쓰는 게 새 포인트다. 이게 앞으로 신경망의 "뉴런 하나의 출력"을 가리키는 표준 표기가 된다. 로지스틱 회귀 = **가장 작은 신경망(뉴런 1개짜리)** 이라고 보면 된다.

### Loss function vs Cost function — 구분해서 쓰기

ML Specialization에서는 이 둘을 크게 구분 안 하고 뭉뚱그려 "cost"라고 부르는 경우가 많았을 텐데, 딥러닝 표기법에서는 명확히 나눈다.

- **Loss function** $\mathcal{L}(a, y)$ — 샘플 **하나**에 대한 에러.
- **Cost function** $J(w, b)$ — 전체 $m$개 샘플에 대한 loss의 **평균**.

로지스틱 회귀의 loss는 제곱오차(squared error)를 안 쓴다. 왜냐면 squared error를 쓰면 $J$가 비볼록(non-convex)이 돼서 gradient descent가 local optimum에 걸릴 위험이 커지기 때문이다. 대신 이런 걸 쓴다.

$$
\mathcal{L}(a, y) = -\big(y\log a + (1-y)\log(1-a)\big)
$$

- $y=1$이면 $\mathcal{L} = -\log a$ → $a$가 1에 가까울수록(=맞출수록) loss가 작아짐.
- $y=0$이면 $\mathcal{L} = -\log(1-a)$ → $a$가 0에 가까울수록 loss가 작아짐.

직관적으로 "맞추면 loss 작고 틀리면 loss 크다"는 로지스틱 회귀에서 봤던 그 cross-entropy 그대로다. 그리고

$$
J(w,b) = \frac{1}{m}\sum_{i=1}^{m}\mathcal{L}(a^{(i)}, y^{(i)})
$$

**cost는 convex**해서 gradient descent가 안정적으로 전역 최적점(global optimum)을 향해 내려간다.

### Gradient Descent — 복습 + 표기

$$
w := w - \alpha \frac{\partial J(w,b)}{\partial w}, \qquad b := b - \alpha \frac{\partial J(w,b)}{\partial b}
$$

$\alpha$는 learning rate. 코드에서는 $\partial J/\partial w$를 그냥 `dw`라고 쓰는 관습이 이 코스부터 확실히 자리잡는다. 앞으로 `dw`, `db`, `dz` 이런 식으로 "이 변수에 대한 최종 loss의 미분"을 줄여 쓰는 표기가 계속 나오니 익숙해지자.

### Computation Graph — 왜 이걸 그리는가

신경망 학습은 크게 두 방향으로 움직인다.

- **Forward pass**: 입력에서 출력(그리고 cost)까지 계산.
- **Backward pass**: cost에서 거꾸로 각 파라미터에 대한 미분을 계산 (= backpropagation).

이 두 흐름을 눈에 보이게 그린 게 computation graph다. 예를 들어 $J(a,b,c) = 3(a+bc)$ 같은 간단한 식이 있으면,

```
b, c → u = bc
a, u → v = a + u
v → J = 3v
```

이렇게 노드를 쭉 이어 그린다. Forward는 왼쪽에서 오른쪽으로 값을 계산하는 거고, backward(미분)는 **오른쪽에서 왼쪽으로** chain rule을 타고 가는 거다. 즉 딥러닝의 backprop이 결국 chain rule을 그래프 위에서 체계적으로 적용하는 것뿐이라는 걸 이 그래프가 보여준다.

미적분이 약해도 괜찮은 이유가 여기 있다 — 실제로 필요한 미분 규칙은 "합성함수 미분(chain rule)"과 몇 개 기본 함수의 도함수뿐이고, 나머지는 이 그래프를 손으로 한 단계씩 따라가기만 하면 유도된다.

### 미분 직관 (아주 러프하게)

$f(a) = 3a$면 $a$를 아주 조금 늘렸을 때 $f$는 3배만큼 늘어난다 — 이게 도함수 $\frac{df}{da}=3$의 의미다. "$a$를 조금 바꾸면 결과가 얼마나 바뀌는가"의 비율. 로지스틱 회귀에서 필요한 도함수는 딱 두 개만 외워도 충분하다.

$$
\frac{d}{dz}\sigma(z) = \sigma(z)(1-\sigma(z)), \qquad \frac{d}{dz}\log z = \frac{1}{z}
$$

### Logistic Regression의 역전파 유도

이게 이번 주 핵심이다. 샘플 하나에 대해 chain rule을 손으로 타고 가보자.

Forward:
$$
z = w^Tx + b, \quad a = \sigma(z), \quad \mathcal{L}(a,y) = -(y\log a + (1-y)\log(1-a))
$$

Backward (뒤에서부터):

$$
da = \frac{\partial \mathcal{L}}{\partial a} = -\frac{y}{a} + \frac{1-y}{1-a}
$$

$$
dz = \frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial a}\cdot\frac{\partial a}{\partial z} = da \cdot a(1-a) = a - y
$$

이 $dz = a-y$가 로지스틱 회귀 전체에서 가장 중요한 한 줄이라고 봐도 된다. 복잡해 보이던 cross-entropy loss와 sigmoid 미분이 싹 정리돼서 **"예측 - 정답"**이라는 극도로 단순한 형태로 떨어진다. 이게 나중에 신경망 출력층에서도 그대로 반복되는 패턴이라 꼭 기억해두자.

이어서 $w$, $b$에 대한 미분은:

$$
dw_1 = x_1 \cdot dz, \quad dw_2 = x_2 \cdot dz, \quad db = dz
$$

$n_x$개 feature, $m$개 샘플 전체로 확장하면 (지금은 for문 버전, 벡터화는 바로 다음 절):

```python
J, dw1, dw2, db = 0, 0, 0, 0
for i in range(m):
    z_i = w1*x1[i] + w2*x2[i] + b
    a_i = sigmoid(z_i)
    J += -(y[i]*log(a_i) + (1-y[i])*log(1-a_i))
    dz_i = a_i - y[i]
    dw1 += x1[i]*dz_i
    dw2 += x2[i]*dz_i
    db  += dz_i
J /= m; dw1 /= m; dw2 /= m; db /= m
```

### Vectorization — for문을 없애라

위 코드는 for문이 두 겹(샘플 루프 + feature 루프)이다. 딥러닝에서는 데이터가 수백만 개일 수 있어서 이런 for문은 치명적으로 느리다. **Vectorization**은 이 루프를 행렬/벡터 연산으로 바꿔서 numpy(내부적으로 SIMD, 병렬 하드웨어 활용)가 대신 처리하게 만드는 거다. GPU/CPU 둘 다 이런 벡터 연산에 최적화돼 있다 — 이걸 SIMD(Single Instruction Multiple Data) 원리라고 부른다.

$m$개 샘플 전체를 한 번에:

$$
Z = w^TX + b, \qquad A = \sigma(Z), \qquad dZ = A - Y
$$

$$
dw = \frac{1}{m}XdZ^T, \qquad db = \frac{1}{m}\text{np.sum}(dZ)
$$

```python
import numpy as np

Z = np.dot(w.T, X) + b        # b는 스칼라지만 broadcasting으로 (1,m)에 자동 확장
A = sigmoid(Z)                 # (1, m)
dZ = A - Y                     # (1, m)
dw = (1/m) * np.dot(X, dZ.T)   # (n_x, 1)
db = (1/m) * np.sum(dZ)        # 스칼라

w -= alpha * dw
b -= alpha * db
```

for문이 완전히 사라졌다 (m개 샘플에 대한 루프도, 여러 iteration의 gradient descent 루프는 어차피 남지만 그건 별개다). `np.dot(w.T, X) + b`에서 `b`가 스칼라인데 `(1,m)` 벡터에 자동으로 더해지는 게 바로 **broadcasting**이다.

### Broadcasting 규칙

numpy는 두 배열의 shape이 정확히 안 맞아도, 마지막 차원부터 비교해서 "1이거나 같으면" 자동으로 복사(늘리기)해서 연산한다.

```python
# (4,3) 행렬에 (1,3) 벡터를 더하면 -> 행렬의 각 행에 같은 벡터가 더해짐
A = np.random.randn(4, 3)
b = np.array([[100, 200, 300]])   # shape (1,3)
A + b   # b가 4번 복제된 것처럼 동작 -> (4,3)

# (4,1) 벡터를 (4,3) 행렬에 더하면 -> 각 열에 같은 벡터가 더해짐
```

일반 규칙: $(m,n)$과 $(1,n)$ 또는 $(m,1)$을 연산하면 작은 쪽이 큰 쪽 shape으로 자동 확장된다. 편리하지만 **의도치 않은 broadcasting이 조용히 버그를 만든다**는 게 함정이다.

### Rank-1 array 함정 — 꼭 조심할 것

```python
a = np.random.randn(5)
a.shape        # (5,)  <- 이게 문제. 이건 벡터도 행렬도 아닌 "rank 1 array"
a.T.shape      # (5,)  <- transpose 해도 그대로! 기대와 다르게 동작
np.dot(a, a.T) # 스칼라가 나옴 (내적처럼 동작) -> 행렬을 기대했다면 버그
```

`(5,)` 같은 shape은 행 벡터도 열 벡터도 아니라서 transpose가 아무 효과가 없고, 곱셈 결과도 직관과 다르게 나올 수 있다. **해결책은 딱 하나, shape을 항상 명시적으로 지정하는 습관**이다.

```python
a = np.random.randn(5, 1)     # 명확한 열벡터, shape (5,1)
assert a.shape == (5, 1)      # 이렇게 assert를 박아두는 것도 좋은 습관
```

이게 이번 주에서 실무적으로 가장 자주 물리는 버그라서, Andrew Ng도 "rank-1 array 쓰지 마라"를 반복해서 강조한다.

---

## Week 3. Shallow Neural Network

이제 뉴런 1개(로지스틱 회귀)에서 **레이어(layer)** 개념으로 확장한다. Hidden layer 1개 + output layer 1개짜리 얕은(shallow) 신경망을 다룬다.

### 표기법 확장

- 레이어 번호는 위첨자 대괄호: $W^{[1]}, b^{[1]}$ = 첫 번째 레이어(hidden layer)의 파라미터, $W^{[2]}, b^{[2]}$ = 두 번째 레이어(output layer).
- $a^{[l]}$ = $l$번째 레이어의 activation, $a^{[0]} = x$ (입력을 레이어 0으로 취급).
- 샘플 번호까지 합치면 $a^{[l](i)}$ = "$i$번째 샘플의 $l$번째 레이어 activation". 대괄호는 레이어, 소괄호는 샘플 — Week 2에서 잡아둔 구분이 여기서 그대로 쓰인다.
- Hidden layer 유닛이 4개, 입력 feature가 3개라면 $W^{[1]} \in \mathbb{R}^{4\times 3}$, $b^{[1]} \in \mathbb{R}^{4\times 1}$. **"(다음 레이어 유닛 수, 이전 레이어 유닛 수)"가 $W$의 shape 규칙**이다 — 이건 Week 4에서 일반화한다.

### Forward Propagation (벡터화)

샘플 1개 기준:

$$
z^{[1]} = W^{[1]}x + b^{[1]}, \quad a^{[1]} = g^{[1]}(z^{[1]})
$$

$$
z^{[2]} = W^{[2]}a^{[1]} + b^{[2]}, \quad a^{[2]} = g^{[2]}(z^{[2]}) = \hat{y}
$$

$m$개 샘플을 열로 쌓은 $X$ 전체로 확장 ($Z^{[1]}, A^{[1]}$ 등은 각 열이 한 샘플):

$$
Z^{[1]} = W^{[1]}X + b^{[1]}, \quad A^{[1]} = g^{[1]}(Z^{[1]})
$$

$$
Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}, \quad A^{[2]} = g^{[2]}(Z^{[2]})
$$

for문은 "샘플 개수만큼"이 아니라 오직 **"레이어 개수만큼"**만 남는다 — 이게 벡터화의 핵심 이득이다.

### Activation Function 선택 가이드

ML Specialization에서 sigmoid는 이미 봤을 거다. 여기서 tanh, ReLU, Leaky ReLU까지 비교하고 "언제 뭘 쓰는가"를 배운다.

| 함수 | 수식 | 범위 | 도함수 | 언제 쓰나 |
|---|---|---|---|---|
| Sigmoid | $\sigma(z)=\frac{1}{1+e^{-z}}$ | $(0,1)$ | $a(1-a)$ | **output layer**에서 이진 분류일 때만 (확률 해석 필요할 때). hidden layer에는 거의 안 씀 |
| tanh | $\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}$ | $(-1,1)$ | $1-a^2$ | hidden layer에서 sigmoid보다 거의 항상 낫다 (평균이 0에 가까워서 다음 레이어 학습이 편해짐) |
| ReLU | $\max(0,z)$ | $[0,\infty)$ | $z>0$이면 1, $z<0$이면 0 | **기본값(default)**. 대부분의 hidden layer에서 가장 무난하게 잘 됨 |
| Leaky ReLU | $\max(0.01z, z)$ | $(-\infty,\infty)$ | $z>0$이면 1, $z<0$이면 0.01 | ReLU가 "죽는(dying ReLU)" 게 걱정될 때 |

핵심 이유: sigmoid와 tanh는 $z$가 아주 크거나 작을 때 기울기가 0에 가까워져서(saturate) gradient descent가 느려진다. ReLU는 $z>0$ 구간에서 기울기가 항상 1이라 학습이 훨씬 빠르다. 이게 Week 1에서 "알고리즘 발전이 딥러닝을 가속했다"고 한 예시 중 하나가 바로 이거다.

> 실무 팁(Andrew Ng): 확신 없으면 **hidden layer는 ReLU를 기본으로 쓰고**, output layer만 문제 종류에 맞춰라(이진 분류 → sigmoid). 모든 레이어에 같은 activation을 고집할 필요 없다.

### 왜 비선형(non-linear) activation이 꼭 필요한가

만약 모든 레이어에서 activation을 안 쓰거나(=linear activation, $g(z)=z$) 쓴다면, 아무리 레이어를 깊게 쌓아도 결국 **선형 함수들의 합성 = 또 다른 선형 함수**라서, 신경망 전체가 그냥 로지스틱/선형 회귀 한 겹이랑 다를 게 없어진다. Hidden layer가 뭔가 "흥미로운" 함수를 배우려면 반드시 비선형 activation이 있어야 한다. (참고로 회귀 문제의 output layer 하나 정도는 linear activation을 쓰기도 한다 — 예: 집값 예측처럼 출력이 실수 전체 범위일 때.)

### Random Initialization — 대칭성(symmetry) 문제

로지스틱 회귀에서는 $w=0$으로 초기화해도 문제없었다. 근데 신경망에서 **모든 가중치를 0(또는 같은 값)으로 초기화하면 절대 안 된다.**

이유: 같은 레이어의 유닛들이 전부 똑같은 값으로 시작하면, forward에서 같은 값을 계산하고, backward에서도 같은 gradient를 받아서, 아무리 학습을 반복해도 **그 유닛들이 영원히 똑같은 함수를 계산**하게 된다 (symmetry breaking이 안 됨). Hidden unit을 여러 개 두는 의미가 사라지는 거다.

그래서 $W$는 작은 랜덤값으로 초기화한다.

```python
W1 = np.random.randn(n1, n0) * 0.01
b1 = np.zeros((n1, 1))   # b는 0으로 초기화해도 대칭성 문제 없음 (W가 이미 깨줌)
```

왜 `*0.01`처럼 **작게** 만드는가: $W$가 크면 $z=Wx+b$도 커지고, sigmoid/tanh 기준으로 $z$가 크면 기울기가 0에 가까운 saturate 구간에 걸려서 학습이 느려진다. 그래서 초기값은 작게 잡아서 활성값이 기울기가 살아있는 구간에서 시작하게 만든다. (레이어가 아주 깊어지면 이 `0.01` 같은 고정 상수 대신 Xavier/He 초기화 같은 좀 더 정교한 방법을 쓰는데, 그건 Course 2에서 다룰 내용이다.)

### 미니 코드 스니펫 — 1 hidden layer 학습 루프

```python
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))

def forward(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)                # hidden layer
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)                # output layer (binary classification)
    return Z1, A1, Z2, A2

def backward(X, Y, Z1, A1, A2, W2):
    m = X.shape[1]
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))   # tanh 도함수
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2
```

`axis=1, keepdims=True`를 꼭 챙기자 — 안 그러면 다시 rank-1 array 함정에 빠진다.

---

## Week 4. Deep L-layer Network

이제 hidden layer가 여러 개인 진짜 "딥"러닝으로 간다. 구조 자체는 Week 3이랑 똑같고, 레이어 개수만 $L$개로 일반화하는 거다.

### 표기법 일반화

- $L$ = 전체 레이어 개수 (output layer 포함, 입력층은 세지 않음).
- $n^{[l]}$ = $l$번째 레이어의 유닛(노드) 개수. $n^{[0]} = n_x$(입력 feature 수).
- $a^{[l]} = g^{[l]}(z^{[l]})$, $a^{[0]} = X$, $a^{[L]} = \hat{y}$.

### 차원(shape) 체크표 — 이 코스에서 제일 자주 쓰는 표

레이어를 몇 개를 쌓든, **아래 규칙만 지키면 차원 에러가 안 난다.**

| 변수 | Shape | 비고 |
|---|---|---|
| $W^{[l]}$ | $(n^{[l]}, n^{[l-1]})$ | (현재 레이어 유닛 수, 이전 레이어 유닛 수) |
| $b^{[l]}$ | $(n^{[l]}, 1)$ | broadcasting으로 $m$개 샘플에 확장됨 |
| $Z^{[l]}, A^{[l]}$ | $(n^{[l]}, m)$ | $m$ = 샘플 개수(배치 전체) |
| $dW^{[l]}$ | $(n^{[l]}, n^{[l-1]})$ | $W^{[l]}$과 shape 동일 |
| $db^{[l]}$ | $(n^{[l]}, 1)$ | $b^{[l]}$과 shape 동일 |
| $dZ^{[l]}, dA^{[l]}$ | $(n^{[l]}, m)$ | $Z^{[l]}, A^{[l]}$과 shape 동일 |

**습관**: 코드를 짜다가 막히면 종이에 이 표부터 채워보라는 게 Andrew Ng의 조언이다. `assert W1.shape == (n1, n0)` 같은 assert를 실제 코드에 박아두는 것도 실무에서 매우 유용하다.

### 왜 Deep인가 (레이어를 깊게 쌓는 이유)

두 가지 직관을 강의에서 제시한다.

1. **계층적 feature 학습(hierarchical feature learning)**: 예를 들어 얼굴 인식이면, 앞쪽 레이어는 edge(선) 같은 단순한 패턴을 찾고, 중간 레이어는 그 edge들을 모아 눈·코·입 같은 부위를 찾고, 뒤쪽 레이어는 그 부위들을 모아 얼굴 전체를 인식한다. 즉 **단순한 것 → 복잡한 것으로 계층적으로 쌓아 올리는** 구조라서, 얕은 네트워크로 같은 일을 하려면 유닛 수가 기하급수적으로 더 많이 필요해진다.
2. **회로 이론(circuit theory) 직관**: 예를 들어 $n$개의 입력 변수에 대한 XOR을 전부 계산하는 문제가 있으면, 깊은 네트워크는 $O(\log n)$ 레이어로 풀 수 있는데, 얕은(hidden layer 1개짜리) 네트워크로 똑같은 걸 하려면 유닛 수가 $O(2^n)$으로 지수적으로 늘어난다. 즉 "깊이"가 "너비"를 대체하면서 필요한 파라미터 수를 극적으로 줄여준다.

그렇다고 무조건 깊게 쌓는 게 능사는 아니다 — 레이어 개수(L), hidden unit 개수 같은 건 정답이 정해진 게 아니라 **경험적으로 실험해서(hyperparameter search)** 찾는 값이다. 아래에서 바로 이어진다.

### Forward/Backward 블록 구조와 Cache

레이어 하나하나를 "블록(block)"으로 생각하면 편하다.

- **Forward 블록** ($l$번째): 입력 $a^{[l-1]}$을 받아서 $z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$, $a^{[l]} = g^{[l]}(z^{[l]})$을 계산하고 출력 $a^{[l]}$을 다음 블록에 넘긴다. 이때 **$z^{[l]}$(그리고 $W^{[l]}, b^{[l]}$)을 "cache"에 저장**해둔다 — backward 계산할 때 다시 필요하기 때문이다.
- **Backward 블록** ($l$번째): $da^{[l]}$을 입력으로 받아서, forward 때 저장해둔 cache를 이용해 $dz^{[l]}, dW^{[l]}, db^{[l]}, da^{[l-1]}$을 계산하고, $da^{[l-1]}$을 앞쪽 블록으로 넘긴다.

전체 흐름을 그리면:

```
X=a[0] --W1,b1--> a[1] --W2,b2--> a[2] --...--> a[L] = y_hat --> L(y_hat,y)
                (forward, cache 저장)

dA[L] <-- dL/dA[L]
   |
   v
da[l-1] <--(W[l],b[l],z[l] cache 이용)-- da[l] --...-- da[L]
                (backward, chain rule)
```

레이어별로 필요한 미분 공식은 Week 3에서 본 거랑 형태가 똑같고, 인덱스만 $l$로 일반화된다.

$$
dZ^{[l]} = dA^{[l]} * g^{[l]\prime}(Z^{[l]})
$$

$$
dW^{[l]} = \frac{1}{m}dZ^{[l]}A^{[l-1]T}, \quad db^{[l]} = \frac{1}{m}\text{np.sum}(dZ^{[l]}, \text{axis=1, keepdims=True})
$$

$$
dA^{[l-1]} = W^{[l]T}dZ^{[l]}
$$

### 미니 코드 스니펫 — L-layer forward 루프

```python
import numpy as np

def linear_activation_forward(A_prev, W, b, activation):
    Z = np.dot(W, A_prev) + b
    if activation == "relu":
        A = np.maximum(0, Z)
    elif activation == "sigmoid":
        A = 1 / (1 + np.exp(-Z))
    cache = (A_prev, W, b, Z)   # 나중에 backward에서 다시 쓸 값들
    return A, cache

def L_model_forward(X, parameters, L):
    caches = []
    A = X
    for l in range(1, L):                          # 1 ~ L-1: hidden layers
        A_prev = A
        W, b = parameters[f"W{l}"], parameters[f"b{l}"]
        A, cache = linear_activation_forward(A_prev, W, b, "relu")
        caches.append(cache)
    WL, bL = parameters[f"W{L}"], parameters[f"b{L}"]
    AL, cache = linear_activation_forward(A, WL, bL, "sigmoid")  # output layer
    caches.append(cache)
    return AL, caches
```

for문이 "레이어 개수 $L$번"만큼만 도는 게 포인트다. 이 안에서 다시 샘플 루프를 도는 순간 벡터화가 깨진 것이니 조심하자.

### Parameters vs Hyperparameters

- **Parameters** — 학습 알고리즘(gradient descent)이 **데이터로부터 직접 학습**하는 값. $W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}, \dots$
- **Hyperparameters** — 학습이 시작되기 전에 **사람이 미리 정해야 하는** 값. 이 값들이 결국 parameters가 어떻게 학습될지를 "제어(control)"한다.
  - learning rate $\alpha$
  - iteration 횟수
  - hidden layer 개수 $L$
  - 각 레이어의 hidden unit 개수 $n^{[l]}$
  - activation function 종류
  - (나중 코스에서: momentum, mini-batch 크기, regularization 파라미터 등)

하이퍼파라미터는 정답이 있는 게 아니라서, **Idea → Code → Experiment**를 계속 반복하면서 실험적으로 감을 잡아야 한다. Andrew Ng은 이걸 "딥러닝은 매우 경험적인(empirical) 프로세스"라고 표현한다 — 처음부터 최적값을 안다는 보장이 없으니, 여러 값을 실제로 돌려보고 cost curve를 보면서 조정하는 게 일반적인 워크플로우다.

### "그래서 뇌(brain)와 무슨 관계가 있나?"

신경망(neural network)이라는 이름 때문에 사람 뇌를 그대로 모방한 거라고 오해하기 쉬운데, Andrew Ng은 이 비유를 **아주 약하게만** 받아들이라고 강의에서 여러 번 경고한다. 하나의 뉴런이 신호를 받아서 활성화되는 걸 "입력을 받아 activation을 계산하는 로지스틱 유닛"에 느슨하게 비유한 것뿐이지, 실제 생물학적 뉴런이 어떻게 학습하는지는 아직도 신경과학적으로 제대로 규명되지 않았다. 딥러닝 알고리즘(gradient descent, backprop)이 뇌의 학습 방식과 실제로 같다는 근거는 없다. 그러니 "인공신경망 = 뇌를 흉내낸 것"이라는 대중적 프레이밍에 너무 의미를 두지 말고, 그냥 **입력→출력을 매핑하는 함수 근사기(function approximator)를 레이어로 쌓은 것**이라는 수학적 관점으로 이해하는 게 훨씬 정확하다.

---

## 핵심 요약

- 딥러닝이 뜬 이유는 데이터·연산·알고리즘 세 축의 스케일업이다. 데이터가 많아질수록 큰 신경망이 전통 알고리즘의 성능 정체(plateau)를 뚫고 계속 좋아진다.
- $X$는 `(n_x, m)`, 샘플은 항상 **열(column)** 로 쌓는다. 이 규칙이 이후 모든 shape 계산의 기준이 된다.
- Loss(샘플 1개) vs Cost(전체 평균)를 구분한다. Cross-entropy loss를 쓰는 이유는 cost를 convex하게 만들어 gradient descent가 안정적으로 수렴하게 하기 위해서다.
- Logistic regression의 역전파는 $dz = a-y$ 하나로 요약된다. 이 패턴은 신경망 output layer에서도 그대로 반복된다.
- Vectorization(for문 제거) + broadcasting은 속도뿐 아니라 코드 자체를 단순하게 만든다. 단, broadcasting이 의도치 않게 발생할 수 있으니 shape을 항상 의심할 것.
- Activation function은 hidden layer에는 기본적으로 ReLU, output layer는 문제 성격에 맞춰(이진 분류 → sigmoid) 고른다. Non-linear activation이 없으면 아무리 깊게 쌓아도 선형 모델과 다를 게 없다.
- 가중치는 반드시 작은 랜덤값으로 초기화한다(symmetry breaking). $b$는 0으로 초기화해도 무방하다.
- Deep network의 shape 규칙: $W^{[l]}: (n^{[l]}, n^{[l-1]})$, $b^{[l]}: (n^{[l]}, 1)$, $Z^{[l]}, A^{[l]}: (n^{[l]}, m)$. $dW, db, dZ, dA$는 각각 대응되는 변수와 shape이 같다.
- Forward에서 cache(주로 $Z^{[l]}, W^{[l]}, b^{[l]}, A^{[l-1]}$)를 저장해둬야 backward에서 재사용할 수 있다.
- Parameters(학습되는 값)와 Hyperparameters(사람이 미리 정하는 값)를 구분하고, 하이퍼파라미터 튜닝은 Idea-Code-Experiment 반복이 기본이라는 걸 받아들이자.

## 헷갈리기 쉬운 점

- **위첨자 소괄호 vs 대괄호**: $x^{(i)}$는 $i$번째 **샘플**, $a^{[l]}$은 $l$번째 **레이어**. $a^{[l](i)}$처럼 같이 쓰이면 "$i$번째 샘플의 $l$번째 레이어 activation"이다.
- **$X$의 shape 방향**: `(n_x, m)`이지 `(m, n_x)`가 아니다. scikit-learn 등 다른 라이브러리에 익숙하면 반대로 짜기 쉬우니 특히 주의.
- **Rank-1 array**: `(5,)` shape은 벡터도 행렬도 아니다. 항상 `(5,1)`처럼 명시적 2차원으로 만들자. `.T`가 안 먹는 것 같으면 rank-1 array를 의심할 것.
- **$dz=a-y$가 왜 이렇게 단순해지는가**: cross-entropy loss와 sigmoid의 도함수가 서로 상쇄되면서 생기는 결과다. "우연히 간단해진 것"이 아니라 이 loss와 activation 조합을 의도적으로 고른 이유이기도 하다.
- **Activation 선택**: sigmoid를 hidden layer에도 무조건 써야 한다고 착각하기 쉬운데, hidden layer 기본값은 ReLU다. Sigmoid는 output layer의 "확률이 필요할 때"로 한정해서 생각하자.
- **가중치 초기화를 0으로 하면 왜 안 되는가**: "학습이 느려진다" 정도가 아니라, 같은 레이어의 유닛들이 **영원히 서로 구분되지 않는(symmetric)** 심각한 문제다. $b=0$은 괜찮지만 $W=0$은 절대 안 된다.
- **딥(deep)이 무조건 좋은 건 아니다**: 레이어를 깊게 쌓으면 회로 이론적으로 표현력은 늘어나지만, $L$이나 $n^{[l]}$ 같은 값 자체는 정답이 없고 실험으로 찾아야 하는 하이퍼파라미터다.
- **신경망 ≠ 뇌 시뮬레이션**: 이름 때문에 생기는 오해일 뿐, 생물학적 근거가 있는 비유가 아니다. 함수 근사기를 레이어로 쌓은 수학적 모델로 이해하는 게 정확하다.

---

## Lab 예제 — 직접 짜보기

> numpy + matplotlib만 쓴다. 뼈대 코드의 `TODO` 부분을 직접 채우고 실행해서 **체크포인트** 출력과 비교해보자. 막히면 접혀있는 정답을 열어보기. 전부 복붙하면 바로 돌아가는 단일 스크립트다.

### Lab 1. 벡터화된 Logistic Regression — for문 vs 벡터화

**목표**: $dZ = A - Y$, $dw = \frac{1}{m}XdZ^T$를 직접 구현하고, for문 버전과 결과는 같은데 속도가 얼마나 차이나는지 눈으로 확인한다. 마지막에 rank-1 array 함정도 직접 찍어본다.

```python
import time
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def propagate(w, b, X, Y):
    # w: (n_x, 1), X: (n_x, m), Y: (1, m)
    m = X.shape[1]
    A = None      # TODO: (1, m)
    cost = None   # TODO: cross-entropy cost (스칼라)
    dZ = None     # TODO: (1, m)
    dw = None     # TODO: (n_x, 1)
    db = None     # TODO: 스칼라
    return dw, db, cost

def propagate_loop(w, b, X, Y):
    # 비교용: 샘플 루프 + feature 루프 두 겹짜리 버전
    n_x, m = X.shape
    cost, dw, db = 0.0, np.zeros((n_x, 1)), 0.0
    for i in range(m):
        z = sum(w[j, 0] * X[j, i] for j in range(n_x)) + b
        a = sigmoid(z)
        cost += -(Y[0, i] * np.log(a) + (1 - Y[0, i]) * np.log(1 - a))
        dz = a - Y[0, i]
        for j in range(n_x):
            dw[j, 0] += X[j, i] * dz
        db += dz
    return dw / m, db / m, cost / m

np.random.seed(1)
m = 20000
X = np.random.randn(2, m)
Y = (X[0:1, :] + 2 * X[1:2, :] > 0.5).astype(float)   # 정답 경계: x1 + 2*x2 = 0.5
w, b = np.zeros((2, 1)), 0.0

t0 = time.time(); dw1, db1, c1 = propagate_loop(w, b, X, Y); t_loop = time.time() - t0
t0 = time.time(); dw2, db2, c2 = propagate(w, b, X, Y); t_vec = time.time() - t0
print(f"loop: {t_loop*1000:.1f}ms, vectorized: {t_vec*1000:.3f}ms")
print("같은 결과?", np.allclose(dw1, dw2), np.isclose(db1, db2), np.isclose(c1, c2))

costs = []
for i in range(1000):
    dw, db, cost = propagate(w, b, X, Y)
    # TODO: gradient descent 업데이트 (learning rate 0.5)
    if i % 100 == 0:
        costs.append(cost)
acc = np.mean((sigmoid(np.dot(w.T, X) + b) > 0.5) == Y)
print(f"train acc: {acc:.3f}, w: {w.ravel()}, b: {b:.3f}")
plt.plot(costs); plt.xlabel("iteration (x100)"); plt.ylabel("cost"); plt.show()

# rank-1 array 함정 직접 확인
a = np.random.randn(5)
print(a.shape, a.T.shape, np.dot(a, a.T).shape)
a = np.random.randn(5, 1)
print(a.shape, a.T.shape, np.dot(a, a.T).shape)
```

**체크포인트**
- `같은 결과? True True True`, 속도는 벡터화 쪽이 수십~수백 배 빠르다.
- train acc ≈ 0.99 이상, `w`의 비율이 대략 1 : 2 (정답 경계 $x_1 + 2x_2 = 0.5$와 같은 방향).
- rank-1 array는 `(5,) (5,) ()` → `.T`가 아무 효과도 없고 내적이 스칼라로 나온다. `(5,1)`로 만들면 `(5, 1) (1, 5) (5, 5)`.

<details>
<summary>정답 보기</summary>

```python
def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)                              # (1, m)
    cost = -np.mean(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    dZ = A - Y                                                   # (1, m)
    dw = np.dot(X, dZ.T) / m                                     # (n_x, 1)
    db = np.sum(dZ) / m
    return dw, db, cost

# 학습 루프 안
    w -= 0.5 * dw
    b -= 0.5 * db
```

</details>

### Lab 2. Shallow NN으로 로지스틱 회귀가 못 푸는 데이터 풀기

**목표**: 꽃잎 모양(planar) 데이터에서 로지스틱 회귀(= hidden layer 없는 신경망)는 직선 경계밖에 못 그어서 실패하는 걸 보고, tanh hidden layer 1개짜리 신경망의 forward/backward를 직접 짜서 해결한다. hidden unit 수 $n^{[1]}$을 바꿔가며 결정 경계가 어떻게 달라지는지도 본다.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z): return 1 / (1 + np.exp(-z))

def make_flower(m=400, seed=1):
    np.random.seed(seed)
    N = m // 2
    X = np.zeros((m, 2)); Y = np.zeros((m, 1))
    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2
        r = 4 * np.sin(4 * t) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j
    return X.T, Y.T          # X: (2, m), Y: (1, m) — 샘플은 열로!

X, Y = make_flower()

def init_params(n_x, n_h, n_y):
    # TODO: W는 작은 랜덤값(*0.01), b는 0으로
    return {"W1": None, "b1": None, "W2": None, "b2": None}

def forward(X, p):
    # TODO: Z1 -> A1(tanh) -> Z2 -> A2(sigmoid)
    A1, A2 = None, None
    return A2, (A1, A2)

def compute_cost(A2, Y):
    return -np.mean(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))

def backward(X, Y, p, cache):
    m = X.shape[1]
    A1, A2 = cache
    # TODO: dZ2, dW2, db2, dZ1 (tanh 도함수 = 1 - A1^2), dW1, db1
    # 힌트: np.sum(..., axis=1, keepdims=True) 안 쓰면 rank-1 array 함정
    return {"W1": None, "b1": None, "W2": None, "b2": None}

def nn_model(X, Y, n_h, num_iter=10000, lr=1.2):
    np.random.seed(3)
    p = init_params(X.shape[0], n_h, 1)
    for i in range(num_iter):
        A2, cache = forward(X, p)
        grads = backward(X, Y, p, cache)
        for k in p:
            p[k] -= lr * grads[k]
        if i % 2000 == 0:
            print(f"  iter {i}: cost {compute_cost(A2, Y):.4f}")
    return p

def plot_boundary(predict, X, Y, title):
    xx, yy = np.meshgrid(np.linspace(X[0].min() - 1, X[0].max() + 1, 200),
                         np.linspace(X[1].min() - 1, X[1].max() + 1, 200))
    Z = predict(np.c_[xx.ravel(), yy.ravel()].T).reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.6)
    plt.scatter(X[0], X[1], c=Y.ravel(), s=15, cmap=plt.cm.Spectral)
    plt.title(title); plt.show()

# 비교 1: 로지스틱 회귀 (Lab 1의 propagate 재사용)
w, b = np.zeros((2, 1)), 0.0
for i in range(3000):
    A = sigmoid(np.dot(w.T, X) + b)
    w -= 0.1 * np.dot(X, (A - Y).T) / X.shape[1]
    b -= 0.1 * np.mean(A - Y)
acc_lr = np.mean((sigmoid(np.dot(w.T, X) + b) > 0.5) == Y)
print(f"logistic regression acc: {acc_lr:.3f}")
plot_boundary(lambda Xg: sigmoid(np.dot(w.T, Xg) + b) > 0.5, X, Y, "Logistic regression")

# 비교 2: hidden unit 수를 바꿔가며
for n_h in [1, 4, 20]:
    print(f"n_h = {n_h}")
    p = nn_model(X, Y, n_h)
    acc = np.mean((forward(X, p)[0] > 0.5) == Y)
    print(f"  acc: {acc:.3f}")
    plot_boundary(lambda Xg: forward(Xg, p)[0] > 0.5, X, Y, f"NN n_h={n_h}")
```

**체크포인트**
- 로지스틱 회귀 acc ≈ 0.47 — 동전 던지기 수준. 직선 하나로는 꽃잎을 못 가른다.
- `n_h=1` → acc ≈ 0.67, `n_h=4` → ≈ 0.91, `n_h=20` → ≈ 0.92. hidden unit이 1개면 사실상 로지스틱 회귀랑 비슷하고, 4개만 돼도 꽃잎 모양 경계가 생긴다.
- 모든 레이어를 linear activation으로 바꾸면(`np.tanh` 대신 그냥 `Z1`) 어떻게 되는지도 직접 해보자 → n_h를 아무리 늘려도 로지스틱 회귀 수준으로 떨어진다(비선형 activation이 필요한 이유).

<details>
<summary>정답 보기</summary>

```python
def init_params(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def forward(X, p):
    Z1 = np.dot(p["W1"], X) + p["b1"]
    A1 = np.tanh(Z1)
    Z2 = np.dot(p["W2"], A1) + p["b2"]
    A2 = sigmoid(Z2)
    return A2, (A1, A2)

def backward(X, Y, p, cache):
    m = X.shape[1]
    A1, A2 = cache
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(p["W2"].T, dZ2) * (1 - A1 ** 2)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
```

</details>

### Lab 3. L-layer 신경망을 블록 단위로 조립하기

**목표**: Week 4의 "forward 블록 / backward 블록 / cache" 구조를 그대로 코드로 옮긴다. 레이어 수를 리스트 하나(`layer_dims`)로 바꿀 수 있게 일반화하고, shape 체크표를 `assert`로 박아둔다.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z): return 1 / (1 + np.exp(-z))

def initialize_parameters_deep(layer_dims):
    # layer_dims 예: [2, 16, 8, 1] -> 입력 2, hidden 16, hidden 8, output 1
    np.random.seed(1)
    params = {}
    L = len(layer_dims) - 1
    for l in range(1, L + 1):
        # TODO: W{l} shape = (n[l], n[l-1]), b{l} shape = (n[l], 1)
        # (깊어지면 *0.01 대신 * np.sqrt(2 / n[l-1]) 가 잘 됨 — He 초기화, Course 2에서 다룸)
        assert params[f"W{l}"].shape == (layer_dims[l], layer_dims[l - 1])
    return params

def linear_activation_forward(A_prev, W, b, activation):
    # TODO: Z 계산 -> relu 또는 sigmoid
    Z, A = None, None
    return A, (A_prev, W, b, Z)     # cache: backward 때 다시 씀

def L_model_forward(X, params):
    caches = []
    A = X
    L = len(params) // 2
    # TODO: 1 ~ L-1 레이어는 relu, L번째 레이어는 sigmoid. cache를 차곡차곡 append
    AL = None
    return AL, caches

def linear_activation_backward(dA, cache, activation):
    A_prev, W, b, Z = cache
    m = A_prev.shape[1]
    # TODO: dZ = dA * g'(Z) -> dW, db, dA_prev
    dA_prev, dW, db = None, None, None
    assert dW.shape == W.shape and db.shape == b.shape and dA_prev.shape == A_prev.shape
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))   # cross-entropy를 AL로 미분
    # TODO: L번째(sigmoid) 먼저, 그 다음 L-1 ~ 1번째(relu)를 거꾸로
    return grads

def update_parameters(params, grads, lr):
    # TODO
    return params

# 데이터: 원 안쪽이면 1, 바깥이면 0
np.random.seed(0)
m = 500
X = np.random.randn(2, m)
Y = (np.sum(X ** 2, axis=0, keepdims=True) < 1.5).astype(float)

params = initialize_parameters_deep([2, 16, 8, 1])
costs = []
for i in range(3000):
    AL, caches = L_model_forward(X, params)
    cost = -np.mean(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    grads = L_model_backward(AL, Y, caches)
    params = update_parameters(params, grads, 0.3)
    if i % 100 == 0:
        costs.append(cost)
    if i % 1000 == 0:
        print(f"iter {i}: cost {cost:.4f}")
print("train acc:", np.mean((L_model_forward(X, params)[0] > 0.5) == Y))
plt.plot(costs); plt.xlabel("iteration (x100)"); plt.ylabel("cost"); plt.show()
```

**체크포인트**
- cost: `0.695 → 0.028(1000) → 0.011(2000)` 근처로 떨어지고, train acc = 1.0.
- 어떤 assert도 안 터져야 한다. 일부러 `np.sum(dZ, axis=1)`에서 `keepdims=True`를 빼보면 `db.shape == b.shape` assert가 바로 잡아준다.
- `layer_dims`를 `[2, 32, 16, 8, 1]`처럼 바꿔도 코드 수정 없이 돌아가야 제대로 일반화한 것.

<details>
<summary>정답 보기</summary>

```python
def initialize_parameters_deep(layer_dims):
    np.random.seed(1)
    params = {}
    L = len(layer_dims) - 1
    for l in range(1, L + 1):
        params[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])
        params[f"b{l}"] = np.zeros((layer_dims[l], 1))
        assert params[f"W{l}"].shape == (layer_dims[l], layer_dims[l - 1])
    return params

def linear_activation_forward(A_prev, W, b, activation):
    Z = np.dot(W, A_prev) + b
    A = np.maximum(0, Z) if activation == "relu" else sigmoid(Z)
    return A, (A_prev, W, b, Z)

def L_model_forward(X, params):
    caches = []
    A = X
    L = len(params) // 2
    for l in range(1, L):
        A, cache = linear_activation_forward(A, params[f"W{l}"], params[f"b{l}"], "relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, params[f"W{L}"], params[f"b{L}"], "sigmoid")
    caches.append(cache)
    return AL, caches

def linear_activation_backward(dA, cache, activation):
    A_prev, W, b, Z = cache
    m = A_prev.shape[1]
    if activation == "relu":
        dZ = dA * (Z > 0)
    else:
        s = sigmoid(Z)
        dZ = dA * s * (1 - s)
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    assert dW.shape == W.shape and db.shape == b.shape and dA_prev.shape == A_prev.shape
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    dA, grads[f"dW{L}"], grads[f"db{L}"] = linear_activation_backward(dAL, caches[L - 1], "sigmoid")
    for l in reversed(range(1, L)):
        dA, grads[f"dW{l}"], grads[f"db{l}"] = linear_activation_backward(dA, caches[l - 1], "relu")
    return grads

def update_parameters(params, grads, lr):
    L = len(params) // 2
    for l in range(1, L + 1):
        params[f"W{l}"] -= lr * grads[f"dW{l}"]
        params[f"b{l}"] -= lr * grads[f"db{l}"]
    return params
```

</details>
