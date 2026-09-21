# Course 2: Improving Deep Neural Networks — Hyperparameter Tuning, Regularization and Optimization

> DL Specialization Course 1(신경망 기초, forward/backprop)은 이미 요약본으로 봤다고 가정하고, 여기서는 "학습이 잘 안 될 때 뭘 고쳐야 하는가"에 집중하는 Course 2 내용만 정리한다. 3주 분량을 한 번 읽으면 강의를 대체할 수 있을 정도의 밀도로 쓴다.

---

## Week 1: Practical Aspects of Deep Learning

### Train / Dev / Test 분할

전통적인 머신러닝(데이터 수만 개 수준)에서는 70/30 또는 60/20/20 비율이 흔했다. 근데 지금은 데이터가 백만 단위로 커졌기 때문에(빅데이터 시대) dev/test set의 "역할"이 달라진다 — dev set은 여러 모델 중 뭐가 나은지 평가만 하면 되고, test set도 최종 성능이 unbiased하게 나오기만 하면 되는 거라서, 굳이 전체의 20~30%씩 떼어줄 필요가 없다. 100만 개 있으면 dev 1만 개, test 1만 개(각각 1%)만으로도 충분히 신뢰할 수 있는 평가가 가능하다. 그래서 데이터가 아주 클 때는 **98/1/1** 같은 비율도 흔하다.

더 중요한 원칙: **dev set과 test set은 같은 분포(distribution)에서 나와야 한다.** 예를 들어 train은 웹에서 크롤링한 고화질 고양이 사진, dev/test는 유저가 앱으로 찍은 저화질 사진이면 안 된다 — 이러면 모델이 잘못된 target을 겨냥해서 최적화하게 된다. train 분포와 dev/test 분포가 다른 것 자체는(mismatched train/dev distribution) 괜찮을 수 있지만, 그건 나중(Course 3에서 다룸)에 다시 언급된다. 지금 단계에서 기억할 건: **dev = test 분포 통일**.

Test set이 없어도(dev만 있어도) 실무에서는 그냥 진행하는 경우도 있다 — "not having a test set might be okay"라고 Ng가 언급하는데, 다만 그러면 최종 성능에 대한 unbiased한 추정치가 없다는 걸 감수해야 한다.

### Bias / Variance 진단

Course 1에서 이미 배웠겠지만 복습 + 실전 활용:

- **High bias** = train error가 높다 (예: train error 15%) → underfitting
- **High variance** = train error는 낮은데 dev error가 훨씬 높다 (예: train 1%, dev 11%) → overfitting
- 둘 다 나쁠 수도 있다 (train 15%, dev 30% — high bias & high variance)
- 둘 다 좋으면 (train 0.5%, dev 1%) low bias & low variance, 이상적인 상태

이 판단 기준은 **Bayes error(사람 수준 에러, optimal error)**가 거의 0에 가깝다는 전제하에 성립한다. Bayes error가 높은 문제(예: 이미지가 너무 흐려서 사람도 구분 못 함)라면 train error 15%가 그렇게 나쁜 게 아닐 수도 있으니, 항상 "train error를 Bayes error와 비교"해서 bias를 판단해야 한다.

### Basic Recipe for Machine Learning

Andrew Ng이 강조하는 딥러닝 시대의 큰 장점 — **bias와 variance를 (거의) 독립적으로 고칠 수 있다**는 것. 예전 ML에서는 "bias-variance tradeoff"라고 해서 하나 줄이면 하나가 늘었는데, 딥러닝에서는 그렇지 않다. 레시피는 다음 순서로 반복:

1. **High bias인가?** (train set 성능 확인) → Yes면:
   - 더 큰 network (더 많은 hidden unit / layer)
   - 더 오래 학습
   - 다른 NN architecture 시도
   - (bias 안 잡히면 위 반복)
2. **High variance인가?** (dev set 성능 확인) → Yes면:
   - 더 많은 데이터 확보
   - Regularization
   - 다른 NN architecture 시도
3. 둘 다 낮아질 때까지 반복

핵심 통찰: "더 큰 network을 쓰면 bias는 거의 항상 줄일 수 있고, 더 많은 데이터를 쓰면 variance는 거의 항상 줄일 수 있다" — 그리고 이 두 가지 다이얼(network 크기, 데이터 양)이 서로를 크게 해치지 않는다. 이게 딥러닝이 전통 ML보다 유리한 지점 중 하나다 (물론 계산 비용과 데이터 확보 비용은 별개 문제).

### Regularization

High variance(overfitting) 잡는 첫 번째 무기. 데이터를 더 모으기 힘들 때 바로 쓸 수 있는 방법.

**L2 Regularization (weight decay)**

로지스틱 회귀 기준 cost function에 페널티 항 추가:

$$J(w, b) = \frac{1}{m}\sum_{i=1}^m \mathcal{L}(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m}\|w\|_2^2$$

신경망 전체로 확장하면 각 레이어의 weight matrix에 대해 **Frobenius norm**을 사용:

$$J(W^{[1]}, b^{[1]}, \dots, W^{[L]}, b^{[L]}) = \frac{1}{m}\sum_i \mathcal{L}(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m}\sum_{l=1}^{L} \|W^{[l]}\|_F^2$$

$$\|W^{[l]}\|_F^2 = \sum_{i=1}^{n^{[l-1]}}\sum_{j=1}^{n^{[l]}} (W^{[l]}_{ij})^2$$

backprop에서는 원래 $dW^{[l]}$에 $\frac{\lambda}{m}W^{[l]}$가 추가된다:

$$dW^{[l]} = (\text{backprop 계산값}) + \frac{\lambda}{m}W^{[l]}$$

update식을 풀어보면:

$$W^{[l]} := W^{[l]} - \alpha\left(\text{backprop} + \frac{\lambda}{m}W^{[l]}\right) = \left(1 - \frac{\alpha\lambda}{m}\right)W^{[l]} - \alpha \cdot \text{backprop}$$

$W^{[l]}$에 매번 1보다 살짝 작은 값을 곱해주는 모양이라서 **weight decay**라고 부른다.

**왜 regularization이 overfitting을 줄이는가 — 직관 두 가지**

1. $\lambda$를 크게 잡으면 $W$가 0에 가깝게 눌린다 → 여러 hidden unit의 영향력이 사실상 0이 되어 네트워크가 더 "단순한" (거의 로지스틱 회귀에 가까운) 모델처럼 행동 → high variance 쪽에서 high bias 쪽으로 다이얼을 움직이는 셈. $\lambda$를 적절히 잡으면 "just right"에 도달.
2. tanh 같은 activation을 생각해보면, $W$가 작아지면 $z = Wa+b$도 작아지고, $z$가 0 근처일 때 tanh는 거의 **선형(linear)**이다. 모든 레이어가 거의 선형이면 전체 네트워크도 거의 선형 함수가 되어버려서 복잡한 decision boundary를 못 그린다 → overfitting이 잘 안 일어난다.

실전 팁: regularization을 켠 채로 cost $J$를 plot해서 iteration마다 단조 감소하는지 확인할 때는 **regularization term까지 포함한 $J$**를 그려야 한다. 안 그러면 monotonically decrease 안 하는 것처럼 보일 수 있다.

**Dropout Regularization**

각 iteration마다 네트워크의 일부 unit을 랜덤하게 "꺼버리는(제거하는)" 방법. 매 iteration마다 다른, 더 작은 네트워크로 학습하는 셈이라 특정 feature에 지나치게 의존(rely)하지 못하게 만든다 — 이건 L2와 비슷하게 weight를 spread out시키는 효과(shrink weight의 adaptive한 형태).

가장 많이 쓰는 구현: **Inverted Dropout**. layer $l$에 대해:

```python
keep_prob = 0.8  # 남길 확률
d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob
a3 = a3 * d3          # 일부 unit 제거
a3 = a3 / keep_prob   # 기댓값 보정 (inverted dropout의 핵심)
```

`/ keep_prob`으로 나눠주는 이유: 다음 레이어로 넘어가는 $a^{[3]}$의 기댓값을 dropout 적용 전과 맞춰주기 위해서다. 이게 없으면 test 시점에 추가 스케일링 작업이 필요해지는데, inverted dropout은 train 때 미리 보정해두기 때문에 **test 시에는 dropout을 아예 끄고 그대로 forward pass만 하면 된다.** (test에서 dropout을 켜면 예측에 불필요한 노이즈만 추가될 뿐이다.)

레이어마다 다른 `keep_prob`을 줄 수 있다 — 파라미터가 많아 overfitting 위험이 큰 레이어(weight matrix가 큰 레이어)는 `keep_prob`을 낮게, input layer나 작은 레이어는 1(dropout 없음)에 가깝게.

주의점: dropout을 쓰면 cost $J$가 iteration마다 잘 정의되지 않는다(매번 다른 네트워크라서) — 그래서 디버깅할 때는 먼저 `keep_prob=1`로 $J$가 단조감소하는지 확인한 뒤 dropout을 켜는 게 좋다. Dropout은 컴퓨터 비전(CV) 분야에서 특히 자주 쓰인다 — 입력(이미지 픽셀) 차원이 커서 데이터가 항상 부족하게 느껴지기 때문. 데이터가 충분히 많은 다른 도메인에서는 굳이 안 쓰기도 한다.

**Data Augmentation**

새 데이터를 모으는 게 비싸거나 불가능할 때 쓰는 "공짜에 가까운" regularization. 이미지라면 좌우 반전(flip), 랜덤 크롭/줌, 회전, 왜곡(distortion) 등을 적용해서 학습 데이터를 인위적으로 늘린다. 완전히 새로운 독립 데이터만큼 좋지는 않지만 거의 비용 없이 variance를 줄여준다.

**Early Stopping**

train 하면서 dev set error를 같이 관찰하다가, dev error가 더 이상 좋아지지 않고 올라가기 시작하는 지점(iteration)에서 학습을 멈추고 그 시점의 $W$를 사용하는 방법. train error는 계속 내려가는데 dev error는 U자 모양을 그리다가 다시 올라가는 지점을 잡는 것.

**단점 (orthogonalization 관점)**: Ng는 머신러닝 문제를 두 단계로 나눠서 생각하라고 권한다 — (1) cost $J$를 잘 최적화하기(optimize), (2) overfitting 안 되게 하기(regularize/generalize). 이상적으로는 이 두 가지를 **독립적인 도구**로 따로따로 다뤄야 하는데(이게 **orthogonalization**의 의미 — 한 다이얼이 한 가지 효과만 내야 디버깅이 쉬워진다), early stopping은 이 두 단계를 **한 번에 묶어버린다.** 학습을 일찍 멈추는 순간 $J$ 최적화도 멈춰버리기 때문. 그래서 Ng는 개인적으로 L2 regularization(+ 적절한 $\lambda$ 탐색)을 더 선호한다고 언급한다 — 다만 계산 비용 면에서는 early stopping이 "한 번의 학습으로 여러 $\lambda$ 값을 시도한 효과"를 내서 더 저렴하다는 장점도 있다.

### 입력 정규화 (Normalizing Inputs)

두 단계:

1. **평균 빼기 (zero out the mean)**: $x := x - \mu$, $\mu = \frac{1}{m}\sum x^{(i)}$
2. **분산 정규화 (normalize variance)**: $x := x / \sigma^2$ (또는 $\sigma$), $\sigma^2 = \frac{1}{m}\sum (x^{(i)})^2$ (평균을 이미 뺐다면)

주의: train set에서 계산한 $\mu, \sigma^2$를 그대로 test set에도 적용해야 한다 (test set으로 따로 계산하면 안 됨) — train/test가 같은 방식으로 변환되어야 하기 때문.

**왜 정규화하는가 — 그림으로 이해하기**: feature들의 스케일이 서로 다르면 (예: $x_1 \in [1,1000]$, $x_2 \in [0,1]$) cost function의 등고선(contour)이 아주 길쭉하고 찌그러진 타원(elongated bowl) 모양이 된다. 이런 지형에서 gradient descent를 돌리면 경로가 지그재그(zigzag)로 왔다갔다 하면서 느리게 수렴한다 — learning rate를 아주 작게 잡아야 발산 안 하기 때문. 반면 정규화를 하면 등고선이 원(sphere)에 가까워져서 어느 방향으로 가든 곧장 최솟값을 향해 나아갈 수 있다 — 더 큰 learning rate를 써도 안전하고 수렴이 훨씬 빠르다.

### Vanishing / Exploding Gradients

레이어 수 $L$이 아주 깊은 네트워크에서, 각 레이어의 weight $W^{[l]}$가 (activation을 무시하고 단순화하면) 대략 identity보다 조금 크거나 작은 값이라고 하면, 최종 출력은 대략 $W^{[L]}W^{[L-1]}\cdots W^{[1]}x$ 꼴이 된다. 이때:

- $W^{[l]}$의 값이 1보다 살짝 크면 → $L$번 곱해질수록 **exponentially 증가** (exploding)
- $W^{[l]}$의 값이 1보다 살짝 작으면 → $L$번 곱해질수록 **exponentially 감소** (vanishing)

이게 gradient에도 똑같이 적용돼서, 깊은 네트워크일수록 초반 레이어의 gradient가 지수적으로 작아지거나(학습이 거의 안 됨) 커져서(발산) 학습이 매우 어려워진다. 완전히 해결하는 방법은 아니지만 **weight initialization을 신중하게 하는 것**이 이 문제를 크게 완화한다.

### Weight Initialization (He / Xavier)

단일 뉴런 기준으로 $z = w_1x_1 + \cdots + w_nx_n$을 생각하면, $n$(입력 개수)이 클수록 각 $w_i$를 작게 잡아야 $z$가 너무 커지지 않는다. 그래서 $w_i$의 분산을 $1/n$ 스케일로 잡는 아이디어:

- **ReLU 계열 activation**(He initialization, He et al.): $\text{Var}(W^{[l]}) = \dfrac{2}{n^{[l-1]}}$

```python
W_l = np.random.randn(n_l, n_prev) * np.sqrt(2 / n_prev)  # n_l: 현재 층 unit 수, n_prev: 이전 층 unit 수
```

- **tanh activation**(Xavier initialization): $\text{Var}(W^{[l]}) = \dfrac{1}{n^{[l-1]}}$ (또는 Yoshua Bengio 버전은 $\sqrt{\frac{2}{n^{[l-1]} + n^{[l]}}}$)

이 초기화 방법들은 vanishing/exploding gradient 문제를 "없애주진" 못해도 크게 줄여준다 — 학습 시작 시점에 각 레이어 출력의 분산을 비슷한 스케일로 맞춰주기 때문.

### Gradient Checking

backprop 구현이 맞는지 확인하는 디버깅 도구 (실제 학습에는 안 쓰고, 버그 찾을 때만 사용).

**수식**: 모든 파라미터 $W^{[1]}, b^{[1]}, \dots, W^{[L]}, b^{[L]}$을 하나의 큰 벡터 $\theta$로 이어붙이고(reshape+concatenate), 마찬가지로 모든 gradient를 $d\theta$로 이어붙인다. 각 원소 $\theta_i$에 대해 **양방향 수치 미분(two-sided difference)**을 계산:

$$d\theta_{\text{approx}}[i] = \frac{J(\theta_1,\dots,\theta_i+\varepsilon,\dots) - J(\theta_1,\dots,\theta_i-\varepsilon,\dots)}{2\varepsilon}$$

그리고 이걸 backprop이 계산한 $d\theta$와 비교하는데, 단순 차이가 아니라 **상대 오차(relative error)**를 본다:

$$\text{diff} = \frac{\|d\theta_{\text{approx}} - d\theta\|_2}{\|d\theta_{\text{approx}}\|_2 + \|d\theta\|_2}$$

보통 $\varepsilon = 10^{-7}$을 쓰고, diff가 $10^{-7}$보다 작으면 훌륭, $10^{-5}$ 정도면 의심해볼 만, $10^{-3}$ 이상이면 버그 가능성이 높다고 판단한다.

**사용 팁**:
- 학습용으로 쓰지 말 것 (모든 $i$에 대해 forward prop을 2번씩 계산해야 해서 극도로 느림) — 디버깅 때만, 그것도 gradient check 통과하면 바로 꺼야 함.
- diff가 크게 나오면 각 원소별로 어느 $\theta_i$(어느 레이어의 $dW$ 또는 $db$)에서 오차가 큰지 살펴보면 버그 위치를 좁힐 수 있다.
- **regularization term을 쓰고 있다면 gradient checking에도 그 항을 포함**해서 계산해야 한다.
- **dropout과는 gradient checking을 같이 쓸 수 없다** (dropout이 매번 랜덤하게 unit을 꺼서 $J$가 잘 정의되지 않기 때문) — dropout 끄고(keep_prob=1) gradient check 먼저 통과시킨 뒤 dropout을 켜는 순서로 진행.
- 파라미터를 랜덤 초기화한 직후(학습 초반)뿐 아니라 학습이 좀 진행된 후에도 한 번 더 돌려보면 좋다 — 초기값 근처에서만 맞고 학습 중간에 틀어지는 버그도 있을 수 있어서.

---

## Week 2: Optimization Algorithms

### Mini-batch Gradient Descent

$m$이 아주 클 때(예: 5백만) 매 iteration마다 전체 데이터로 gradient를 계산(batch gradient descent)하면 한 스텝 내딛기까지 너무 오래 걸린다. 그래서 학습 데이터를 작은 덩어리(mini-batch)로 나눠서, **한 mini-batch를 처리할 때마다 파라미터를 업데이트**한다. $X^{\{t\}}, Y^{\{t\}}$로 $t$번째 mini-batch를 표기.

- 전체 데이터를 한 번 다 도는 것 = **1 epoch**. batch GD에서는 1 epoch = 1 gradient step이지만, mini-batch GD에서는 1 epoch 안에 (m / batch_size) 번 gradient step이 일어난다 → **훨씬 빠르게 진전(progress)**.
- Cost curve 모양: batch GD는 매끄럽게 단조 감소하지만, mini-batch GD는 **noisy하게 오르내리면서 전반적으로 감소**하는 모양이다 (매 mini-batch마다 난이도가 다른 데이터를 만나기 때문 — 정상적인 현상).

**Batch size 선택**:
- `mini_batch_size = m` → Batch GD. 노이즈는 적지만 iteration당 너무 느림 (작은 데이터셋(대략 m ≤ 2000)에서만 실용적).
- `mini_batch_size = 1` → **Stochastic GD**. 매 샘플마다 업데이트해서 매우 noisy하고, 절대 수렴하지 않고 최솟값 근처에서 계속 진동(oscillate)한다. 또 벡터화(vectorization)의 이점을 완전히 잃어서 속도도 느리다.
- 실전에서는 **그 사이의 적당한 값** (보통 64~512) 사용 — 벡터화 이점을 살리면서도 전체 데이터를 다 보기 전에 진전을 만들 수 있음.
- 관행적으로 batch size는 **2의 거듭제곱(64, 128, 256, 512, ...)**으로 잡는다 — 메모리 접근/캐시 구조상 이 값들일 때 코드가 더 빨리 도는 경향이 있어서.
- 데이터셋(CPU/GPU 메모리)에 mini-batch가 다 들어가는지 확인 필요 — 안 들어가면 성능이 갑자기 나빠짐.

### Exponentially Weighted (Moving) Averages

Momentum/RMSprop/Adam의 기반이 되는 개념. 하루하루의 값 $\theta_t$(예: 일별 기온)에 대해:

$$v_t = \beta v_{t-1} + (1-\beta)\theta_t$$

$v_t$는 대략 **최근 $\dfrac{1}{1-\beta}$일치 데이터의 평균**으로 해석할 수 있다. 예를 들어 $\beta=0.9$면 최근 10일 평균, $\beta=0.98$이면 최근 50일 평균에 가까운 느낌 — $\beta$가 클수록 더 많은 과거를 반영해서 곡선이 더 매끄럽지만(smoother) 실제 변화에 반응이 느려지고(더 lag), $\beta$가 작을수록 노이즈에 민감하지만 반응은 빠르다.

**왜 $1/(1-\beta)$인가**: $(1-\varepsilon)^{1/\varepsilon} \approx 1/e \approx 0.35$라는 근사를 이용하면($\varepsilon = 1-\beta$), $v_t$에서 대략 $1/(1-\beta)$일 지난 시점의 가중치가 원래의 $1/e$ 수준으로 줄어드는 지점이라서, 그 근방까지의 값들이 "실질적으로 평균에 기여하는 범위"라고 볼 수 있다.

**Bias correction**: $v_0 = 0$으로 초기화하기 때문에 초반 몇 스텝은 $v_t$가 실제 값보다 많이 작게 나온다(예: $v_1 = (1-\beta)\theta_1$, $\beta=0.98$이면 $\theta_1$의 2%밖에 안 됨). 초반 추정치를 보정하려면:

$$v_t^{\text{corrected}} = \frac{v_t}{1-\beta^t}$$

$t$가 커지면 $\beta^t \to 0$이라서 보정 효과는 자연히 사라진다. 실전에서는(특히 Adam 구현에서) bias correction을 거의 항상 적용하지만, 곡선을 그저 warm-up 구간 이후만 신경 쓰면 되는 경우엔 생략하기도 한다.

### Gradient Descent with Momentum

핵심 아이디어: gradient의 **exponentially weighted average**를 구해서 그 방향으로 업데이트한다.

$$v_{dW} = \beta v_{dW} + (1-\beta) dW, \qquad v_{db} = \beta v_{db} + (1-\beta) db$$
$$W := W - \alpha v_{dW}, \qquad b := b - \alpha v_{db}$$

(보통 bias correction은 생략해도 됨 — 10번 정도 iteration 지나면 warm-up 문제가 사라지기 때문.) $\beta = 0.9$가 가장 흔한 기본값(최근 10 iteration 평균 정도).

**직관 (그림으로)**: cost function 등고선이 세로로 길쭉한 타원(elongated bowl)이라고 하면, plain GD는 최적점을 향해 가면서도 세로 방향으로 계속 진동(zigzag)한다 — 이 진동 때문에 learning rate를 크게 못 잡는다. Momentum을 쓰면 세로 방향(왔다갔다 하는 방향)의 진동은 평균 내면서 서로 상쇄(cancel out)되고, 가로 방향(계속 같은 방향으로 가는 진전 방향)의 움직임은 누적(accumulate)돼서 커진다. 마치 공을 그릇(bowl) 안에서 굴리는데 $v$가 속도(velocity), $dW$가 가속도(acceleration), $\beta$가 마찰(friction) 역할을 하는 것과 비슷하다 — 그래서 이름이 momentum.

### RMSprop (Root Mean Square Prop)

$$s_{dW} = \beta_2 s_{dW} + (1-\beta_2) (dW)^2 \quad (\text{elementwise})$$
$$s_{db} = \beta_2 s_{db} + (1-\beta_2) (db)^2$$
$$W := W - \alpha \frac{dW}{\sqrt{s_{dW}} + \varepsilon}, \qquad b := b - \alpha \frac{db}{\sqrt{s_{db}} + \varepsilon}$$

($\varepsilon \approx 10^{-8}$은 분모가 0에 너무 가까워지는 걸 막는 안전장치.)

**직관**: momentum과 같은 문제(세로 진동 억제, 가로 진전 유지)를 다른 방식으로 푼다. 진동이 심한 방향(세로, 여기서는 $b$ 방향이라고 가정)은 $dW$/$db$ 값 자체가 크니까 $s$도 커져서 → 업데이트 스텝이 그 방향으로 작아진다(나눠주니까). 반대로 진전 방향(가로, $W$ 방향)은 $dW$가 작으니 $s$도 작고 → 스텝이 상대적으로 커진다. 결과적으로 진동은 줄이고 진전 방향은 살리면서 더 큰 learning rate를 쓸 수 있게 해준다.

### Adam (Adaptive Moment Estimation)

**Momentum + RMSprop을 합친 것.** 실전에서 가장 널리 쓰이고 다양한 문제에서 잘 동작하는 것으로 검증된 optimizer.

$$v_{dW} = \beta_1 v_{dW} + (1-\beta_1) dW, \qquad s_{dW} = \beta_2 s_{dW} + (1-\beta_2)(dW)^2$$
$$v_{dW}^{\text{corrected}} = \frac{v_{dW}}{1-\beta_1^t}, \qquad s_{dW}^{\text{corrected}} = \frac{s_{dW}}{1-\beta_2^t}$$
$$W := W - \alpha \frac{v_{dW}^{\text{corrected}}}{\sqrt{s_{dW}^{\text{corrected}}} + \varepsilon}$$

($b$에 대해서도 동일하게 $v_{db}, s_{db}$를 계산하고 업데이트. 여기서는 bias correction을 반드시 적용.)

**기본 하이퍼파라미터** (거의 항상 이대로 씀, 튜닝 잘 안 함 — $\alpha$만 주로 튜닝):

| 하이퍼파라미터 | 기본값 |
|---|---|
| $\alpha$ | 튜닝 필요 |
| $\beta_1$ | 0.9 |
| $\beta_2$ | 0.999 |
| $\varepsilon$ | $10^{-8}$ |

이름의 유래: **Adaptive Moment estimation** — $\beta_1$은 gradient의 1차 moment(평균), $\beta_2$는 gradient 제곱의 2차 moment(분산)를 추정하는 것에서 왔다.

### Learning Rate Decay

학습 초반에는 큰 스텝으로 빠르게 진전하고, 최솟값 근처에 갈수록 스텝을 줄여서(learning rate를 점점 줄여서) 진동 폭을 좁혀 더 정밀하게 수렴하게 만드는 기법. mini-batch GD처럼 애초에 noisy한 알고리즘에서 특히 유용 — 고정된 $\alpha$면 최솟값 근처에서 계속 큰 범위로 왔다갔다 하지만(never exactly converge), decay를 적용하면 그 범위가 점점 좁아진다.

대표적인 decay 공식들:

$$\alpha = \frac{1}{1+\text{decay\_rate} \times \text{epoch\_num}} \cdot \alpha_0$$
$$\alpha = 0.95^{\text{epoch\_num}} \cdot \alpha_0 \quad \text{(exponential decay)}$$
$$\alpha = \frac{k}{\sqrt{\text{epoch\_num}}} \cdot \alpha_0 \quad \text{또는} \quad \frac{k}{\sqrt{t}}\alpha_0$$

혹은 discrete staircase(몇 epoch마다 절반으로 줄이기)나 수동 decay(사람이 보면서 직접 조정)도 실전에서 쓰인다. Ng는 하이퍼파라미터 튜닝 우선순위에서 learning rate decay는 상대적으로 낮은 우선순위라고 언급한다 — 다른 것들(아래 Week 3) 먼저 튜닝하고 여유 있을 때 시도.

### Local Optima보다 Saddle Point / Plateau가 진짜 문제

저차원(2D) 직관과 달리 고차원 공간(파라미터가 수만~수백만 개)에서는 **모든 방향에서 볼록(convex, 즉 아래로 볼록 — 모든 방향에서 그릇처럼 위로 휘어 올라가는 모양)한 진짜 local optimum(극소점)**은 매우 드물다 — 파라미터가 수만~수백만 개라 "방향"의 개수도 그만큼 많은데, 그 모든 방향에서 동시에 아래로 볼록해야 진짜 극소점이 되기 때문이다. 어느 한 방향에서라도 반대로(위로 볼록하게, 즉 그 방향으로는 아래로 내려갈 수 있게) 휘어져 있을 확률이 높아서, gradient가 0인 지점 대부분은 **saddle point(안장점)**다 (일부 방향은 아래로, 일부 방향은 위로 볼록한 지점 — 말안장 모양).

그래서 딥러닝에서 실제로 학습을 느리게 만드는 주범은 local optima에 갇히는 게 아니라, **plateau(평지)** — gradient가 아주 오랫동안 0에 가까운, 넓고 평평한 구간을 느릿느릿 가로질러야 하는 것이다. 이럴 때 Momentum, RMSprop, Adam 같은 알고리즘이 이 plateau를 더 빨리 빠져나가게 도와준다는 게 이 optimizer들의 실질적인 가치다.

---

## Week 3: Hyperparameter Tuning, Batch Norm, Softmax & Frameworks

### 하이퍼파라미터 튜닝 우선순위

지금까지 나온 하이퍼파라미터가 꽤 많다: $\alpha$, $\beta$ (momentum), $\beta_1,\beta_2,\varepsilon$ (Adam), layer 수, 각 layer의 hidden unit 수, learning rate decay, mini-batch size... Ng이 제시하는 대략적인 우선순위:

1. **1순위 (거의 항상 가장 중요): $\alpha$ (learning rate)**
2. **2순위**: momentum term $\beta$ (보통 0.9 근처), mini-batch size, hidden unit 개수
3. **3순위**: layer 개수, learning rate decay
4. **거의 튜닝 안 함**: Adam의 $\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$ — 이건 "이건 걱정 안 해도 된다"에 해당하는, 사실상 고정값으로 취급.

(참고: 이 우선순위는 절대적이지 않고 문제/도메인에 따라 달라질 수 있다는 언급이 있지만, 대체로 이 순서를 따르면 된다.)

### Grid Search 대신 Random Search

전통적인 ML에서는 하이퍼파라미터 2개를 격자(grid)로 만들어서 각 교차점을 다 시도하는 **grid search**를 많이 썼다. 그런데 딥러닝에서는 **random search**를 권장한다.

**이유**: 하이퍼파라미터마다 결과에 미치는 영향력이 다르다. 예를 들어 $\alpha$는 결과에 큰 영향을 주고 $\varepsilon$은 거의 영향이 없다고 하자. Grid search(5×5 grid라면)로는 $\alpha$ 값을 딱 5가지만 시도해본 셈이 된다 (나머지 축인 $\varepsilon$을 바꿔봐야 사실상 같은 $\alpha$를 5번 반복 시도한 것). 반면 random search로 25개 점을 무작위로 뽑으면 $\alpha$ 값을 25가지 다른 값으로 시도해보게 된다 — 어떤 하이퍼파라미터가 중요한지 미리 알 수 없는 상황에서 훨씬 효율적으로 탐색 공간을 커버한다.

**Coarse to Fine**: 넓은 범위에서 random search로 대충 탐색한 뒤, 성능이 좋았던 하이퍼파라미터들이 몰려있는 좁은 영역을 찾아서 그 영역 안에서 다시 더 촘촘하게(finer) random search를 반복하는 전략. 탐색을 점점 좁혀가며 refine한다.

### 적절한 스케일 선택 (Log Scale)

하이퍼파라미터를 무작위로 뽑을 때 **uniform(선형) 스케일로 뽑으면 안 되는 경우**가 있다.

- **$\alpha$ (learning rate)**: 범위가 $0.0001 \sim 1$이라고 하면, 선형(uniform)하게 뽑으면 뽑힌 값의 90%가 $0.1\sim1$ 구간에 몰리고 $0.0001\sim0.001$ 구간은 거의 안 뽑힌다 — 근데 실제로는 작은 $\alpha$ 구간에서도 세밀한 탐색이 똑같이 중요하다. 그래서 **log scale**로 뽑는다:

```python
r = -4 * np.random.rand()      # r ~ Uniform(-4, 0)
alpha = 10 ** r                 # alpha ~ [10^-4, 10^0], log-uniform
```

- **$\beta$ (exponentially weighted average의 계수, 예: momentum)**: 범위가 $0.9 \sim 0.999$라고 하면, $\beta$가 1에 아주 가까워질수록 $1/(1-\beta)$(평균 내는 기간)가 훨씬 민감하게 커진다 ($\beta=0.9\to0.9005$는 별 차이 없지만 $\beta=0.999\to0.9995$는 평균 기간이 1000일→2000일로 확 뛴다). 그래서 $\beta$가 아니라 **$1-\beta$를 log scale로** 뽑는다:

```python
r = -3 * np.random.rand() - 1   # r ~ Uniform(-3, -1) 이런 식으로 범위 조정
one_minus_beta = 10 ** r
beta = 1 - one_minus_beta
```

즉 $1-\beta \in [0.001, 0.1]$을 log-uniform하게 뽑는 것.

### Pandas vs. Caviar (하이퍼파라미터 튜닝을 실행하는 두 가지 방식)

컴퓨팅 자원 상황에 따라 튜닝하는 방식 자체가 달라진다.

- **Panda 방식 (Babysitting one model)**: 자원이 부족해서(GPU 한두 대) 여러 모델을 동시에 학습시킬 여유가 없을 때, 모델 하나를 오랜 기간 지켜보면서 하이퍼파라미터를 조금씩 수동으로 조정해가는 방식 — 판다가 새끼를 한 마리만 낳고 정성껏 돌보는 것에 비유.
- **Caviar 방식 (Training many models in parallel)**: 자원이 충분할 때, 여러 하이퍼파라미터 조합으로 여러 모델을 동시에 병렬로 학습시킨 뒤 가장 좋은 걸 고르는 방식 — 물고기가 알을 아주 많이 낳고 그중 일부만 살아남길 기대하는 것에 비유.

### Batch Normalization

**아이디어**: input feature를 정규화(Week 1)하면 학습이 빨라진다는 걸 배웠다. 그런데 깊은 네트워크에서는 각 레이어의 출력 $a^{[l]}$(혹은 정확히는 activation 적용 전의 $z^{[l]}$)도 그다음 레이어 입장에서는 "input"이다. 그러니 **각 hidden layer의 $z^{[l]}$도 정규화**하면 그다음 레이어의 학습도 빨라지지 않을까 — 이게 batch norm의 출발점.

**수식** (mini-batch 단위로, layer $l$의 $z^{(i)}$들에 대해):

$$\mu = \frac{1}{m}\sum_i z^{(i)}, \qquad \sigma^2 = \frac{1}{m}\sum_i (z^{(i)}-\mu)^2$$
$$z_{\text{norm}}^{(i)} = \frac{z^{(i)}-\mu}{\sqrt{\sigma^2+\varepsilon}}$$
$$\tilde{z}^{(i)} = \gamma z_{\text{norm}}^{(i)} + \beta$$

여기서 $\gamma, \beta$는 **학습 가능한(learnable) 파라미터**다 (Adam이나 momentum의 $\beta$와는 다른, 완전히 별개의 파라미터니 헷갈리지 말 것). $\gamma, \beta$를 두는 이유: 항상 평균 0, 분산 1로 고정해버리면 hidden unit이 표현할 수 있는 값의 범위가 너무 제한된다 — 예를 들어 sigmoid activation을 쓰는 경우 $z$가 항상 0 근처에만 몰려 있으면 sigmoid의 비선형(nonlinear) 구간을 활용 못 하고 거의 선형 구간만 쓰게 된다. $\gamma, \beta$를 학습시켜서 네트워크가 원하는 평균/분산을 스스로 찾게 해준다. (참고로 $\gamma=\sqrt{\sigma^2+\varepsilon}$, $\beta=\mu$로 학습되면 정규화를 사실상 무효화(identity)할 수도 있다 — 즉 batch norm은 "정규화를 강제"하는 게 아니라 "정규화 여부/정도를 학습 가능하게" 만드는 것.)

네트워크 구조상 $z^{[l]} = W^{[l]}a^{[l-1]}+b^{[l]}$를 계산한 뒤 batch norm을 적용해서 $\tilde{z}^{[l]}$을 만들고, 여기에 activation을 적용해 $a^{[l]} = g^{[l]}(\tilde{z}^{[l]})$을 얻는 순서다. 정규화 과정에서 평균을 빼기 때문에 원래의 $b^{[l]}$은 어차피 상쇄되어 사라진다 — 그래서 **batch norm을 쓰는 레이어에서는 $b^{[l]}$을 생략**해도 되고, 대신 $\beta^{[l]}$이 그 역할(offset/bias)을 대신한다.

**왜 batch norm이 효과가 있는가**

1. Input 정규화와 같은 원리로, 각 레이어 입력 스케일을 비슷하게 맞춰줘서 학습을 빠르게 함.
2. **Covariate shift 완화**: 앞쪽 레이어의 파라미터가 학습 중에 계속 바뀌면, 뒤쪽 레이어 입장에서는 자기 입력의 분포가 계속 변하는 것처럼 보인다(covariate shift) — 마치 흑고양이만 보고 학습했는데 시험 때는 다른 색 고양이가 나오는 것처럼, 학습 도중 "타겟이 계속 움직이는" 문제. Batch norm은 각 레이어 입력의 평균/분산을 ($\gamma,\beta$로 제어되는 값으로) 어느 정도 안정적으로 유지시켜줘서, 뒤쪽 레이어가 앞쪽 레이어의 변화에 덜 흔들리게 만든다 — 각 레이어가 좀 더 독립적으로 학습할 수 있게 되어 전체 학습이 안정되고 빨라진다.
3. **약간의 regularization 부수 효과**: mini-batch 단위로 평균/분산을 계산하다 보니 그 mini-batch에만 존재하는 노이즈가 $\tilde z$에 섞여 들어간다 — dropout과 비슷하게 약한 noise를 주입하는 효과가 있어서 아주 약간의 regularization 효과를 낸다. (다만 이건 부수 효과일 뿐, batch norm을 regularizer로 쓰려고 설계된 건 아니다. mini-batch size를 키우면 이 noise/regularization 효과는 줄어든다.)

**Test 시점 처리**: test할 때는 보통 샘플을 한 개씩 처리하는데, 그러면 mini-batch의 $\mu, \sigma^2$를 계산할 수가 없다. 그래서 학습 중에 각 레이어에서 만난 $\mu, \sigma^2$들의 **exponentially weighted (moving) average(running average)**를 별도로 계속 추적해뒀다가, test 시에는 그 running average 값을 $\mu, \sigma^2$로 사용해서 $z_{\text{norm}}$과 $\tilde z$를 계산한다.

### Softmax Regression

이진 분류를 넘어 **다중 클래스 분류(multi-class classification, $C$개 클래스)**로 확장하는 방법. 마지막 레이어(layer $L$)에서:

$$z^{[L]} \in \mathbb{R}^C, \qquad t = e^{z^{[L]}} \ (\text{elementwise}), \qquad a^{[L]}_i = \frac{t_i}{\sum_{j=1}^C t_j}$$

$a^{[L]}$의 각 원소는 0~1 사이이고 전체 합이 1이 되어 각 클래스에 속할 확률로 해석된다. ($C=2$인 softmax는 사실상 로지스틱 회귀와 동등하다는 점도 알아두면 좋다.)

**Loss function**: 정답 label을 one-hot 벡터 $y$로 표현하고,

$$\mathcal{L}(\hat y, y) = -\sum_{j=1}^C y_j \log \hat y_j$$

정답 클래스가 $k$번째라면 $y_k=1$, 나머지는 0이라서 결국 $\mathcal{L} = -\log \hat y_k$ — 정답 클래스에 매겨진 확률을 최대한 1에 가깝게(log 값을 0에 가깝게) 만드는 방향으로 학습하는 것과 같다. 전체 cost는 이 loss를 $m$개 샘플에 대해 평균낸 것.

Backprop 시 $dz^{[L]} = \hat y - y$ 형태로 아주 깔끔하게 나온다는 것도 로지스틱 회귀의 $dz$ 형태와 닮아있다는 점에서 기억해두면 좋다.

### 딥러닝 프레임워크

밑바닥부터 numpy로 forward/backward를 다 구현하는 건 공부용으로는 좋지만 실전에서는 TensorFlow, PyTorch 같은 프레임워크를 쓴다 — **자동 미분(automatic differentiation)** 덕분에 forward pass만 정의하면 backprop을 프레임워크가 알아서 계산해준다.

TensorFlow에서 `GradientTape`를 쓰는 짧은 예시 (cost $J(w) = w^2 - 10w + 25$를 최소화):

```python
import tensorflow as tf

w = tf.Variable(0, dtype=tf.float32)
optimizer = tf.keras.optimizers.Adam(0.1)

def train_step():
    with tf.GradientTape() as tape:
        cost = w ** 2 - 10 * w + 25   # forward prop만 정의
    trainable_variables = [w]
    grads = tape.gradient(cost, trainable_variables)   # 자동 미분
    optimizer.apply_gradients(zip(grads, trainable_variables))

for _ in range(1000):
    train_step()

print(w.numpy())  # 대략 5.0 (실제 최솟값)
```

`GradientTape` 블록 안에서 일어난 연산들을 기록해뒀다가, `tape.gradient(cost, variables)`를 호출하면 그 연산 그래프를 거슬러 올라가며 자동으로 gradient를 계산해준다 — 우리가 직접 $dw$ 수식을 유도할 필요가 없다는 게 핵심.

프레임워크 선택 기준으로 Ng가 언급하는 것들: (1) 프로그래밍의 용이성(개발+배포), (2) 실행 속도, (3) 커뮤니티/생태계가 진짜로 열려있는지(단순 오픈소스를 넘어 좋은 거버넌스로 계속 유지되는지) — 지금은 이 정도만 참고하고 실제 선택은 상황(팀, 배포 환경)에 따라 하면 된다.

---

## 핵심 요약

- **Train/dev/test**: 데이터가 크면 dev/test 비율을 확 줄여도 된다(98/1/1 등). 단, dev와 test는 반드시 같은 분포에서 나와야 한다.
- **Bias/variance**: train error는 Bayes error와, dev error는 train error와 비교해서 진단한다. 딥러닝에서는 "더 큰 network → bias↓", "더 많은 데이터/regularization → variance↓"가 거의 독립적으로 작동한다.
- **Regularization 도구함**: L2(weight decay), dropout(inverted dropout, test 시엔 꺼야 함), data augmentation, early stopping(orthogonalization 관점에서 비추천이긴 하나 계산 비용은 저렴).
- **입력/레이어 정규화**: 입력 정규화는 cost 등고선을 원형에 가깝게 만들어 GD를 빠르게 한다. 같은 원리를 hidden layer의 $z$에 적용한 것이 batch norm.
- **초기화**: 깊은 네트워크의 vanishing/exploding gradient는 He(ReLU) / Xavier(tanh) 초기화로 완화한다.
- **Gradient checking**: 디버깅 전용, 수식은 two-sided difference + relative error, regularization 포함 / dropout 미포함 상태로 체크.
- **Optimizer 발전 순서**: (mini-batch GD) → Momentum($v$, $\beta\approx0.9$) → RMSprop($s$, $\beta_2$) → Adam(둘 다 + bias correction). Adam 기본값 $\beta_1=0.9,\beta_2=0.999,\varepsilon=10^{-8}$은 거의 안 건드림.
- **고차원에서는 local optima보다 saddle point/plateau가 진짜 문제**이고, 이걸 momentum류 optimizer가 빠르게 통과하도록 돕는다.
- **하이퍼파라미터 튜닝**: 우선순위는 $\alpha$ > ($\beta$, mini-batch size, hidden unit 수) > (layer 수, decay) > Adam의 $\beta_1,\beta_2,\varepsilon$. Grid보다 random search, coarse-to-fine, 필요하면 log scale로 샘플링($\alpha$는 $10^r$, $\beta$는 $1-\beta$를 log scale로).
- **Batch Norm**: $z$를 정규화한 뒤 학습 가능한 $\gamma,\beta$로 스케일/이동. Covariate shift를 줄여 레이어 간 의존성을 낮추고, 부수적으로 약한 regularization 효과도 있다. Test 시엔 학습 중 추적한 running average $\mu,\sigma^2$ 사용.
- **Softmax**: 다중 클래스로의 로지스틱 회귀 확장, loss는 $-\sum y_j\log\hat y_j$, $dz^{[L]}=\hat y - y$.
- **프레임워크**: forward pass만 정의하면 automatic differentiation(예: TF `GradientTape`)이 backprop을 대신 해준다.

## 헷갈리기 쉬운 점

- **Batch norm의 $\beta$ vs. momentum/Adam의 $\beta$**: 이름만 같지 완전히 다른 파라미터다. Batch norm의 $\gamma,\beta$는 각 레이어마다 학습되는 파라미터, momentum/Adam의 $\beta,\beta_1,\beta_2$는 exponentially weighted average의 감쇠율(하이퍼파라미터)이다.
- **L2 regularization vs. weight decay라는 이름**: 둘은 SGD/momentum 기준에서는 수학적으로 동일한 효과를 내지만("decay"라는 이름이 붙은 이유가 업데이트식에서 $W$에 1보다 작은 계수가 곱해지기 때문), Adam 같은 adaptive optimizer에서는 L2 penalty를 gradient에 더하는 것과 진짜 weight decay(업데이트 시 곱해주는 것)가 정확히 같지 않다는 논의도 있다 — 이 강의 수준에서는 "L2 reg = weight decay"로 봐도 무방하지만 완전한 동의어는 아니라는 것만 알아두자.
- **Dropout: train과 test 동작이 다르다**는 걸 자꾸 잊기 쉽다. Inverted dropout을 쓰면 test 코드에는 dropout 관련 코드가 전혀 없어야 한다(끄는 게 아니라 애초에 안 넣는 것). `keep_prob`으로 나눠주는 스케일 보정을 train 때 이미 해놨기 때문.
- **Early stopping이 "나쁜 방법"이라는 게 아니라, orthogonalization 원칙에 안 맞아서 Ng가 개인적으로 덜 선호한다는 것**이다. 실무에서는 계산 비용 때문에 오히려 자주 쓰인다 — "이론적으로 덜 깔끔함"과 "실전에서 못 쓸 방법"은 다른 얘기다.
- **Mini-batch GD의 cost curve가 noisy한 건 버그가 아니라 정상**이다. 매 스텝마다 다른 mini-batch(난이도가 다른 데이터 조합)를 보기 때문에 생기는 자연스러운 현상 — batch GD처럼 완전히 매끄럽게 내려가길 기대하면 안 된다.
- **RMSprop/Adam의 $s$(제곱의 이동평균)와 momentum의 $v$(값 자체의 이동평균)를 헷갈리지 말 것**: $v$는 방향(direction)을 부드럽게 만들고, $s$는 각 축(파라미터)마다 스텝 크기를 적응적으로(adaptive) 조절하는 역할이다. Adam은 이 둘을 같이 쓰는 것.
- **Local optimum과 saddle point는 다르다.** gradient가 0이라고 다 local optimum은 아니다 — 고차원에서는 오히려 saddle point가 훨씬 흔하고, 학습이 느려지는 진짜 원인은 (local optima에 갇히는 게 아니라) plateau를 가로지르는 데 오래 걸리는 것이다.
- **Batch norm이 "regularizer로 설계된 것"은 아니다.** Regularization 효과는 mini-batch 노이즈에서 나오는 부수 효과일 뿐이고, 애초 목적은 학습 속도/안정성(covariate shift 완화)이다. Regularization이 필요하면 여전히 L2나 dropout을 따로 고려해야 한다.
- **Hyperparameter 우선순위에서 Adam의 $\beta_1,\beta_2,\varepsilon$이 "거의 안 건드린다"고 해서 튜닝이 절대 불필요하다는 뜻은 아니다** — 다만 $\alpha$ 대비 ROI가 훨씬 낮으니 시간이 부족하면 여기부터 포기해도 된다는 우선순위 판단이다.

---

## Lab 예제 — 직접 짜보기

> numpy + matplotlib만 쓴다. 뼈대 코드의 `TODO`를 채우고 실행해서 **체크포인트**와 비교해보자. 정답은 접혀있다.

### Lab 1. 가중치 초기화 — zeros / large / He 비교 + symmetry 직접 보기

**목표**: 10층짜리 ReLU 네트워크에 데이터를 흘려보내면서 초기화 방법에 따라 activation 크기가 어떻게 변하는지 본다(vanishing/exploding). 그리고 모든 가중치를 같은 값으로 시작하면 학습 후에도 hidden unit들이 전부 똑같다는 걸(symmetry) 확인한다.

```python
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))
def relu(z): return np.maximum(0, z)

def init_params(layer_dims, method):
    np.random.seed(3)
    p = {}
    for l in range(1, len(layer_dims)):
        shape = (layer_dims[l], layer_dims[l - 1])
        if method == "zeros":
            p[f"W{l}"] = None    # TODO
        elif method == "large":
            p[f"W{l}"] = None    # TODO: randn * 10
        elif method == "he":
            p[f"W{l}"] = None    # TODO: randn * sqrt(2 / n[l-1])
        p[f"b{l}"] = np.zeros((layer_dims[l], 1))
    return p

def activation_stats(p, X):
    A = X
    L = len(p) // 2
    stds = []
    for l in range(1, L):
        A = relu(np.dot(p[f"W{l}"], A) + p[f"b{l}"])
        stds.append(A.std())
    return stds

np.random.seed(0)
X = np.random.randn(100, 1000)
dims = [100] + [100] * 10 + [1]
for method in ["zeros", "large", "he"]:
    stds = activation_stats(init_params(dims, method), X)
    print(f"{method:>6}: 레이어별 activation std = {np.round(stds[::3], 3)}")

# symmetry 직접 확인: hidden unit 4개짜리 2층 네트워크를 학습
def train_2layer(W1, X, Y, iters=200, lr=0.5):
    W2 = np.full((1, W1.shape[0]), 0.5)
    b1, b2 = np.zeros((W1.shape[0], 1)), 0.0
    m = X.shape[1]
    for _ in range(iters):
        A1 = np.tanh(W1 @ X + b1); A2 = sigmoid(W2 @ A1 + b2)
        dZ2 = A2 - Y
        dW2 = dZ2 @ A1.T / m; db2 = dZ2.mean()
        dZ1 = (W2.T @ dZ2) * (1 - A1 ** 2)
        dW1 = dZ1 @ X.T / m; db1 = dZ1.mean(axis=1, keepdims=True)
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
    return W1

Xs = np.random.randn(2, 200); Ys = (Xs[0:1] * Xs[1:2] > 0).astype(float)
W1_sym = train_2layer(np.full((4, 2), 0.3), Xs, Ys)            # 전부 같은 값으로 시작
W1_rand = train_2layer(np.random.randn(4, 2) * 0.5, Xs, Ys)     # 랜덤으로 시작
print("같은 값 초기화 후 학습된 W1:\n", np.round(W1_sym, 3))
print("랜덤 초기화 후 학습된 W1:\n", np.round(W1_rand, 3))
```

**체크포인트**
- `zeros`: std가 전부 0 — 신호가 아예 안 흐른다.
- `large`: std가 `58 → 1.7e7 → 5e12 → 1.6e18`처럼 레이어마다 폭발(exploding).
- `he`: `0.8 → 0.7 → 0.6 → 0.5` 정도로 10층을 지나도 적당한 크기가 유지된다.
- 같은 값 초기화: 200번 학습해도 W1의 4개 행이 **완전히 똑같다** (예: `[-0.111, -0.117]` × 4). hidden unit 4개가 사실상 1개인 셈. 랜덤 초기화는 행마다 다른 값으로 학습된다.

<details>
<summary>정답 보기</summary>

```python
        if method == "zeros":
            p[f"W{l}"] = np.zeros(shape)
        elif method == "large":
            p[f"W{l}"] = np.random.randn(*shape) * 10
        elif method == "he":
            p[f"W{l}"] = np.random.randn(*shape) * np.sqrt(2 / layer_dims[l - 1])
```

</details>

### Lab 2. L2 regularization과 Inverted Dropout

**목표**: 노이즈 낀 작은 데이터(150개)에 큰 네트워크를 학습시켜 일부러 overfitting을 만들고, L2와 dropout을 직접 구현해서 train/dev 성능 차이(variance)가 어떻게 바뀌는지 본다. inverted dropout에서 `/ keep_prob`가 왜 필요한지도 숫자로 확인한다.

```python
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))
def relu(z): return np.maximum(0, z)

def he_init(layer_dims):
    np.random.seed(3)
    return {k: v for l in range(1, len(layer_dims)) for k, v in
            [(f"W{l}", np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])),
             (f"b{l}", np.zeros((layer_dims[l], 1)))]}

def cost_with_l2(AL, Y, params, lambd):
    m = Y.shape[1]
    ce = -np.mean(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    l2 = None   # TODO: (λ / 2m) * 모든 W의 제곱합 (Frobenius norm^2)
    return ce + l2

def dropout_forward(A, keep_prob):
    D = None    # TODO: keep_prob 확률로 True인 mask (A와 같은 shape)
    A = None    # TODO: 끄고, keep_prob로 나눠서 기댓값 유지 (inverted dropout)
    return A, D

def dropout_backward(dA, D, keep_prob):
    return None  # TODO: forward에서 끈 뉴런은 backward에서도 끄기

# inverted dropout 확인: 1로 가득찬 activation에 dropout 적용 후 평균
np.random.seed(1)
A_drop, D = dropout_forward(np.ones((1000, 50)), keep_prob=0.8)
print("살아남은 비율:", D.mean(), " dropout 후 평균(≈1이어야):", A_drop.mean())

def make_data(m, seed):
    rng = np.random.RandomState(seed)
    X = rng.randn(2, m)
    # 원 안쪽=1, 단 15%는 라벨을 뒤집어서 노이즈를 넣음
    Y = ((X[0] ** 2 + X[1] ** 2 < 1.2) ^ (rng.rand(m) < 0.15)).astype(float).reshape(1, m)
    return X, Y

def train(X, Y, lambd=0.0, keep_prob=1.0, iters=15000, lr=0.3):
    np.random.seed(2)
    p = he_init([2, 64, 32, 1])
    m = X.shape[1]
    for i in range(iters):
        Z1 = p["W1"] @ X + p["b1"]; A1 = relu(Z1)
        if keep_prob < 1: A1, D1 = dropout_forward(A1, keep_prob)
        Z2 = p["W2"] @ A1 + p["b2"]; A2 = relu(Z2)
        if keep_prob < 1: A2, D2 = dropout_forward(A2, keep_prob)
        Z3 = p["W3"] @ A2 + p["b3"]; A3 = sigmoid(Z3)
        dZ3 = A3 - Y
        # L2의 gradient: 기존 dW에 (λ/m) * W 를 더하면 끝 (= weight decay)
        dW3 = dZ3 @ A2.T / m + lambd / m * p["W3"]; db3 = dZ3.sum(1, keepdims=True) / m
        dA2 = p["W3"].T @ dZ3
        if keep_prob < 1: dA2 = dropout_backward(dA2, D2, keep_prob)
        dZ2 = dA2 * (Z2 > 0)
        dW2 = dZ2 @ A1.T / m + lambd / m * p["W2"]; db2 = dZ2.sum(1, keepdims=True) / m
        dA1 = p["W2"].T @ dZ2
        if keep_prob < 1: dA1 = dropout_backward(dA1, D1, keep_prob)
        dZ1 = dA1 * (Z1 > 0)
        dW1 = dZ1 @ X.T / m + lambd / m * p["W1"]; db1 = dZ1.sum(1, keepdims=True) / m
        for k, g in zip(["W1", "b1", "W2", "b2", "W3", "b3"], [dW1, db1, dW2, db2, dW3, db3]):
            p[k] -= lr * g
    return p

def predict(p, X):
    A1 = relu(p["W1"] @ X + p["b1"])   # 테스트 때는 dropout 끄기!
    A2 = relu(p["W2"] @ A1 + p["b2"])
    return sigmoid(p["W3"] @ A2 + p["b3"]) > 0.5

X_tr, Y_tr = make_data(150, 0)
X_dev, Y_dev = make_data(1000, 1)
for name, kw in [("no reg", {}), ("L2 λ=0.7", {"lambd": 0.7}), ("dropout 0.7", {"keep_prob": 0.7})]:
    p = train(X_tr, Y_tr, **kw)
    print(f"{name:>12}: train {np.mean(predict(p, X_tr) == Y_tr):.3f} | dev {np.mean(predict(p, X_dev) == Y_dev):.3f}")
```

**체크포인트**
- 살아남은 비율 ≈ 0.8, dropout 후 평균 ≈ 1.0. `/ keep_prob`를 빼면 평균이 0.8로 줄어든다 → 테스트 때(dropout 끔) 스케일이 안 맞게 된다.
- `no reg`: train 0.987 / dev 0.760 — 노이즈까지 외운 전형적인 high variance.
- `L2 λ=0.7`: train 0.827 / dev 0.796 — train은 떨어졌지만 dev가 올라감. 노이즈 15%를 넣었으니 이론상 최선이 0.85 근처라는 걸 생각하면 train 0.83은 "외우기를 포기한" 상태.
- `dropout 0.7`: train 0.887 / dev 0.750 — 이 작은 데이터/작은 네트워크에선 L2만큼 효과가 안 난다. keep_prob, λ를 직접 바꿔보면서 "regularization도 하이퍼파라미터"라는 감을 잡자.

<details>
<summary>정답 보기</summary>

```python
def cost_with_l2(AL, Y, params, lambd):
    m = Y.shape[1]
    ce = -np.mean(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    L = len(params) // 2
    l2 = (lambd / (2 * m)) * sum(np.sum(np.square(params[f"W{l}"])) for l in range(1, L + 1))
    return ce + l2

def dropout_forward(A, keep_prob):
    D = np.random.rand(*A.shape) < keep_prob
    A = A * D
    A = A / keep_prob
    return A, D

def dropout_backward(dA, D, keep_prob):
    return dA * D / keep_prob
```

</details>

### Lab 3. Gradient Checking

**목표**: 양쪽 차분(two-sided difference)으로 gradient를 수치적으로 근사하고, backprop으로 구한 gradient와 비교하는 함수를 만든다. 일부러 버그를 심은 backprop을 잡아내는지 확인한다.

```python
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))

def f_and_grad(theta, x, y):
    # 작은 로지스틱 회귀: theta = [w1, w2, b] 를 한 줄 벡터로 편 것
    w, b = theta[:2].reshape(2, 1), theta[2]
    a = sigmoid(w.T @ x + b)
    J = -np.mean(y * np.log(a) + (1 - y) * np.log(1 - a))
    dz = a - y
    grad = np.concatenate([(x @ dz.T / x.shape[1]).ravel(), [dz.mean()]])
    return J, grad

def gradient_check(f, theta, eps=1e-7):
    _, grad = f(theta)
    grad_approx = np.zeros_like(theta)
    for i in range(len(theta)):
        # TODO: theta[i]만 +eps, -eps 한 복사본 두 개를 만들어
        #       (J(θ+ε) - J(θ-ε)) / 2ε 로 grad_approx[i] 계산
        pass
    diff = None   # TODO: ||grad - approx|| / (||grad|| + ||approx||)
    return diff

np.random.seed(0)
x = np.random.randn(2, 50); y = (np.random.rand(1, 50) > 0.5).astype(float)
theta = np.random.randn(3)
print("정상 backprop diff:", gradient_check(lambda t: f_and_grad(t, x, y), theta))

def buggy(theta):
    J, g = f_and_grad(theta, x, y)
    g[2] *= 2       # db 계산에 버그를 심음
    return J, g
print("버그 backprop diff:", gradient_check(buggy, theta))
```

**체크포인트**
- 정상: diff ≈ `1e-9` (강의 기준 `1e-7` 이하면 OK).
- 버그: diff ≈ `0.13` (`1e-3` 이상이면 거의 확실히 버그). 어느 `i`에서 `grad`와 `grad_approx`가 크게 다른지 찍어보면 버그 위치(`db`)까지 찾을 수 있다.
- 주의: gradient check는 느리니까 디버깅할 때만, dropout은 끄고(`keep_prob=1`), L2를 쓰면 J에 L2 항까지 포함해서 비교.

<details>
<summary>정답 보기</summary>

```python
def gradient_check(f, theta, eps=1e-7):
    _, grad = f(theta)
    grad_approx = np.zeros_like(theta)
    for i in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += eps; tm[i] -= eps
        grad_approx[i] = (f(tp)[0] - f(tm)[0]) / (2 * eps)
    diff = np.linalg.norm(grad - grad_approx) / (np.linalg.norm(grad) + np.linalg.norm(grad_approx))
    return diff
```

</details>

### Lab 4. 지수가중평균 → Momentum / RMSprop / Adam

**목표**: 온도 데이터로 $\beta$에 따른 EWMA의 모양과 bias correction 효과를 보고, 같은 EWMA를 gradient에 적용한 Momentum/RMSprop/Adam을 직접 구현해서 한쪽으로 길쭉한 cost 함수 위에서 궤적을 비교한다. mini-batch 쪼개기도 같이 짜본다.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
days = np.arange(1, 181)
temps = 15 + 10 * np.sin(days / 30) + np.random.randn(180) * 3

def ewma(x, beta, bias_correction=False):
    v, out = 0.0, []
    for t, xt in enumerate(x, start=1):
        v = None    # TODO: v_t = β v_{t-1} + (1-β) θ_t
        out.append(None)   # TODO: bias_correction이면 v / (1 - β^t)
    return np.array(out)

plt.scatter(days, temps, s=5, c="gray", label="raw")
for beta in [0.5, 0.9, 0.98]:
    plt.plot(days, ewma(temps, beta), label=f"β={beta}")
plt.plot(days, ewma(temps, 0.98, True), "--", label="β=0.98 + bias correction")
plt.legend(); plt.title("Exponentially weighted averages"); plt.show()

def grad(p):   # f = 0.5*(x^2 + 25*y^2): y 방향으로 25배 가파른 그릇
    return np.array([p[0], 25 * p[1]])

def run(optimizer, lr, steps=60, beta1=0.9, beta2=0.999, eps=1e-8):
    p = np.array([-9.0, 2.0])
    v, s = np.zeros(2), np.zeros(2)
    path = [p.copy()]
    for t in range(1, steps + 1):
        g = grad(p)
        if optimizer == "gd":
            p = p - lr * g
        elif optimizer == "momentum":
            pass   # TODO: v = EWMA(g), p -= lr * v
        elif optimizer == "rmsprop":
            pass   # TODO: s = EWMA(g^2), p -= lr * g / (sqrt(s) + eps)
        elif optimizer == "adam":
            pass   # TODO: v, s 둘 다 + bias correction(v_hat, s_hat)
        path.append(p.copy())
    return np.array(path)

xx, yy = np.meshgrid(np.linspace(-10, 2, 200), np.linspace(-3, 3, 200))
plt.contour(xx, yy, 0.5 * (xx ** 2 + 25 * yy ** 2), levels=30, alpha=0.4)
for name, lr in [("gd", 0.075), ("momentum", 0.075), ("rmsprop", 0.3), ("adam", 0.5)]:
    path = run(name, lr)
    plt.plot(path[:, 0], path[:, 1], ".-", label=name)
    print(f"{name:>8}: 60 step 후 위치 {np.round(path[-1], 4)}")
plt.legend(); plt.title("Optimizer trajectories"); plt.show()

def random_mini_batches(X, Y, batch_size=64, seed=0):
    rng = np.random.RandomState(seed)
    m = X.shape[1]
    # TODO: 열(샘플) 순서를 섞고(X, Y 같은 순서로!), batch_size씩 잘라 리스트로 반환
    return []

batches = random_mini_batches(np.random.randn(3, 1000), np.random.randn(1, 1000))
print("mini-batch 개수:", len(batches), " 마지막 배치 크기:", batches[-1][0].shape)
```

**체크포인트**
- β=0.5는 노이즈를 거의 그대로 따라가고, β=0.98은 매끄럽지만 오른쪽으로 밀린다(≈ 최근 $\frac{1}{1-\beta}=50$일 평균이라 반응이 늦음). β=0.98 bias correction 없는 선은 초반에 0 근처에서 출발 — correction을 넣으면 첫날부터 실제 온도 근처에서 시작.
- 궤적: gd는 lr을 조금만 키워도(0.08 이상) y 방향으로 발산한다(`2/25 = 0.08`이 한계). momentum은 y 방향 진동이 상쇄돼서 부드럽게, RMSprop/Adam은 방향별로 step 크기가 정규화돼서 거의 대각선으로 최소점(0,0)을 향한다.
- mini-batch: 개수 16, 마지막 배치 `(3, 40)` (1000 = 64×15 + 40).

<details>
<summary>정답 보기</summary>

```python
def ewma(x, beta, bias_correction=False):
    v, out = 0.0, []
    for t, xt in enumerate(x, start=1):
        v = beta * v + (1 - beta) * xt
        out.append(v / (1 - beta ** t) if bias_correction else v)
    return np.array(out)

# run() 안
        elif optimizer == "momentum":
            v = beta1 * v + (1 - beta1) * g
            p = p - lr * v
        elif optimizer == "rmsprop":
            s = beta2 * s + (1 - beta2) * g ** 2
            p = p - lr * g / (np.sqrt(s) + eps)
        elif optimizer == "adam":
            v = beta1 * v + (1 - beta1) * g
            s = beta2 * s + (1 - beta2) * g ** 2
            v_hat = v / (1 - beta1 ** t)
            s_hat = s / (1 - beta2 ** t)
            p = p - lr * v_hat / (np.sqrt(s_hat) + eps)

def random_mini_batches(X, Y, batch_size=64, seed=0):
    rng = np.random.RandomState(seed)
    m = X.shape[1]
    perm = rng.permutation(m)
    X_sh, Y_sh = X[:, perm], Y[:, perm]
    return [(X_sh[:, k:k + batch_size], Y_sh[:, k:k + batch_size]) for k in range(0, m, batch_size)]
```

</details>

### Lab 5. Batch Norm, Softmax, 그리고 log scale random search

**목표**: batch norm forward를 구현해서 unit마다 스케일이 제각각인 $Z$가 평균 0/분산 1로 정리되는 것, 그리고 $\gamma, \beta$로 다시 원하는 분포로 옮길 수 있다는 걸 본다. test 때 쓸 running average도 계산해본다. 이어서 수치적으로 안정한 softmax와 learning rate를 log scale로 뽑는 random search까지.

```python
import numpy as np

def batchnorm_forward(Z, gamma, beta, eps=1e-8):
    # Z: (n_units, m) — 미니배치 안에서 unit(행)별로 정규화
    mu = None       # TODO: (n_units, 1)
    var = None      # TODO: (n_units, 1)
    Z_norm = None   # TODO
    Z_tilde = None  # TODO: γ * Z_norm + β
    return Z_tilde, mu, var

np.random.seed(0)
scale, shift = np.array([[1], [10], [100], [0.1]]), np.array([[5], [-3], [50], [0]])
Z = np.random.randn(4, 256) * scale + shift
Zt, mu, var = batchnorm_forward(Z, np.ones((4, 1)), np.zeros((4, 1)))
print("BN 전 unit별 평균/표준편차:", np.round(Z.mean(1), 2), np.round(Z.std(1), 2))
print("BN 후 unit별 평균/표준편차:", np.round(Zt.mean(1), 2), np.round(Zt.std(1), 2))
Zt2, _, _ = batchnorm_forward(Z, np.full((4, 1), 2.0), np.full((4, 1), 7.0))
print("γ=2, β=7 이면:", np.round(Zt2.mean(1), 2), np.round(Zt2.std(1), 2))

# test 때는 미니배치가 없으니 학습 중 μ, σ²의 지수가중평균을 저장해뒀다가 씀
running_mu = np.zeros((4, 1))
for _ in range(100):
    Zb = np.random.randn(4, 64) * scale + shift
    _, mu_b, var_b = batchnorm_forward(Zb, 1, 0)
    running_mu = None   # TODO: 0.9 * running_mu + 0.1 * mu_b
print("test 때 쓸 running mean:", np.round(running_mu.ravel(), 2))

def softmax(Z):
    # TODO: 열마다 max를 빼고(overflow 방지) exp -> 열 합으로 나누기
    return None

def softmax_cost(A, Y):
    return -np.mean(np.sum(Y * np.log(A + 1e-12), axis=0))

Z = np.array([[5.0, 1000.0], [2.0, 999.0], [-1.0, 0.0], [3.0, 998.0]])   # 두 번째 열은 그냥 exp하면 overflow
A = softmax(Z)
print("softmax:\n", np.round(A, 3), "\n열 합:", A.sum(axis=0))
Y = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
print("cost:", softmax_cost(A, Y), " dZ = A - Y:\n", np.round(A - Y, 3))

# learning rate를 [1e-4, 1]에서 뽑을 때: log scale vs uniform
np.random.seed(0)
alphas = None    # TODO: r ~ U[-4, 0], α = 10^r
uniform = np.random.uniform(0.0001, 1, 10000)
for lo, hi in [(1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1)]:
    print(f"[{lo:g}, {hi:g}) 구간 비율 - log scale: {np.mean((alphas >= lo) & (alphas < hi)):.2f}, "
          f"uniform: {np.mean((uniform >= lo) & (uniform < hi)):.2f}")
betas = None     # TODO: momentum β ∈ [0.9, 0.999] — 1-β를 log scale로 (10^U[-3,-1])
print("momentum β 후보:", np.round(betas, 4))
```

**체크포인트**
- BN 후 모든 unit이 평균 0, 표준편차 1. `γ=2, β=7`이면 평균 7, 표준편차 2 — 즉 BN은 분포를 "강제로 고정"하는 게 아니라 네트워크가 $\gamma,\beta$로 원하는 분포를 **학습**하게 해준다.
- running mean ≈ `[5, -3.7, 50.5, 0]` — 실제 shift `[5, -3, 50, 0]`에 가까워진다.
- softmax 열 합이 정확히 1, 두 번째 열(1000, 999, ...)도 nan 없이 `[0.665, 0.245, 0, 0.09]`. max 빼는 걸 빼먹으면 `exp(1000)` = inf → nan.
- log scale은 네 구간에 각각 25%씩, uniform은 89%가 `[0.1, 1)`에 몰린다. 작은 learning rate 쪽을 거의 탐색 못 하는 것.

<details>
<summary>정답 보기</summary>

```python
def batchnorm_forward(Z, gamma, beta, eps=1e-8):
    mu = np.mean(Z, axis=1, keepdims=True)
    var = np.var(Z, axis=1, keepdims=True)
    Z_norm = (Z - mu) / np.sqrt(var + eps)
    Z_tilde = gamma * Z_norm + beta
    return Z_tilde, mu, var

    running_mu = 0.9 * running_mu + 0.1 * mu_b

def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)   # 결과는 동일, overflow만 방지
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

r = -4 * np.random.rand(10000)
alphas = 10 ** r
betas = 1 - 10 ** np.random.uniform(-3, -1, 5)
```

</details>
