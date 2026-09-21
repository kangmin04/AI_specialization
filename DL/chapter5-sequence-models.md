# Chapter 5. Sequence Models (Deep Learning Specialization, Course 5)

> ML Specialization 다 듣고 신경망/역전파/최적화는 이미 아는 상태에서, Deep Learning Specialization의 5번째 코스(Sequence Models)만 압축해서 훑는 노트다. Andrew Ng 강의 특유의 "직관 먼저, 수식은 나중" 순서를 그대로 따라간다. 세부 수식은 시험 보는 게 아니니 다 외울 필요 없고, "이게 왜 필요한지"랑 "어디에 쓰는지"만 확실히 잡고 가면 된다.

---

## Week 1. Recurrent Neural Networks (RNN)

### 왜 시퀀스 모델이 필요한가

지금까지 다룬 데이터(이미지, 정형 데이터)는 입력 하나에 출력 하나가 대응되는 구조였다. 근데 아래처럼 **순서(order)가 의미를 가지는 데이터**들이 있다:

- Speech recognition: 오디오 시퀀스 → 텍스트 시퀀스
- Music generation: 출력 자체가 시퀀스
- Sentiment classification: 문장(시퀀스) → 별점(고정 출력)
- DNA sequence analysis: 염기서열(A,C,G,T) 시퀀스
- Machine translation: 시퀀스 → 시퀀스 (길이도 다를 수 있음)
- Video activity recognition: 프레임 시퀀스 → 라벨
- Named entity recognition (NER): 단어 시퀀스 → 각 단어별 라벨

이런 문제들은 입력·출력 중 하나 또는 둘 다 시퀀스라는 공통점이 있고, 이걸 다루려고 만든 게 RNN이다.

### 표기법(notation)

예시: NER 문제, "Harry Potter and Hermione Granger invented a new spell."

- $x^{<t>}$: 입력 시퀀스의 $t$번째 원소 (단어)
- $y^{<t>}$: 출력 시퀀스의 $t$번째 원소
- $T_x$: 입력 시퀀스 길이, $T_y$: 출력 시퀀스 길이 (다를 수 있음)
- $x^{(i)<t>}$: $i$번째 훈련 샘플의 $t$번째 원소, $T_x^{(i)}$도 샘플마다 다를 수 있음

**단어를 숫자로 표현하기**: vocabulary(단어 사전, 보통 1만~5만 단어)를 만들고 각 단어를 **one-hot vector**로 표현한다. 사전에 없는 단어는 `<UNK>` 토큰으로 처리한다. one-hot이라 벡터 크기 = 사전 크기라서 굉장히 sparse한데, Week2의 word embedding이 이 한계를 풀어준다.

### 왜 표준 신경망(standard NN)이 안 되는가

1. 입력/출력 길이가 샘플마다 다르다 (표준 NN은 고정 크기 입력을 가정)
2. 문장의 서로 다른 위치에서 학습한 특징(feature)을 공유하지 못한다. 예를 들어 "Harry"가 문장 앞에서 이름이라는 걸 학습해도, 그 지식이 문장 뒤쪽 위치에 자동으로 전달되지 않는다 (CNN이 이미지 위치에 상관없이 필터를 공유하는 것과 비슷한 문제).
3. one-hot 입력을 다 이어붙이면 입력 차원이 너무 커진다.

### RNN forward propagation

핵심 아이디어: 각 타임스텝마다 **같은 파라미터를 공유**하면서, 이전 타임스텝의 정보를 다음 스텝으로 넘겨주는 "활성값(activation)"을 유지한다. 그림으로 말하면, 왼쪽에서 오른쪽으로 하나의 셀(cell)이 반복되면서 화살표 하나가 옆으로(activation), 화살표 하나가 위로(output) 나가는 구조다.

$$a^{<0>} = \vec{0}$$
$$a^{<t>} = g_1(W_{aa} a^{<t-1>} + W_{ax} x^{<t>} + b_a)$$
$$\hat{y}^{<t>} = g_2(W_{ya} a^{<t>} + b_y)$$

$g_1$은 보통 $\tanh$ (또는 ReLU), $g_2$는 문제에 따라 sigmoid(binary) 또는 softmax(multiclass).

**표기 단순화**: $W_{aa}$와 $W_{ax}$를 옆으로 이어붙여(concatenate) 하나의 $W_a$로 쓴다.

$$W_a = [W_{aa} \mid W_{ax}], \quad [a^{<t-1>}, x^{<t>}] = \begin{bmatrix} a^{<t-1>} \\ x^{<t>} \end{bmatrix}$$
$$a^{<t>} = g_1(W_a [a^{<t-1>}, x^{<t>}] + b_a)$$

이렇게 묶어두면 나중에 GRU/LSTM 수식 볼 때 표기가 훨씬 깔끔해진다.

**한계**: 이 기본 RNN은 시퀀스의 앞쪽 정보만 보고 뒤쪽 정보는 못 본다(단방향). "Teddy bears are on sale" vs "Teddy Roosevelt was a president"에서 "Teddy"가 이름인지 아닌지는 뒤에 나오는 단어를 봐야 아는데, 기본 RNN 구조로는 안 된다 → **bidirectional RNN**으로 해결(아래에서 다룸).

### Backpropagation through time (BPTT)

이름 그대로, 시간 축을 따라 펼쳐놓은(unroll) 그래프에서 오른쪽 끝 loss부터 왼쪽으로 gradient를 흘려보내는 것이다. 각 타임스텝 loss를 다 더한 게 전체 loss:

$$\mathcal{L} = \sum_{t=1}^{T_y} \mathcal{L}^{<t>}(\hat{y}^{<t>}, y^{<t>})$$

직접 손으로 미분 유도하는 건 프레임워크(TensorFlow/Keras/PyTorch)가 다 해주니까 세부 미분식은 안 외워도 된다. 중요한 건 **"시간 방향으로 gradient가 여러 번 곱해지면서 흘러간다"**는 그림이고, 이게 뒤에 나올 vanishing gradient 문제의 원인이 된다.

### RNN 구조 유형

풀고자 하는 문제에 따라 입출력 개수가 다르게 설계된다:

| 유형 | 입력:출력 | 예시 |
|---|---|---|
| One-to-one | 1:1 | 그냥 표준 NN |
| One-to-many | 1:다 | Music generation (장르 하나 → 음악 시퀀스) |
| Many-to-one | 다:1 | Sentiment classification (문장 → 별점) |
| Many-to-many ($T_x = T_y$) | 다:다, 길이 같음 | NER (단어별로 라벨) |
| Many-to-many ($T_x \ne T_y$) | 다:다, 길이 다름 | Machine translation (encoder-decoder 구조, Week3에서 자세히) |

### Language model과 sequence generation

Language model이 하는 일: 어떤 문장이 나올 확률 $P(y^{<1>}, y^{<2>}, ..., y^{<T_y>})$을 추정하는 것. 음성인식에서 "The apple and pair salad"보다 "The apple and pear salad"가 훨씬 그럴듯하다고 판단하는 게 language model의 역할이다.

학습 방법: 큰 텍스트 corpus를 토큰화(tokenize)하고, RNN이 각 타임스텝에서 "지금까지 본 단어들이 주어졌을 때 다음 단어가 뭘지" 예측하도록 훈련한다. 즉 $t$번째 스텝의 입력 $x^{<t>}$는 실제 정답 $y^{<t-1>}$을 그대로 넣어주고(**teacher forcing**), softmax로 다음 단어 확률분포를 뽑는다.

**Sequence sampling(새 문장 생성)**: 학습이 끝난 모델에서, 첫 타임스텝엔 $x^{<1>} = \vec{0}$을 넣고 softmax 출력에서 확률적으로(random sampling) 단어 하나를 뽑는다. 그 뽑힌 단어를 다음 스텝 입력으로 넣고 반복 → `<EOS>` 토큰이 나오거나 정해진 길이에 도달하면 멈춘다.

**Character-level language model**: vocabulary를 단어 대신 알파벳/문자 단위로 잡는 방식. 장점은 `<UNK>` 문제가 없다는 것(어떤 단어든 문자 조합으로 표현 가능), 단점은 시퀀스가 훨씬 길어져서 앞쪽 문맥(long-range dependency)을 잡기 어렵고 학습/계산 비용이 크다. 실무에서는 word-level이 기본이고, 특수한 경우(고유명사 많은 도메인 등)에 character-level을 섞어 쓴다.

### Vanishing / Exploding gradients

시퀀스가 길어지면 BPTT 과정에서 gradient가 여러 번 곱해지는데,

- 1보다 작은 값들이 계속 곱해지면 → **vanishing gradient**: gradient가 0에 수렴해서 먼 과거 타임스텝의 영향을 학습하기 어려워진다. "The cat, which already ate ..., **was** full" 처럼 주어-동사 수 일치(long-term dependency)를 기본 RNN이 잘 못 잡는 이유가 이거다. → 해결책이 **GRU, LSTM**(아래).
- 1보다 큰 값들이 곱해지면 → **exploding gradient**: gradient가 NaN까지 튈 수 있다. 이건 상대적으로 다루기 쉬운데, **gradient clipping**(gradient의 norm이 threshold를 넘으면 threshold 크기로 잘라내는 것)만 적용해도 대부분 해결된다.

```python
# gradient clipping 개념 (의사코드)
if np.linalg.norm(gradient) > max_norm:
    gradient = gradient * max_norm / np.linalg.norm(gradient)
```

### GRU (Gated Recurrent Unit)

Vanishing gradient를 완화하려고, "이 정보를 계속 기억할지 말지"를 게이트(gate)로 결정하게 만든 구조다. 직관: 문장 "The cat, which already ate a lot of food, **was** full"에서 "cat"이 단수라는 정보를 "was"가 나올 때까지 memory cell에 담아두고 싶다 — 그 담아두는 스위치 역할을 하는 게 gate다.

Simplified GRU (직관 먼저):

$$\tilde{c}^{<t>} = \tanh(W_c [c^{<t-1>}, x^{<t>}] + b_c) \quad \text{(새로 후보로 넣을 값)}$$
$$\Gamma_u = \sigma(W_u [c^{<t-1>}, x^{<t>}] + b_u) \quad \text{(update gate, 0~1 사이 값)}$$
$$c^{<t>} = \Gamma_u \odot \tilde{c}^{<t>} + (1 - \Gamma_u) \odot c^{<t-1>}$$

$\Gamma_u$가 0에 가까우면 $c^{<t>} \approx c^{<t-1>}$이라서 옛날 정보를 그대로 유지 — 이게 vanishing gradient를 막아주는 핵심 트릭이다(gate가 닫혀있으면 gradient가 그대로 통과).

Full GRU는 여기에 **reset gate** $\Gamma_r$이 추가된다 — "새 후보값을 계산할 때 과거 정보를 얼마나 반영할지"를 조절:

$$\Gamma_r = \sigma(W_r [c^{<t-1>}, x^{<t>}] + b_r)$$
$$\tilde{c}^{<t>} = \tanh(W_c [\Gamma_r \odot c^{<t-1>}, x^{<t>}] + b_c)$$

기본 RNN에서는 $a^{<t>} = c^{<t>}$로 봐도 된다(GRU에선 activation과 memory cell이 같음, LSTM은 다르다).

### LSTM (Long Short-Term Memory)

GRU보다 먼저 나온, 더 강력하고 일반적인 구조. Gate가 3개(update, forget, output)고, memory cell $c^{<t>}$와 activation $a^{<t>}$을 분리해서 관리한다.

$$\tilde{c}^{<t>} = \tanh(W_c [a^{<t-1>}, x^{<t>}] + b_c)$$
$$\Gamma_u = \sigma(W_u [a^{<t-1>}, x^{<t>}] + b_u) \quad \text{(update gate)}$$
$$\Gamma_f = \sigma(W_f [a^{<t-1>}, x^{<t>}] + b_f) \quad \text{(forget gate)}$$
$$\Gamma_o = \sigma(W_o [a^{<t-1>}, x^{<t>}] + b_o) \quad \text{(output gate)}$$
$$c^{<t>} = \Gamma_u \odot \tilde{c}^{<t>} + \Gamma_f \odot c^{<t-1>}$$
$$a^{<t>} = \Gamma_o \odot \tanh(c^{<t>})$$

GRU는 update/forget을 하나의 게이트($\Gamma_u$, $1-\Gamma_u$)로 묶어 파라미터가 적고 계산이 가볍다(큰 모델 쌓기 유리). LSTM은 게이트가 독립적이라 더 유연하고 역사적으로 default 선택지였다. 실무 기준: 요즘은 대부분 GRU/LSTM 둘 다 시도해보고 데이터로 고르거나, 아예 Transformer(Week4) 계열로 넘어가는 경우가 많다 — 정답은 없다.

### Bidirectional RNN (BRNN)

한 방향(과거→미래)만 보는 한계를 풀려고, **정방향 RNN**과 **역방향 RNN**을 둘 다 돌려서 매 타임스텝 출력에서 두 방향의 activation을 합친다(concat).

$$\hat{y}^{<t>} = g(W_y [\overrightarrow{a}^{<t>}, \overleftarrow{a}^{<t>}] + b_y)$$

NER처럼 문장 전체를 다 보고 판단해도 되는 문제(offline)에는 아주 잘 맞는다. 단점은 예측하려면 **전체 시퀀스가 다 있어야** 한다는 것 — 그래서 실시간 음성인식처럼 스트리밍이 필요한 곳엔 그대로 못 쓴다(더 복잡한 방식 필요).

### Deep RNN

RNN도 layer를 여러 층 쌓을 수 있다. 표기: $a^{[l]<t>}$ = $l$번째 layer, $t$번째 타임스텝의 activation. 다만 시퀀스 방향으로도 이미 "깊은" 계산이라 3층 정도만 쌓아도 파라미터가 많아지고 학습이 무거워진다 — 그래서 보통 RNN layer를 2~3개 쌓은 뒤, 맨 위에 (시퀀스 축이 없는) 일반 fully-connected layer를 얹는 식으로 마무리하는 경우가 많다.

---

## Week 2. Word Embeddings

### One-hot 표현의 한계

Week1에서 쓴 one-hot vector는 단어 간 **유사도 개념이 전혀 없다** — "orange"와 "apple"의 one-hot 벡터끼리 내적(inner product)을 해도 0이라, 모델 입장에서 두 단어가 비슷한 단어라는 걸 전혀 알 수 없다. "I want a glass of orange ___"를 학습해도 그 지식이 "I want a glass of apple ___"로 전혀 일반화가 안 된다.

### Featurized representation → Word embedding

해결책: 각 단어를 **여러 개의 의미 특징(feature) 값을 가진 dense vector**로 표현하자는 아이디어. 예를 들어 Gender, Royal, Age, Food 같은 축(실제로는 학습되는 추상적인 축이라 사람이 이름 붙인 게 아님)에서 "King"과 "Queen"은 Royal 축에서 비슷하고 Gender 축에서 다르다, "Apple"과 "Orange"는 Food 축에서 비슷하다 — 이런 식으로 저차원(보통 50~300차원) 실수 벡터에 의미를 녹여낸 게 **word embedding**이다.

t-SNE 같은 차원축소로 이 임베딩들을 2D에 뿌려보면 비슷한 의미의 단어들이 실제로 뭉쳐있는 걸 볼 수 있다.

### Word embedding과 transfer learning

절차:
1. 아주 큰 텍스트 corpus(웹 텍스트 등, 1B~100B 단어급)에서 word embedding을 미리 학습(pre-train)한다.
2. 이 임베딩을 상대적으로 작은 labeled 데이터셋(예: NER)에 transfer한다.
3. (선택) 작은 데이터셋 태스크에서 임베딩을 추가로 fine-tuning한다 — 단, task용 데이터셋이 충분히 클 때만.

이건 face recognition의 encoding(Siamese network 등)이랑 개념이 비슷하다 — 미리 학습된 표현을 가져다 쓴다는 점에서. 큰 데이터로 학습한 일반적인 언어 지식을 작은 task로 옮기는 전형적인 transfer learning 사례다.

### 유추(analogy) 문제와 cosine similarity

Word embedding의 유명한 성질: $e_{man} - e_{woman} \approx e_{king} - e_{queen}$. 즉 "man : woman = king : ?"을 풀 때, $e_{king} - e_{man} + e_{woman}$과 가장 유사한 벡터를 embedding 공간에서 찾으면 된다.

유사도 측정은 보통 **cosine similarity**를 쓴다:

$$\text{sim}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$$

유클리드 거리보다 cosine similarity를 쓰는 이유는 벡터의 "방향"(의미)에 집중하고 크기(magnitude)에는 덜 민감하기 때문.

### Embedding matrix

Vocabulary 크기가 $V$, 임베딩 차원이 $d$라면 embedding matrix $E$는 $(d, V)$ 크기다. 단어 $w$의 one-hot vector $o_w$에 대해:

$$e_w = E \cdot o_w$$

실제 구현에서는 one-hot과의 행렬곱을 그대로 하지 않고, $E$에서 해당 열(column)을 그냥 **lookup(인덱싱)** 한다 — Keras의 `Embedding` layer가 내부적으로 이렇게 동작한다.

```python
# Keras 예시
from tensorflow.keras.layers import Embedding
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
```

### 학습 방법들

**1) Neural language model 방식**: 앞의 몇 단어(context)로 다음 단어를 예측하는 신경망을 학습하면서 그 과정에서 $E$가 같이 학습된다(word2vec 이전의 초기 접근, Bengio et al.).

**2) Word2Vec — Skip-gram**: 문장에서 하나의 단어를 context로 잡고, 그 주변(window) 단어들을 target으로 예측하도록 학습한다. 예: "I want a glass of orange juice to go along with my cereal"에서 context="orange"로 target="juice", "glass" 등을 예측.

$$P(t \mid c) = \frac{e^{\theta_t^T e_c}}{\sum_{j=1}^{V} e^{\theta_j^T e_c}} \quad \text{(softmax)}$$

**문제점**: vocabulary $V$가 수만~수십만이면 이 softmax 분모 계산이 매 스텝 너무 비싸다. 해결책으로 **hierarchical softmax**(이진 트리 구조로 $O(\log V)$까지 줄임)를 쓰기도 하지만 구현이 복잡하고, 실무에서 더 널리 쓰인 건 아래 negative sampling이다.

**3) Negative sampling**: skip-gram의 무거운 softmax를 여러 개의 binary classification 문제로 바꾼 것. (context, target) 쌍이 진짜 문장에서 나온 "positive" 쌍인지 아닌지를 예측하는 로지스틱 회귀 문제로 재정의한다. 매 스텝마다 positive 쌍 1개 + 랜덤하게 뽑은 negative 쌍 $k$개(작은 데이터셋은 $k=5\sim20$, 큰 데이터셋은 $k=2\sim5$)만 가지고 학습하기 때문에 훨씬 가볍다. Negative 단어를 뽑는 분포는 균등분포도, 코퍼스 등장 빈도 그대로도 아니고, 경험적으로 좋은 성능을 낸다고 알려진 $\frac{f(w)^{3/4}}{\sum_j f(w_j)^{3/4}}$ 같은 heuristic 분포를 쓴다.

**4) GloVe (Global Vectors)**: 개념만 — 코퍼스 전체에서 단어 $i$, $j$가 함께 등장한 횟수 $X_{ij}$(co-occurrence count)를 미리 계산해두고, $\theta_i^T e_j$가 $\log X_{ij}$를 잘 근사하도록 학습하는 방식이다. Skip-gram/negative sampling처럼 슬라이딩 윈도우로 매번 샘플링하지 않고 co-occurrence 통계를 직접 활용한다는 점이 다르다. Word2Vec보다 간단하면서 비슷하게 잘 작동해서 많이 쓰였다.

(참고: 요즘 실무에서는 word2vec/GloVe 같은 고정(static) 임베딩보다, Week4에서 다룰 Transformer 기반 contextual embedding — BERT 등 — 이 대세지만, 이 코스 범위상 static embedding까지만 다룬다.)

### Sentiment classification

문장 → 별점(1~5) 같은 many-to-one 문제. 간단 버전: 문장의 각 단어 임베딩을 평균/합산해서 softmax에 넣는 방식도 있지만, 단어 순서를 무시하기 때문에 "completely lacking in good taste, good service, and good ambiance" 같은 문장에서 "good"이 여러 번 나온다고 긍정으로 오분류하기 쉽다. 더 나은 방법은 RNN(many-to-one)에 임베딩 시퀀스를 그대로 넣는 것 — 순서 정보를 살릴 수 있다. 그리고 word embedding을 큰 corpus에서 pre-train해서 가져오면, sentiment 데이터셋 자체는 작아도(라벨링 비용 문제) 학습이 잘 된다 — 이게 transfer learning의 실전 이득이다.

### 임베딩의 편향(bias) 제거

큰 텍스트 corpus로 학습하다 보니, embedding이 "Man : Computer_Programmer = Woman : Homemaker" 같은 성별/인종 편향을 그대로 학습해버리는 문제가 있다. 대응 절차(개념):

1. Bias 방향(예: gender direction) 찾기 — "he"-"she", "male"-"female" 같은 쌍들의 차이 벡터를 평균/SVD로 구함
2. Bias와 무관해야 할 단어들(예: "doctor", "babysitter")을 그 bias 축에 대해 **neutralize**(투영 성분 제거)
3. Bias가 있어도 되는 쌍(예: "grandmother"-"grandfather")은 **equalize**(bias 축 기준으로 대칭이 되도록 조정)

완벽한 해결책은 아니지만, 모델이 사회적 편향을 그대로 증폭시키지 않도록 하는 실무적으로 중요한 단계다.

---

## Week 3. Sequence-to-Sequence & Attention

### Seq2seq (Encoder-Decoder)

기계번역, 이미지 캡셔닝처럼 입출력 길이가 다른 many-to-many 문제에 쓰는 구조. **Encoder** RNN이 입력 시퀀스를 다 읽어서 하나의 벡터(context)로 압축하고, 그 벡터를 초기 상태로 받은 **Decoder** RNN이 출력 시퀀스를 한 단어씩 생성한다.

$$\text{"Jane visite l'Afrique en septembre."} \xrightarrow{\text{encoder}} \text{context} \xrightarrow{\text{decoder}} \text{"Jane visits Africa in September."}$$

### 기계번역 = Conditional language model

Week1의 language model이 $P(y^{<1>}, ..., y^{<T_y>})$를 모델링했다면, 번역은 입력 문장 $x$가 주어졌을 때의 조건부 확률 $P(y^{<1>}, ..., y^{<T_y>} \mid x)$을 모델링하는 것 — 그래서 "conditional" language model이라고 부른다. Decoder 부분 구조는 language model이랑 거의 같고, 다른 건 시작 벡터가 $\vec{0}$이 아니라 encoder의 context라는 점.

### Greedy search의 문제와 Beam search

Language model 샘플링처럼 매 스텝 가장 확률 높은 단어 하나만 고르는 **greedy decoding**은 번역에서 최적이 아니다 — 매 스텝 지역적으로 최선인 선택이 전체 문장으로는 최선이 아닐 수 있기 때문(전체 문장 확률 $\prod P(y^{<t>}\mid ...)$을 최대화하는 게 목표인데, greedy는 이걸 보장 못함).

**Beam search**: 매 스텝마다 상위 $B$개(beam width)의 후보 문장을 유지하면서 진행한다. $B=1$이면 greedy search와 같다.

- Step 1: 첫 단어 후보 중 확률 top $B$개를 유지
- Step 2: 각 후보 뒤에 올 수 있는 다음 단어를 다 계산해서, $B \times V$개 조합 중 다시 top $B$개만 남김
- 이걸 문장이 끝날 때까지(EOS) 반복

**Length normalization**: 확률을 그대로 곱하면(0~1 사이 값들의 곱) 문장이 길어질수록 값이 계속 작아져서 numerical underflow도 나고, 짧은 문장에 편향된 결과를 낸다. 그래서 실제로는 로그를 취한 합을 쓰고, 길이로 정규화한다:

$$\text{score} = \frac{1}{T_y^{\alpha}} \sum_{t=1}^{T_y} \log P(y^{<t>} \mid x, y^{<1>}, ..., y^{<t-1>})$$

$\alpha$는 보통 0.7 정도의 하이퍼파라미터(softer normalization).

$B$가 클수록 결과는 좋아지지만 메모리/속도 비용이 커진다. 실무에서는 $B=10$ 정도가 흔하고, 연구용으로 $B=1000\sim3000$까지 쓰기도 한다. Beam search는 (BFS/DFS 같은) exact search가 아니라 **근사(approximate) search**라는 걸 기억해두면 좋다 — 최적해를 보장하지 않는다.

### Error analysis: RNN 탓인가 Beam search 탓인가

번역 결과가 이상할 때, 문제가 RNN(모델) 때문인지 beam search(탐색) 때문인지 구분하는 방법: 사람이 만든 정답 번역 $y^*$와 모델이 만든 번역 $\hat{y}$에 대해 $P(y^* \mid x)$와 $P(\hat{y} \mid x)$를 모델로 계산해서 비교한다.

- $P(y^*) > P(\hat{y})$인데 모델이 $\hat{y}$를 골랐다면 → **beam search가 문제** (더 좋은 후보가 있었는데 못 찾음, beam width 키우면 개선)
- $P(y^*) \le P(\hat{y})$라면 → **RNN(모델)이 문제** (모델 입장에선 $\hat{y}$가 더 그럴듯하다고 잘못 학습된 것, 모델 구조/데이터/정규화 개선 필요)

이런 식으로 여러 오류 샘플을 분석해서 어느 쪽 문제가 더 많은지 비율을 보고 개선 방향을 정한다.

### BLEU score (개념)

번역 품질을 사람이 일일이 평가하기 힘드니까 자동으로 점수를 매기는 지표. 기본 아이디어: 모델 출력 문장의 n-gram들이 사람이 만든 reference 번역(들)에 얼마나 등장하는지를 precision으로 계산한다(같은 단어가 reference보다 많이 반복되지 않게 clip 처리하고, 문장이 너무 짧으면 불리하게 brevity penalty를 곱함). 완벽한 지표는 아니지만 서로 다른 모델/설정을 빠르게 비교하는 단일 실수값 지표로 유용해서 번역/캡셔닝 논문에서 표준처럼 쓰인다. 세부 공식은 몰라도 되고, "n-gram 겹침 기반 자동 평가 지표"라는 것만 기억하면 된다.

### Attention model

**동기**: 기본 encoder-decoder는 입력 문장 전체를 고정 크기 벡터 하나로 압축한다. 문장이 짧으면 괜찮은데, 길어지면(예: 30~40단어 이상) 번역 품질(BLEU)이 뚝 떨어진다 — 사람은 긴 문장을 번역할 때도 한 번에 다 외워서 번역하지 않고, 원문의 관련된 부분을 그때그때 다시 보면서 번역하는데 기본 구조는 그걸 못 한다.

**직관**: decoder가 매 출력 단어를 만들 때, 입력 문장의 **모든 위치**에 대해 "지금 이 단어를 만드는 데 얼마나 관련 있는지"를 나타내는 가중치 $\alpha^{<t,t'>}$를 계산하고, 그 가중치로 입력 activation들을 가중합해서 context를 만든다. 즉 매 출력 스텝마다 "입력의 어디를 볼지"가 달라진다(spotlight를 이동시키는 느낌).

$$context^{<t>} = \sum_{t'} \alpha^{<t,t'>} a^{<t'>}$$

여기서 $\alpha^{<t,t'>}$는 출력 $t$번째 단어가 입력 $t'$번째 단어에 주는 가중치이다. 이 $\alpha$가 어디서 나오냐면 — 작은 신경망(1~2 layer짜리 fully-connected)이 "decoder의 직전 상태 $s^{<t-1>}$"와 "encoder의 $t'$번째 activation $a^{<t'>}$"을 입력으로 받아 관련도 점수 $e^{<t,t'>}$를 출력하고, 이 점수들을 $t'$ 축으로 softmax를 취한 게 $\alpha^{<t,t'>}$다:

$$e^{<t,t'>} = \text{small NN}(s^{<t-1>}, a^{<t'>}), \qquad \alpha^{<t,t'>} = \frac{\exp(e^{<t,t'>})}{\sum_{t'} \exp(e^{<t,t'>})}$$

즉 각 $t$에 대해 $\sum_{t'} \alpha^{<t,t'>} = 1$이 되도록 정규화된다. 이 small NN도 전체 네트워크와 함께 backprop으로 학습되기 때문에, "어느 입력 단어에 집중할지"를 사람이 정해주지 않아도 모델이 알아서 배운다. $\alpha$들을 시각화하면 실제로 번역 시 대응되는 원문 단어에 가중치가 크게 실리는 걸 볼 수 있다(대각선에 가까운 패턴).

이 attention 아이디어가 나중에 Week4의 self-attention / Transformer로 발전한다 — 여기서는 "긴 시퀀스에서 관련된 부분에 집중하는 가중치 메커니즘"이라는 직관만 잡고 넘어가면 충분하다.

### Speech recognition과 CTC

오디오(시간축이 매우 긴 시퀀스, 예: 10초 오디오를 100Hz로 샘플링하면 1000 타임스텝) → 텍스트(훨씬 짧은 시퀀스) 문제라서, 입력과 출력 길이가 크게 다르다. **CTC (Connectionist Temporal Classification)** 방식은 매 입력 타임스텝마다 문자를 출력하게 하되, 반복된 문자와 특수 blank 토큰(`_`)을 허용한다. 예: "ttt_h_eee___  _qqq__" 같은 raw 출력에서, "blank로 구분되지 않은 연속 반복 문자는 하나로 합치고 blank는 제거"하는 규칙으로 "the q..." 처럼 최종 텍스트를 복원한다. 이렇게 하면 입력 길이 = 출력 길이인 many-to-many 구조를 그대로 쓰면서도 실제 텍스트 길이가 짧은 문제를 풀 수 있다.

### Trigger word detection

"Hey Siri", "OK Google" 같은 wake word를 감지하는 문제. 오디오 시퀀스를 RNN에 흘려보내면서, trigger word가 끝나는 시점의 라벨을 0에서 1로 바꿔주고 그 뒤 몇 개 타임스텝 동안 1을 유지하도록(라벨을 약간 두껍게) 학습시키는 게 실전 트릭이다 — 그냥 정확히 그 순간 한 타임스텝만 1로 라벨링하면 0과 1의 비율이 너무 불균형해서 학습이 잘 안 된다.

---

## Week 4. Transformer

### 동기: RNN의 한계

RNN/GRU/LSTM은 구조상 **순차적(sequential)** 으로 계산해야 한다 — $t$번째 activation을 계산하려면 $t-1$번째가 끝나야 한다. 이건 두 가지 문제를 만든다:

1. 병렬화(parallelization)가 안 돼서 긴 시퀀스일수록 학습이 느리다 (GPU를 제대로 못 씀)
2. 여전히 vanishing gradient 계열 문제로 아주 긴 거리의 의존관계를 잡기 어렵다 (GRU/LSTM으로 완화는 했지만 완전히 해결은 아님)

Transformer는 순차 계산을 버리고, **self-attention**으로 시퀀스 전체를 한 번에(병렬로) 처리하면서도 단어 간 관계를 잡아낸다는 게 핵심 아이디어다.

### Self-Attention

직관: 문장 "Jane visite l'Afrique en septembre, et elle admire la culture."에서 "l'Afrique"라는 단어를 이해하려면 — 이게 지리적 장소인지, 뭘 하는 행위인지 등 — 문장의 다른 단어들과의 관계를 봐야 한다. Self-attention은 각 단어에 대해 "문장의 다른 모든 단어를 참고해서 이 단어의 의미를 다시 표현(re-represent)"하는 연산이다.

각 단어의 임베딩에서 세 개의 벡터를 만든다: **Query($Q$), Key($K$), Value($V$)** — 학습되는 가중치 행렬 $W^Q, W^K, W^V$를 곱해서 얻는다.

- $Q$: "나는 지금 무엇을 찾고 있는가"에 대한 질문 벡터
- $K$: "나는 이런 정보를 갖고 있다"는 이름표/키 벡터
- $V$: 실제로 전달할 값

직관적으로, 각 단어의 $Q$를 다른 모든 단어의 $K$와 내적(dot product)해서 "얼마나 관련 있는지" 점수를 구하고(내적이 클수록 관련도 높음), 그 점수를 softmax로 정규화한 뒤 그 가중치로 $V$들을 가중합한다. 도서관에서 질문(Q)을 들고 가서 책 제목(K)들과 비교해 가장 관련 있는 책을 찾고, 그 책 내용(V)을 가져오는 것에 비유할 수 있다.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

$\sqrt{d_k}$(key 벡터 차원의 제곱근)로 나눠주는 건 내적 값이 차원이 커질수록 너무 커져서 softmax gradient가 작아지는(vanishing) 걸 막기 위한 스케일링이다. 이 식 자체는 "관련도로 가중평균 내는 것"이라는 그림만 잡으면 되고, 세부 유도는 안 외워도 된다.

### Multi-head Attention

한 세트의 $Q,K,V$만 쓰면 "관련도"를 한 가지 관점으로만 보게 된다. Multi-head attention은 $W^Q, W^K, W^V$를 여러 세트(head, 보통 8개 등) 두고 각 head가 서로 다른 종류의 관계(예: 어떤 head는 "누가 누구에게 무엇을 했는가", 다른 head는 "언제 일어났는가")를 병렬로 학습하게 한 뒤, 각 head의 attention 출력을 이어붙이고(concat) 다시 한 번 선형변환한다.

$$\text{MultiHead}(Q,K,V) = \text{concat}(head_1, ..., head_h) W^O$$

CNN에서 필터를 여러 개 두어 서로 다른 특징을 잡는 것과 비슷한 발상이라고 생각하면 된다.

### Transformer 전체 구조

기본은 여전히 encoder-decoder 구조인데, RNN 대신 attention block을 쌓아서 만든다.

**Encoder** (여러 층 반복):
1. Multi-head self-attention
2. Residual connection + Layer normalization ("Add & Norm")
3. Feed-forward network(단순 fully-connected, 위치별로 동일하게 적용)
4. 다시 Residual + Layer norm

**Decoder** (여러 층 반복):
1. **Masked** multi-head self-attention — 아직 생성되지 않은 미래 단어를 "커닝"하지 못하도록, 자기 자신 이후 위치의 attention score를 $-\infty$로 masking한 뒤 softmax를 취한다(그래서 실질적으로 0이 됨). Language model처럼 왼쪽 문맥만 보고 다음 단어를 예측하게 강제하는 장치.
2. Add & Norm
3. Encoder 출력을 $K,V$로, decoder 자신의 상태를 $Q$로 쓰는 **encoder-decoder attention** (Week3의 attention과 개념적으로 같은 역할 — "출력을 만들 때 입력의 어디를 볼지")
4. Add & Norm → Feed-forward → Add & Norm

**Positional encoding**: self-attention 자체는 순서 개념이 없다(단어 순서를 바꿔도 집합으로는 같은 계산) — RNN처럼 순차 처리를 안 하기 때문에 위치 정보를 별도로 주입해야 한다. 그래서 각 위치 $pos$, 차원 $i$마다 $\sin$/$\cos$ 함수 값으로 만든 고정된(학습 안 되는) 벡터를 단어 임베딩에 더해준다:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

sin/cos를 쓰는 이유는 이 함수들의 주기성 덕분에 상대적 위치 관계를 표현하기 좋고, 학습 때 못 본 더 긴 시퀀스 길이에도 확장이 되기 때문이다 — 공식 자체보다 "위치 정보를 attention에 넣어주는 장치"라는 역할만 기억하면 된다.

**Residual connection + Layer normalization**: **Residual(skip) connection**은 sub-layer의 입력을 출력에 그대로 더해주는 것이다 — $x_{out} = x_{in} + \text{SubLayer}(x_{in})$. 층이 깊어져도 gradient가 이 "지름길"을 타고 그대로 앞으로 전달될 수 있어서, 신경망이 깊어질수록 오히려 학습이 안 되는 문제(degradation)를 완화해준다(CNN 계열의 ResNet에서 처음 쓰인 아이디어인데, 여기선 "입력을 출력에 더해서 gradient가 죽지 않게 지름길을 만든다"는 그림만 알면 된다). 이렇게 더한 결과를 **layer normalization**으로 정리하는데, 이건 batch normalization(미니배치 안에서 같은 feature 축으로 정규화)과 달리 **한 샘플(하나의 단어 벡터) 안에서** 평균/분산을 구해 정규화하는 방식이다 — 시퀀스마다 길이가 다른 NLP에서 batch norm보다 더 잘 맞는다.

```python
# 개념적 pseudo-code (Keras 스타일)
x = x + MultiHeadAttention(Q=x, K=x, V=x)   # self-attention + residual
x = LayerNorm(x)
x = x + FeedForward(x)                       # FFN + residual
x = LayerNorm(x)
```

---

## 핵심 요약

- 시퀀스 데이터는 표준 NN으로 못 다룬다 → **RNN**이 타임스텝마다 파라미터를 공유하면서 activation을 이어받는 구조로 해결.
- RNN은 vanishing gradient 때문에 긴 문맥 기억이 약하다 → **GRU/LSTM**이 gate(update/forget/output)로 정보를 선택적으로 유지·삭제하도록 개선.
- 문맥을 양방향으로 봐야 하는 문제(NER 등)엔 **Bidirectional RNN**, 층을 더 쌓고 싶으면 **Deep RNN**.
- 단어를 one-hot이 아니라 **word embedding**(dense vector)으로 표현하면 단어 간 유사도/유추가 가능해지고, 큰 corpus에서 pre-train해서 작은 task로 transfer할 수 있다. 학습 방법은 Word2Vec(skip-gram + negative sampling), GloVe 등.
- 입출력 길이가 다른 문제(번역 등)는 **encoder-decoder(seq2seq)** 구조, 디코딩은 greedy 대신 **beam search**(+length normalization)로 여러 후보를 탐색.
- 긴 문장에서 성능이 떨어지는 문제를 **attention**(입력의 관련 부분에 가중치를 주는 메커니즘)으로 해결.
- RNN의 순차 처리 한계(병렬화 불가, 여전한 장거리 의존성 문제)를 **self-attention**으로 없앤 게 **Transformer** — Q/K/V, multi-head attention, positional encoding, residual+layer norm, masked attention으로 구성되며 현재 NLP(그리고 그 너머)의 표준 구조가 됐다.

## 헷갈리기 쉬운 점

- **$T_x$ vs $T_y$**: 입력 길이와 출력 길이는 다를 수 있다. Many-to-many 유형에서도 "길이가 같은 경우"(NER)와 "다른 경우"(번역, encoder-decoder 필요)를 구분해야 한다.
- **GRU의 $c^{<t>}$ vs LSTM의 $c^{<t>}$, $a^{<t>}$**: GRU는 memory cell과 activation을 사실상 같은 것으로 취급하지만($a^{<t>} = c^{<t>}$), LSTM은 둘을 분리해서 관리한다(output gate가 $c$에서 $a$를 걸러냄).
- **Update gate 개수**: GRU는 update gate 하나로 "새로 채울 비율"과 "옛날 걸 유지할 비율"을 $\Gamma_u$, $1-\Gamma_u$로 묶어서 쓰고, LSTM은 update와 forget이 독립된 별개 게이트라 둘이 반드시 합쪽 1일 필요가 없다.
- **Word2vec(skip-gram/negative sampling) vs GloVe**: 둘 다 word embedding 학습법이지만, word2vec은 (context, target) 쌍을 슬라이딩 윈도우로 뽑아 예측 문제로 풀고, GloVe는 co-occurrence 통계(카운트)를 미리 계산해서 직접 회귀하듯 맞춘다.
- **Attention(Week3) vs Self-attention(Week4)**: Week3 attention은 "decoder가 encoder의 어디를 볼지" 정하는 용도(입력-출력 사이 attention, cross-attention과 같은 역할). Week4 self-attention은 "같은 시퀀스 내에서 단어들끼리 서로를 참고"하는 것 — Transformer에는 두 종류가 다 들어있다(encoder self-attention, decoder masked self-attention, encoder-decoder attention).
- **Masking의 의미**: Transformer decoder의 masking은 gradient clipping의 "clipping"과 전혀 다른 개념이다 — masking은 "미래 정보를 못 보게 가리는 것", clipping은 "gradient 크기를 제한하는 것".
- **BLEU score는 완벽한 지표가 아니다**: n-gram 겹침만 보기 때문에 의미는 같지만 단어 선택이 다른 좋은 번역에 낮은 점수를 줄 수 있다. 그래도 모델 비교용 단일 지표로는 실무에서 표준으로 쓰인다.
- **Beam search는 근사 알고리즘**: beam width $B$가 작으면 최적해를 못 찾을 수 있다. 번역이 이상할 때 이게 beam 탓인지 모델 탓인지 구분하려면 $P(y^*\mid x)$와 $P(\hat y\mid x)$를 직접 비교해봐야 한다(Week3 error analysis 참고).

## RNN → GRU → LSTM → Attention → Transformer 한눈에 비교

| 구조 | 핵심 아이디어 | 해결한 문제 | 남은 한계 |
|---|---|---|---|
| **RNN** | 타임스텝마다 같은 파라미터($W_a$) 공유, activation을 다음 스텝에 전달 | 가변 길이 시퀀스, 위치 간 특징 공유 | Vanishing gradient로 장거리 의존성 학습 약함 |
| **GRU** | Update gate $\Gamma_u$로 memory cell $c^{<t>}$를 유지할지 갱신할지 조절 | Vanishing gradient 완화, 장거리 의존성 개선 | LSTM보다 단순하지만 표현력은 다소 제한적 |
| **LSTM** | Update/Forget/Output 3개 gate로 $c^{<t>}$와 $a^{<t>}$를 분리 관리 | GRU보다 더 유연한 정보 제어 | 여전히 순차 계산이라 병렬화 불가, 파라미터 많음 |
| **Attention (seq2seq)** | 출력 매 스텝마다 입력 전체에 대한 가중치 $\alpha^{<t,t'>}$ 계산 후 가중합 | 긴 문장에서 고정 context vector의 정보 손실 문제 | 여전히 RNN 기반이라 순차 계산 병목은 남아있음 |
| **Transformer (Self-Attention)** | Q/K/V 기반으로 시퀀스 내 모든 위치를 한 번에(병렬) 비교 | 순차 계산 제거(병렬화), 장거리 의존성도 한 번의 attention으로 직접 연결 | 시퀀스 길이에 대해 $O(n^2)$ 연산/메모리, 위치 정보 별도 주입(positional encoding) 필요 |

---

## Lab 예제 — 직접 짜보기

> numpy + matplotlib만으로 RNN 셀부터 Transformer의 attention까지 직접 짠다. 뼈대의 `TODO`를 채우고 **체크포인트**와 비교해보자. 정답은 접혀있다.

### Lab 1. RNN forward + gradient clipping + vanishing/exploding 체감

**목표**: RNN 셀 하나 → 시퀀스 전체 forward를 구현한다(shape: $x$는 `(n_x, m, T_x)`). exploding gradient 대책인 clipping 두 방식(값 기준 / norm 기준)을 짜보고, 같은 $W_{aa}$를 여러 번 곱할 때 gradient 크기가 어떻게 되는지 그래프로 본다.

```python
import numpy as np
import matplotlib.pyplot as plt

def softmax(z, axis=0):
    e = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def rnn_cell_forward(xt, a_prev, p):
    # xt: (n_x, m), a_prev: (n_a, m)
    a_next = None    # TODO: tanh(Waa a_prev + Wax xt + ba)
    yt_pred = None   # TODO: softmax(Wya a_next + by)
    return a_next, yt_pred

def rnn_forward(x, a0, p):
    n_x, m, T_x = x.shape
    n_y, n_a = p["Wya"].shape
    a = np.zeros((n_a, m, T_x)); y_pred = np.zeros((n_y, m, T_x))
    a_next = a0
    # TODO: t = 0 ~ T_x-1 돌면서 셀 호출, a와 y_pred의 t번째 칸에 저장 (파라미터는 모든 t에서 공유!)
    return a, y_pred

np.random.seed(1)
n_x, n_a, n_y, m, T_x = 3, 5, 2, 10, 4
p = {"Waa": np.random.randn(n_a, n_a), "Wax": np.random.randn(n_a, n_x), "ba": np.random.randn(n_a, 1),
     "Wya": np.random.randn(n_y, n_a), "by": np.random.randn(n_y, 1)}
a, y_pred = rnn_forward(np.random.randn(n_x, m, T_x), np.random.randn(n_a, m), p)
print("a.shape:", a.shape, " y_pred.shape:", y_pred.shape, " y_pred 합(=1):", y_pred[:, 0, 0].sum())

def clip(grads, max_value):
    return None   # TODO: 모든 gradient를 [-max_value, max_value]로 자르기 (np.clip)

def clip_by_norm(grads, max_norm):
    return None   # TODO: 전체 gradient norm이 max_norm보다 크면 전체를 같은 비율로 축소 (방향 유지)

grads = {"dWaa": np.array([[50.0, -3.0], [0.5, -80.0]]), "db": np.array([[10.0], [-0.1]])}
print("value clip:", clip(grads, 5)["dWaa"].tolist())
print("norm clip:", np.round(clip_by_norm(grads, 5)["dWaa"], 3).tolist())

# BPTT에서 gradient가 매 step Waa^T를 곱하며 뒤로 간다고 단순화
for scale in [0.5, 1.0, 1.5]:
    Waa = np.eye(n_a) * scale
    g = np.ones((n_a, 1))
    norms = []
    for t in range(50):
        g = Waa.T @ g
        norms.append(np.linalg.norm(g))
    plt.semilogy(norms, label=f"Waa = {scale} * I")
plt.xlabel("steps back"); plt.ylabel("|grad|")
plt.legend(); plt.title("Vanishing / exploding gradients"); plt.show()
```

**체크포인트**
- `a.shape: (5, 10, 4)`, `y_pred.shape: (2, 10, 4)`, softmax 합 1.0.
- value clip `[[5, -3], [0.5, -5]]` — 원소마다 따로 잘려서 **방향이 바뀐다**. norm clip `[[2.634, -0.158], [0.026, -4.214]]` — 비율(방향)은 그대로, 크기만 줄어듦.
- 그래프: 0.5면 50 step 뒤 `1e-15` 수준으로 사라지고(vanishing), 1.5면 `1e9`로 폭발(exploding). 앞쪽 단어의 정보가 뒤까지 못 가는 이유.

<details>
<summary>정답 보기</summary>

```python
def rnn_cell_forward(xt, a_prev, p):
    a_next = np.tanh(p["Waa"] @ a_prev + p["Wax"] @ xt + p["ba"])
    yt_pred = softmax(p["Wya"] @ a_next + p["by"])
    return a_next, yt_pred

def rnn_forward(x, a0, p):
    n_x, m, T_x = x.shape
    n_y, n_a = p["Wya"].shape
    a = np.zeros((n_a, m, T_x)); y_pred = np.zeros((n_y, m, T_x))
    a_next = a0
    for t in range(T_x):
        a_next, yt = rnn_cell_forward(x[:, :, t], a_next, p)
        a[:, :, t] = a_next; y_pred[:, :, t] = yt
    return a, y_pred

def clip(grads, max_value):
    return {k: np.clip(g, -max_value, max_value) for k, g in grads.items()}

def clip_by_norm(grads, max_norm):
    total = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
    scale = min(1.0, max_norm / total)
    return {k: g * scale for k, g in grads.items()}
```

</details>

### Lab 2. LSTM / GRU 셀 — gate가 memory를 지키는 걸 보기

**목표**: LSTM 셀과 GRU 셀의 forward를 구현한다. 그리고 gate bias를 극단적으로 줘서 (forget≈1, update≈0) 랜덤 입력을 100 step 넣어도 memory cell $c$가 거의 그대로 보존된다는 걸 확인한다 — vanishing gradient를 푸는 원리.

```python
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))

def lstm_cell_forward(xt, a_prev, c_prev, p):
    concat = np.concatenate([a_prev, xt], axis=0)     # [a<t-1>; x<t>]
    ft = None       # TODO: forget gate  = σ(Wf concat + bf)
    it = None       # TODO: update gate  = σ(Wi concat + bi)
    cct = None      # TODO: 후보 memory   = tanh(Wc concat + bc)
    c_next = None   # TODO: ft * c_prev + it * cct
    ot = None       # TODO: output gate  = σ(Wo concat + bo)
    a_next = None   # TODO: ot * tanh(c_next)
    return a_next, c_next, (ft, it, ot)

def gru_cell_forward(xt, c_prev, p):
    concat = np.concatenate([c_prev, xt], axis=0)
    gamma_r = None   # TODO: relevance gate
    gamma_u = None   # TODO: update gate
    c_tilde = None   # TODO: tanh(Wc [Γr * c_prev; xt] + bc)
    c_next = None    # TODO: Γu * c_tilde + (1 - Γu) * c_prev
    return c_next, gamma_u

n_x, n_a, m = 3, 5, 10
np.random.seed(2)
lp = {k: np.random.randn(n_a, n_a + n_x) for k in ["Wf", "Wi", "Wc", "Wo"]}
lp.update({k: np.zeros((n_a, 1)) for k in ["bf", "bi", "bc", "bo"]})
a_next, c_next, gates = lstm_cell_forward(np.random.randn(n_x, m), np.random.randn(n_a, m), np.random.randn(n_a, m), lp)
print("LSTM a_next, c_next shape:", a_next.shape, c_next.shape)

lp["bf"][:] = 10; lp["bi"][:] = -10       # forget gate ≈ 1, update gate ≈ 0
c = np.ones((n_a, 1)); a_prev = np.zeros((n_a, 1))
for t in range(100):
    a_prev, c, _ = lstm_cell_forward(np.random.randn(n_x, 1), a_prev, c, lp)
print("100 step 뒤 memory cell (bf=10, bi=-10):", np.round(c.ravel(), 3))

gp = {k: np.random.randn(n_a, n_a + n_x) for k in ["Wr", "Wu", "Wc"]}
gp.update({"br": np.zeros((n_a, 1)), "bu": np.full((n_a, 1), -10.0), "bc": np.zeros((n_a, 1))})
c = np.ones((n_a, 1))
for t in range(100):
    c, gu = gru_cell_forward(np.random.randn(n_x, 1), c, gp)
print("GRU 100 step 뒤 (bu=-10 -> Γu≈0):", np.round(c.ravel(), 3))
```

**체크포인트**
- shape `(5, 10) (5, 10)`.
- LSTM: 100 step 뒤에도 $c$ ≈ `[0.98, 1.0, 0.99, 0.79, 1.0]` — 처음 넣은 1이 거의 그대로. bf/bi를 0으로 되돌리면 금방 다른 값으로 덮어써진다.
- GRU도 $\Gamma_u \approx 0$이면 $c^{<t>} \approx c^{<t-1>}$이라 ≈ 1 근처 유지. 강의의 "The cat, which already ate ..., **was** full"에서 단수/복수 정보를 문장 끝까지 들고 가는 원리가 이것.

<details>
<summary>정답 보기</summary>

```python
def lstm_cell_forward(xt, a_prev, c_prev, p):
    concat = np.concatenate([a_prev, xt], axis=0)
    ft = sigmoid(p["Wf"] @ concat + p["bf"])
    it = sigmoid(p["Wi"] @ concat + p["bi"])
    cct = np.tanh(p["Wc"] @ concat + p["bc"])
    c_next = ft * c_prev + it * cct
    ot = sigmoid(p["Wo"] @ concat + p["bo"])
    a_next = ot * np.tanh(c_next)
    return a_next, c_next, (ft, it, ot)

def gru_cell_forward(xt, c_prev, p):
    concat = np.concatenate([c_prev, xt], axis=0)
    gamma_r = sigmoid(p["Wr"] @ concat + p["br"])
    gamma_u = sigmoid(p["Wu"] @ concat + p["bu"])
    c_tilde = np.tanh(p["Wc"] @ np.concatenate([gamma_r * c_prev, xt], axis=0) + p["bc"])
    c_next = gamma_u * c_tilde + (1 - gamma_u) * c_prev
    return c_next, gamma_u
```

</details>

### Lab 3. Word embedding — cosine similarity, 유추, 편향 제거

**목표**: 직접 만든 5차원 toy 임베딩(gender, royal, age, food, occupation)으로 cosine similarity와 `man : woman = king : ?` 유추를 구현한다. 그 다음 gender 방향 $g$를 구해서 neutralize(직업 단어에서 성별 성분 제거)와 equalize(grandmother/grandfather를 대칭으로)를 해본다.

```python
import numpy as np

E = {   # [gender(+=male), royal, age, food, occupation]
    "man": [1.0, 0.0, 0.6, 0.0, 0.0], "woman": [-1.0, 0.0, 0.6, 0.0, 0.0],
    "king": [0.95, 0.95, 0.7, 0.0, 0.1], "queen": [-0.97, 0.95, 0.7, 0.0, 0.1],
    "apple": [0.0, 0.0, 0.0, 0.95, 0.0], "orange": [0.01, 0.0, 0.0, 0.97, 0.0],
    "boy": [1.0, 0.0, 0.1, 0.0, 0.0], "girl": [-1.0, 0.0, 0.1, 0.0, 0.0],
    "doctor": [0.3, 0.0, 0.5, 0.0, 0.95], "nurse": [-0.4, 0.0, 0.5, 0.0, 0.9],
    "babysitter": [-0.35, 0.0, 0.3, 0.0, 0.8],
    "grandmother": [-0.9, 0.0, 1.0, 0.0, 0.3],     # 일부러 occupation 쪽에 편향을 섞어둠
    "grandfather": [1.0, 0.0, 1.0, 0.0, 0.0],
}
np.random.seed(0)
E = {w: np.array(v) + np.random.randn(5) * 0.02 for w, v in E.items()}
E = {w: v / np.linalg.norm(v) for w, v in E.items()}   # equalize 공식은 단위벡터를 가정

def cosine_similarity(u, v):
    return None   # TODO

def complete_analogy(a, b, c, E):
    # a : b = c : ?  ->  e_b - e_a 와 e_? - e_c 의 cosine이 가장 큰 단어 (a, b, c 자신은 제외)
    return None   # TODO

print("cos(apple, orange):", round(cosine_similarity(E["apple"], E["orange"]), 3))
print("cos(apple, king):", round(cosine_similarity(E["apple"], E["king"]), 3))
print("man : woman = king :", complete_analogy("man", "woman", "king", E))
print("man : woman = boy :", complete_analogy("man", "woman", "boy", E))
print("boy : girl = man :", complete_analogy("boy", "girl", "man", E))

g = np.mean([E["man"] - E["woman"], E["boy"] - E["girl"], E["grandfather"] - E["grandmother"]], axis=0)

def neutralize(e, g):
    return None   # TODO: e에서 g 방향 성분(projection)을 뺀다

def equalize(pair, g, E):
    e1, e2 = E[pair[0]], E[pair[1]]
    mu = (e1 + e2) / 2
    mu_B = np.dot(mu, g) / np.sum(g ** 2) * g
    mu_orth = mu - mu_B                                   # 두 단어가 공유할 "성별 없는" 부분
    scale = np.sqrt(np.abs(1 - np.sum(mu_orth ** 2)))
    e1B = np.dot(e1, g) / np.sum(g ** 2) * g
    e2B = np.dot(e2, g) / np.sum(g ** 2) * g
    e1B_c = scale * (e1B - mu_B) / np.linalg.norm(e1B - mu_B)
    e2B_c = scale * (e2B - mu_B) / np.linalg.norm(e2B - mu_B)
    return mu_orth + e1B_c, mu_orth + e2B_c

for w in ["doctor", "nurse", "babysitter"]:
    before = cosine_similarity(E[w], g)
    after = cosine_similarity(neutralize(E[w], g), g)
    print(f"{w:>10}: gender 방향과 cos {before:+.3f} -> neutralize 후 {after:+.3f}")
e_gm, e_gf = equalize(("grandmother", "grandfather"), g, E)
e_bs = neutralize(E["babysitter"], g)
print("equalize 전 babysitter와의 거리:", round(np.linalg.norm(e_bs - E["grandmother"]), 3), round(np.linalg.norm(e_bs - E["grandfather"]), 3))
print("equalize 후 babysitter와의 거리:", round(np.linalg.norm(e_bs - e_gm), 3), round(np.linalg.norm(e_bs - e_gf), 3))
```

**체크포인트**
- cos(apple, orange) ≈ 0.999, cos(apple, king) ≈ -0.01.
- 유추: `queen`, `girl`, `woman`.
- neutralize 전: doctor `+0.22`(남성 쪽), nurse `-0.40`, babysitter `-0.43`(여성 쪽) → 후: 전부 `0.000`.
- equalize 전 babysitter와의 거리 `1.024 vs 1.151`(grandmother 쪽이 더 가까움 = 편향) → 후 `1.089 vs 1.089`로 동일.

<details>
<summary>정답 보기</summary>

```python
def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def complete_analogy(a, b, c, E):
    target = E[b] - E[a]
    best, best_sim = None, -np.inf
    for w, vec in E.items():
        if w in (a, b, c):
            continue
        sim = cosine_similarity(target, vec - E[c])
        if sim > best_sim:
            best, best_sim = w, sim
    return best

def neutralize(e, g):
    e_bias = np.dot(e, g) / np.sum(g ** 2) * g
    return e - e_bias
```

</details>

### Lab 4. Greedy search vs Beam search

**목표**: 작은 가짜 "번역 모델" $P(y^{<t>} \mid x, y^{<1>},\dots,y^{<t-1>})$ 확률표 위에서 greedy search와 beam search(+ length normalization)를 구현하고, greedy가 당장 확률 높은 단어("going")를 고르다가 전체 문장 확률에서 손해 보는 걸 확인한다.

```python
import numpy as np

def next_word_probs(prefix):
    # "Jane visite l'Afrique en septembre" 번역 모델을 흉내낸 확률표
    table = {
        (): {"jane": 0.9, "in": 0.1},
        ("jane",): {"is": 1.0},
        ("jane", "is"): {"going": 0.5, "visiting": 0.4, "in": 0.1},
        ("jane", "is", "going"): {"to": 0.6, "in": 0.4},
        ("jane", "is", "going", "to"): {"september": 0.55, "africa": 0.35, "<eos>": 0.1},
        ("jane", "is", "going", "in"): {"september": 0.3, "africa": 0.7},
        ("jane", "is", "visiting"): {"africa": 0.95, "september": 0.05},
        ("jane", "is", "visiting", "africa"): {"in": 0.9, "<eos>": 0.1},
        ("jane", "is", "visiting", "africa", "in"): {"september": 1.0},
    }
    return table.get(tuple(prefix), {"<eos>": 1.0})

def greedy_search(max_len=8):
    seq, logp = [], 0.0
    # TODO: 매 step 가장 확률 높은 단어 하나만 고르고 log 확률을 누적. <eos>면 종료
    return seq, logp

def beam_search(B=3, max_len=8, alpha=0.7):
    beams = [([], 0.0, False)]           # (seq, log P, 끝났는지)
    for _ in range(max_len):
        candidates = []
        # TODO: 각 beam마다
        #   - 이미 끝났으면 그대로 candidates에
        #   - 아니면 모든 다음 단어로 확장 (<eos>면 끝난 상태로)
        # TODO: log P 기준 상위 B개만 남기기
        if all(done for _, _, done in beams):
            break
    # TODO: length normalization — log P / T^alpha 가 가장 큰 후보의 (seq, logp) 반환
    return None

for name, (seq, logp) in [("greedy", greedy_search()), ("beam B=1", beam_search(B=1)), ("beam B=3", beam_search(B=3))]:
    print(f"{name:>9}: {' '.join(seq):<40} P = {np.exp(logp):.4f}")
```

**체크포인트**
- greedy: `jane is going to september` P = 0.1485 — 이상한 번역인데 고른 이유는 매 step "going"(0.5 > 0.4), "to"(0.6)가 당장 높았기 때문.
- beam B=1은 greedy와 똑같다(B=1 beam search = greedy).
- beam B=3: `jane is visiting africa in september` P = 0.3078 — 두 배 더 높은 확률의 문장을 찾는다.
- log 확률을 더하는 이유: 확률을 계속 곱하면 문장이 길어질 때 underflow. `alpha`를 0으로 하면 짧은 문장이 유리해지는 것도 확인해보자.

<details>
<summary>정답 보기</summary>

```python
def greedy_search(max_len=8):
    seq, logp = [], 0.0
    for _ in range(max_len):
        probs = next_word_probs(seq)
        w = max(probs, key=probs.get)
        logp += np.log(probs[w])
        if w == "<eos>":
            break
        seq.append(w)
    return seq, logp

def beam_search(B=3, max_len=8, alpha=0.7):
    beams = [([], 0.0, False)]
    for _ in range(max_len):
        candidates = []
        for seq, logp, done in beams:
            if done:
                candidates.append((seq, logp, True))
                continue
            for w, pr in next_word_probs(seq).items():
                if w == "<eos>":
                    candidates.append((seq, logp + np.log(pr), True))
                else:
                    candidates.append((seq + [w], logp + np.log(pr), False))
        beams = sorted(candidates, key=lambda c: -c[1])[:B]
        if all(done for _, _, done in beams):
            break
    return max(beams, key=lambda c: c[1] / max(len(c[0]), 1) ** alpha)[:2]
```

</details>

### Lab 5. Scaled dot-product attention → Multi-head → Transformer encoder 한 층

**목표**: $\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$를 구현하고, decoder용 causal mask, multi-head attention, sin/cos positional encoding, Add & Norm까지 이어 붙여 encoder sub-layer 하나를 만든다. 마지막으로 "positional encoding이 없으면 self-attention은 순서를 모른다"를 실험으로 확인한다.

```python
import numpy as np
import matplotlib.pyplot as plt

def softmax(z, axis=-1):
    e = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q: (T_q, d_k), K: (T_k, d_k), V: (T_k, d_v)
    scores = None    # TODO: Q K^T / sqrt(d_k)
    if mask is not None:
        scores = None   # TODO: mask가 False인 곳을 -1e9로 (softmax 후 0이 되게)
    weights = None   # TODO: 행(쿼리)마다 softmax
    return weights @ V, weights

def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, mask=None):
    # TODO: head마다 X@Wq[h], X@Wk[h], X@Wv[h]로 attention -> 결과를 concat -> @ Wo
    return None

def positional_encoding(T, d):
    PE = np.zeros((T, d))
    # TODO: PE[pos, 2i] = sin(pos / 10000^(2i/d)), PE[pos, 2i+1] = cos(같은 각도)
    return PE

def layer_norm(x, eps=1e-6):
    # batch norm과 달리 "한 토큰 벡터 안에서"(마지막 축) 평균/분산
    return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + eps)

np.random.seed(0)
T, d_model, n_heads = 5, 16, 4
d_k = d_model // n_heads
X = np.random.randn(T, d_model) + positional_encoding(T, d_model)   # 단어 임베딩 + 위치
out, w = scaled_dot_product_attention(X, X, X)                      # self-attention: Q=K=V=X
print("attention weights 각 행의 합:", np.round(w.sum(axis=1), 3))

causal = np.tril(np.ones((T, T), dtype=bool))    # 자기 자신 + 과거만 볼 수 있음
_, w_masked = scaled_dot_product_attention(X, X, X, mask=causal)
print("causal mask weights:\n", np.round(w_masked, 2))

Wq = np.random.randn(n_heads, d_model, d_k) * 0.3
Wk = np.random.randn(n_heads, d_model, d_k) * 0.3
Wv = np.random.randn(n_heads, d_model, d_k) * 0.3
Wo = np.random.randn(n_heads * d_k, d_model) * 0.3
Z = layer_norm(X + multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads))   # Add & Norm
print("encoder sub-layer 출력 shape:", Z.shape, " 위치별 평균≈0:", np.round(Z.mean(axis=-1), 3))

# PE 없이 입력 순서를 섞으면?
perm = np.random.permutation(T)
X_raw = np.random.randn(T, d_model)
o1, _ = scaled_dot_product_attention(X_raw, X_raw, X_raw)
o2, _ = scaled_dot_product_attention(X_raw[perm], X_raw[perm], X_raw[perm])
print("PE 없이 순서 섞으면 출력도 똑같이 섞이기만 함(순서 정보 없음):", np.allclose(o1[perm], o2))

PE = positional_encoding(50, 64)
plt.imshow(PE, aspect="auto", cmap="RdBu"); plt.xlabel("dimension i"); plt.ylabel("position")
plt.colorbar(); plt.title("Positional encoding"); plt.show()
```

**체크포인트**
- attention weights 각 행의 합 = 1.
- causal mask: **위쪽 삼각형이 전부 0**인 하삼각 행렬. 첫 토큰은 자기 자신만 봐서 `[1, 0, 0, 0, 0]`.
- encoder 출력 shape `(5, 16)` — 입력과 같은 shape이라 층을 계속 쌓을 수 있다. layer norm 덕분에 토큰별 평균 ≈ 0.
- `True` — self-attention은 순서를 바꾸면 결과도 똑같이 섞일 뿐(permutation equivariant)이라 "Jane visits Africa"와 "Africa visits Jane"을 구분 못 한다. 그래서 positional encoding을 더해줘야 한다.
- PE 히트맵: 앞쪽 차원은 빠르게, 뒤쪽 차원은 느리게 진동하는 줄무늬 — 차원마다 주기가 다른 "위치 바코드".

<details>
<summary>정답 보기</summary>

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights

def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, mask=None):
    heads = []
    for h in range(n_heads):
        out, _ = scaled_dot_product_attention(X @ Wq[h], X @ Wk[h], X @ Wv[h], mask)
        heads.append(out)
    return np.concatenate(heads, axis=-1) @ Wo

def positional_encoding(T, d):
    pos = np.arange(T)[:, None]
    i = np.arange(d)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d)
    PE = np.zeros((T, d))
    PE[:, 0::2] = np.sin(angle[:, 0::2])
    PE[:, 1::2] = np.cos(angle[:, 1::2])
    return PE
```

</details>
