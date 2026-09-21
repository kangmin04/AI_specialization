# Week2 핵심 정리: 추천 시스템 개요 + PCA

Course 3(Unsupervised Learning, Recommenders, Reinforcement Learning) week2는 원래 추천 시스템(collaborative filtering, content-based filtering)과 PCA를 같이 다루는데, 시간이 없어서 강의 대신 이 문서로 핵심만 훑는다. 추천 시스템은 짧게 개요만 잡고, PCA를 제대로 정리하는 게 목표다.

## 목차

1. 추천 시스템 개요 (짧게)
2. PCA는 왜 필요한가
3. 직관: 데이터가 가장 "퍼지는" 축 찾기
4. 전처리: feature scaling / mean normalization
5. principal component 고르는 기준: 분산 최대화 = 투영 오차 최소화
6. 여러 개의 principal component (서로 직교)
7. 재구성 (reconstruction / approximation)
8. explained variance ratio로 k 고르기
9. PCA vs 선형 회귀 — 흔한 오해
10. 실전 조언 (Andrew Ng가 강조한 포인트)
11. scikit-learn 코드 예시
12. 핵심 요약
13. 헷갈리기 쉬운 점

---

## 1. 추천 시스템 개요 (짧게)

- **Collaborative filtering**: 유저들의 평점(rating) 패턴 자체로부터 학습한다. "비슷한 유저는 비슷한 아이템을 좋아한다" / "이 유저가 준 평점 패턴을 재현하는 유저 벡터 $w^{(j)}$와 아이템 벡터 $x^{(i)}$를 찾자"는 아이디어다. 비용 함수는 대략:
  $$J = \sum_{(i,j):r(i,j)=1} \left( w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)} \right)^2 + \text{정규화항}$$
  여기서 $r(i,j)=1$은 "유저 $j$가 아이템 $i$에 실제로 평점을 매겼다"는 뜻이고, 합은 이렇게 평점이 존재하는 쌍에 대해서만 계산한다(평점이 없는 칸은 학습에서 아예 빠진다). 아이템에 대한 별도 feature(장르, 설명 등)가 없어도, 평점 데이터만 충분히 많으면 작동한다. 다만 신규 유저/아이템처럼 평점이 거의 없는 경우("cold start")에 약하다.
- **Content-based filtering**: 유저 profile feature $x_u$와 아이템 feature $x_m$을 각각 신경망(neural network)에 넣어서 임베딩 벡터를 만들고, 두 벡터의 내적(dot product)으로 평점/선호도를 예측한다. 비용 함수는 collaborative filtering이랑 비슷한 squared error 형태를 쓰지만, 입력이 "평점 행렬"이 아니라 "유저/아이템의 실제 feature"라는 게 핵심 차이다. 유저나 아이템에 대한 부가 정보(feature)가 있을 때, 그리고 cold start 상황에서 더 유리하다.
- 언제 쓰나: 데이터가 (유저, 아이템, 평점) 형태의 상호작용 로그 위주면 collaborative filtering, 유저/아이템 각각의 속성 정보가 풍부하면 content-based filtering 쪽이 자연스럽다. 실무에서는 둘을 섞어 쓰는 경우도 많다.

여기까지가 개요고, 이 문서의 나머지는 전부 PCA다.

---

## 2. PCA는 왜 필요한가

feature가 2개짜리인 데이터는 그냥 산점도(scatter plot)로 눈으로 보면 된다. 그런데 feature가 50개, 100개면? 사람이 50차원 그래프를 볼 수는 없다. 그래서 고차원 데이터를 2차원이나 3차원으로 줄여서 "눈으로 보기 위한" 용도로 PCA를 쓴다.

또 하나의 동기는 feature 개수 자체를 줄이는 것(dimensionality reduction)이다. 예를 들어 자동차 데이터에 "길이(length)"와 "너비(width)" 두 feature가 있다고 하자. 그런데 대부분의 자동차는 길이가 길수록 너비도 비례해서 커지는 경향이 있다 — 즉 이 두 feature가 거의 하나의 축을 따라 늘어서 있다는 거다. 이럴 때는 "길이+너비를 합친 크기(size)" 같은 새로운 축 하나로 정보 손실을 크게 주지 않고 표현할 수 있다. 2개 feature를 1개로 줄이는 것, 이게 PCA가 하는 일의 가장 단순한 예시다.

수학적 세부사항(고유값 분해, SVD 등)을 몰라도 PCA를 쓰는 데는 전혀 지장 없다. 아이디어와 언제/어떻게 쓰는지만 알면 된다.

---

## 3. 직관: 데이터가 가장 "퍼지는" 축 찾기

PCA의 핵심 아이디어는 이거다: 데이터를 어떤 축(axis)에 투영(projection)했을 때, 그 축 위에서 데이터가 **가장 넓게 퍼지도록** 축을 고른다.

왜 "퍼지는 정도"가 중요할까? 데이터가 넓게 퍼져 있다는 건 그 축이 데이터의 변화(variance)를 많이 담고 있다는 뜻이고, 반대로 좁게 뭉쳐 있으면 그 축으로는 데이터를 구분할 정보가 거의 없다는 뜻이다. 즉 정보를 가장 적게 잃는 방향을 찾는 거다.

위의 자동차 예시로 다시 보면: 길이-너비 산점도에서 점들이 대각선 방향으로 길게 늘어서 있다면, 그 대각선 방향이 바로 데이터가 가장 많이 퍼지는 축이다. 그 축을 새로운 좌표축(첫 번째 principal component, $z_1$)으로 삼고 각 데이터 포인트를 그 축 위로 투영시키면, 원래 2차원이었던 정보를 1차원 숫자 하나로 거의 손실 없이 표현할 수 있다.

반대로 그 대각선에 수직인 방향으로 투영하면 점들이 거의 한 점에 뭉쳐버린다 — 이 방향은 정보가 별로 없다는 뜻이므로 PCA는 이 방향을 고르지 않는다.

---

## 4. 전처리: feature scaling / mean normalization

PCA를 돌리기 전에 반드시 해야 하는 전처리가 있다.

- **Mean normalization**: 각 feature에서 그 feature의 평균을 빼서, 데이터의 중심이 원점(origin)에 오도록 만든다. PCA는 원점을 기준으로 축을 찾기 때문에, 이 작업 없이는 축이 엉뚱하게 잡힌다.
- **Feature scaling**: feature들의 스케일(단위)이 크게 다르면(예: 집 크기는 수백~수천 단위, 방 개수는 1~5 단위) 스케일이 큰 feature가 "분산이 크다"는 이유만으로 PCA에서 과도하게 중요하게 취급된다. 그래서 보통 표준편차로 나누는 정규화(standardization)를 같이 해준다.

정리하면 각 feature $x_j$에 대해
$$x_j^{(i)} \leftarrow \frac{x_j^{(i)} - \mu_j}{\sigma_j}$$
를 적용한 뒤에 PCA를 돌리는 게 표준적인 흐름이다. (linear regression에서 gradient descent 전에 하던 feature scaling과 똑같은 이유다.)

---

## 5. principal component 고르는 기준: 분산 최대화 = 투영 오차 최소화

"데이터를 가장 넓게 퍼지게 하는 축을 고른다"는 말을 조금 더 formal하게 쓰면: 각 데이터 포인트 $x^{(i)}$를 어떤 단위 벡터(unit vector) 축에 투영했을 때, 투영된 값들의 **분산(variance)을 최대화**하는 축을 찾는 것이다.

재밌는 점은, 이게 다른 관점에서 보면 **투영 오차(projection error)를 최소화**하는 것과 동치라는 거다. 투영 오차란 원래 점에서 그 점을 축 위로 내렸을 때(수직으로 투영했을 때)까지의 거리를 말한다. 분산을 최대화하는 축을 고르면 자연스럽게 점들이 축에서 덜 떨어지게 되고, 결과적으로 투영 오차도 최소가 된다. 즉 두 목표(분산 최대화 / 오차 최소화)는 같은 축을 가리킨다 — 이걸 굳이 증명까지 알 필요는 없고, "같은 결과를 낸다"는 사실만 기억하면 된다.

이렇게 찾은 첫 번째 축을 **첫 번째 principal component**($z_1$ 축)이라고 부른다.

---

## 6. 여러 개의 principal component (서로 직교)

축을 하나만 쓰는 게 아니라 여러 개 쓸 수도 있다. 두 번째 principal component를 고를 때는, 첫 번째 축과 **직교(orthogonal, 수직)**하면서 그 다음으로 분산을 가장 많이 설명하는 축을 고른다. 세 번째, 네 번째도 마찬가지로 이전 축들과 모두 직교하는 조건 하에서 남은 분산을 가장 많이 설명하는 방향을 순서대로 찾아나간다.

그래서 $n$차원 데이터에서 최대 $n$개의 principal component를 뽑을 수 있고, 이 중 처음 몇 개($k$개, $k < n$)만 골라 쓰면 그게 바로 차원 축소(dimensionality reduction)다. 원래 $n$개 feature였던 데이터가 $k$개의 새로운 축(z_1, z_2, ..., z_k) 값으로 표현되는 것이다.

---

## 7. 재구성 (reconstruction / approximation)

차원을 줄였다는 건 정보를 버렸다는 뜻이다. 그런데 PCA는 줄인 $z$ 값들로부터 원래 차원의 근사값을 다시 만들어낼 수 있다 — 이걸 **reconstruction(재구성)**이라고 한다.

예를 들어 2차원 $(x_1, x_2)$를 1차원 $z_1$으로 줄였다면, $z_1$에 다시 그 축의 방향 벡터를 곱해서 2차원 공간으로 되돌릴 수 있다. 단, 이건 완벽한 복원이 아니라 **근사(approximation)**다 — 애초에 축에 수직인 방향의 정보(투영 오차만큼)는 버려졌기 때문에, 재구성된 점은 원래 점과 정확히 같지 않고 그 점이 투영됐던 축 위의 위치로 돌아갈 뿐이다.

한 가지 주의할 점: 이 재구성은 PCA가 실제로 작업했던 공간, 즉 4번에서 mean normalization/scaling을 적용한 뒤의 공간에서 이루어진다. 그래서 재구성된 값은 그 자체로는 "정규화된 스케일"의 근사값이고, 원래 단위(예: 실제 집 크기, 실제 방 개수)로 다시 보고 싶다면 scaling을 적용했던 순서의 역순으로(표준편차를 곱하고 평균을 더해서) 한 번 더 되돌려줘야 한다.

reconstruction은 "이 정도 차원까지 줄여도 원본 정보를 얼마나 잘 유지하는지" 감을 잡는 데 유용하다. 예를 들어 이미지 데이터에 PCA를 적용해서 차원을 확 줄인 뒤 다시 복원해보면, k가 너무 작을 경우 이미지가 뭉개져 보이는 걸 눈으로 확인할 수 있다.

---

## 8. explained variance ratio로 k 고르기

그러면 $k$(몇 개의 principal component를 쓸지)는 어떻게 정할까? 기준은 **"원본 데이터의 분산을 몇 %나 유지하고 싶은가"**다.

각 principal component가 전체 분산 중 얼마를 설명하는지를 **explained variance ratio**라고 부른다. 이 값들을 첫 번째 component부터 누적해서 더해보고, 누적 비율이 원하는 임계치(예: 95%, 99%)를 넘는 최소 $k$를 고르면 된다.

예를 들어 explained variance ratio가 [0.7, 0.2, 0.05, 0.03, ...] 이라면, $k=2$만 써도 90%(0.7+0.2)의 분산을 유지하는 셈이다. "99%의 분산을 유지하는 최소 차원"처럼 기준을 정해두고 그걸 만족하는 $k$를 자동으로 찾는 방식이 실무에서 흔하다.

---

## 9. PCA vs 선형 회귀 — 흔한 오해

PCA를 처음 보면 선형 회귀(linear regression)랑 비슷해 보인다 — 둘 다 "가장 잘 맞는 직선/평면을 찾는다"는 느낌이라서. 하지만 완전히 다른 문제다.

- **선형 회귀**: $x$로부터 정답 label $y$를 예측하는 게 목적이다. 오차는 **수직 방향(y축 방향)**으로만 잰다 — 즉 $y$가 특별한 존재이고, 나머지 feature들은 $y$를 예측하기 위한 입력일 뿐이다.
- **PCA**: 정답 label 같은 게 없다(unsupervised). 모든 feature를 동등하게 취급하고, 점에서 축까지의 오차를 **직교 방향(수직으로 내린 최단 거리)**으로 잰다. 어떤 feature도 "특별한 target"이 아니다.

그래서 그림으로 그려보면 선형 회귀의 fitted line과 PCA의 첫 번째 principal component 축이 비슷해 보일 수 있어도, 오차를 재는 방향 자체가 다르기 때문에 일반적으로 두 직선은 서로 다르다. "PCA = 선형 회귀의 unsupervised 버전"이라고 오해하기 쉬운데, 목적과 오차 정의가 근본적으로 다르다는 걸 기억해두자.

---

## 10. 실전 조언 (Andrew Ng가 강조한 포인트)

- PCA는 요즘(딥러닝 시대) 실무에서는 예전만큼 많이 쓰이지 않는다. 예전에는 데이터 압축(compression)이나 학습 속도 향상(feature 수를 줄여서 학습을 빠르게 하는 것) 목적으로 자주 쓰였지만, 지금은 저장 공간과 연산 능력이 훨씬 좋아져서 그런 용도로는 잘 안 쓴다.
- **과적합(overfitting) 방지 목적으로 PCA를 쓰지 말 것.** feature 수를 줄이면 과적합이 줄어들 것 같다는 생각이 들 수 있지만, PCA는 label $y$를 전혀 보지 않고 오직 $x$만 보고 축소를 하기 때문에 예측에 중요한 정보까지 날려버릴 위험이 있다. 과적합 방지가 목적이면 regularization(정규화)을 쓰는 게 훨씬 낫다.
- 오늘날 PCA가 여전히 유용한 대표적인 용도는 **시각화(visualization)**다. 고차원 데이터를 2D/3D로 줄여서 산점도로 찍어보고, 데이터에 어떤 군집이나 패턴이 있는지 사람이 눈으로 확인하는 용도로는 지금도 잘 쓰인다.
- 모델링을 시작하기 전에 "일단 PCA부터 돌려보자"는 습관보다는, 먼저 원본 feature로 시도해보고 필요할 때(예: 시각화가 필요하거나 feature가 극단적으로 많을 때) PCA를 고려하는 순서를 권장한다.

---

## 11. scikit-learn 코드 예시

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1) 전처리: mean normalization + feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # X: (m, n) 원본 데이터

# 2) PCA 학습 + 변환 (k=2로 축소, 시각화 목적)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)  # (m, 2)

# 3) 각 principal component가 설명하는 분산 비율 확인
print(pca.explained_variance_ratio_)      # 예: [0.72, 0.15]
print(sum(pca.explained_variance_ratio_)) # 누적 설명 비율

# 4) 재구성 (PCA가 실제로 작업한 "스케일링된 공간"으로 근사 복원)
X_reconstructed_scaled = pca.inverse_transform(X_reduced)  # (m, n), X_scaled의 근사값 (원본 X의 근사값이 아님!)

# 5) 원래 단위로 보고 싶으면 scaling도 역변환까지 해줘야 진짜 원본 X의 근사값이 된다
X_reconstructed = scaler.inverse_transform(X_reconstructed_scaled)  # (m, n), 원본 X의 근사값
```

`n_components`를 정수 대신 `0.95`처럼 넣으면, explained variance 누적이 95%를 넘는 최소 개수의 component를 자동으로 골라주기도 한다.

(위 4)/5) 두 단계를 헷갈리기 쉬운데, `pca.inverse_transform`은 PCA가 지운 차원만 되돌려줄 뿐 `StandardScaler`가 했던 스케일링까지 되돌려주지는 않는다. 원본과 직접 비교하려면 반드시 `scaler.inverse_transform`까지 거쳐야 한다.)

---

## 12. 핵심 요약

- PCA는 고차원 데이터를 사람이 볼 수 있는 저차원(주로 2D/3D)으로 줄여서 **시각화**하거나, feature 개수를 줄이는 데 쓰는 unsupervised 기법이다.
- 핵심 아이디어는 데이터를 투영했을 때 **분산이 최대**가 되는 축(principal component)을 찾는 것 — 이는 투영 오차를 최소화하는 것과 동치다.
- PCA 전에는 반드시 **mean normalization + feature scaling**을 해줘야 한다.
- 여러 개의 principal component는 서로 **직교(orthogonal)**하도록 순서대로 뽑는다.
- 축소한 데이터는 **reconstruction**으로 원본 차원에 근사 복원할 수 있지만 완벽히 같지는 않다.
- $k$는 **explained variance ratio**의 누적값이 원하는 임계치(예: 95%)를 넘는 최소값으로 정한다.
- PCA와 선형 회귀는 비슷해 보이지만 목적(예측 vs 표현)과 오차 방향(수직 vs 직교)이 다른 별개의 기법이다.
- 요즘은 압축/학습 가속보다는 **시각화** 용도로 주로 쓰이고, **과적합 방지 목적으로는 쓰지 않는 게** 좋다 — 그건 regularization의 역할이다.

## 13. 헷갈리기 쉬운 점

- **"PCA가 자동으로 유의미한 feature를 골라준다"는 오해**: PCA가 만드는 새 축($z_1, z_2, ...$)은 원본 feature 중 하나를 고르는 게 아니라, 여러 feature를 섞은(선형결합) 완전히 새로운 값이다. 그래서 PCA 후에는 "이 축이 정확히 무슨 의미인지" 해석이 어려워질 수 있다.
- **label을 본다고 착각하기 쉬움**: PCA는 $y$(정답)를 전혀 사용하지 않는 완전한 unsupervised 기법이다. 그래서 "예측에 도움되는 방향"이 아니라 "데이터 자체의 변화가 큰 방향"을 찾는다는 점을 잊으면 안 된다 — 이게 과적합 방지용으로 쓰면 안 되는 이유이기도 하다.
- **PCA ≠ 선형 회귀** (9번 섹션 참고): 오차를 재는 방향이 다르다는 점, target $y$의 존재 여부가 다르다는 점을 혼동하지 말 것.
- **분산 최대화와 오차 최소화가 "우연히 같은 결과"라고 생각하기 쉬운데**, 사실 수학적으로 정확히 동치인 두 관점일 뿐이다. 어느 쪽으로 설명을 들어도 같은 축을 가리킨다.
- **scaling을 빼먹는 실수**: feature 단위 차이가 크면 스케일이 큰 feature가 부당하게 "분산이 크다"고 여겨져서 principal component가 왜곡된다. PCA 전 전처리는 선택이 아니라 필수에 가깝다.
- **reconstruction이 원본을 완벽히 복원해줄 거라는 오해**: reconstruction은 어디까지나 근사(approximation)다. $k$를 원본 차원 $n$보다 작게 잡은 이상 정보 손실은 필연적이다.
