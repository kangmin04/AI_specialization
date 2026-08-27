import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.datasets import make_blobs

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

RANDOM_STATE = 55

# ----- softmax 함수 직접 구현 -----
# 각 클래스의 score(z)를 확률 분포로 바꿔주는 함수. 다 더하면 1이 되어야 함.
def my_softmax(z):
    ez = np.exp(z)
    sm = ez / np.sum(ez)
    return sm

z_example = np.array([1., 2., 3., 4.])
a_example = my_softmax(z_example)
print("z:", z_example)
print("softmax(z):", a_example, " -> 합:", np.sum(a_example))

# ----- 다중분류용 예제 데이터 생성 (클래스 4개짜리 2D blob) -----
classes = 4
m = 2000
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=m, centers=centers, cluster_std=1.0, random_state=RANDOM_STATE)
# make_blobs: 클러스터링 데스트를 위해 군집을 만드는 함수. 
#centers = cluster's centroids 위치
#cluster_std = 분산정도. (make_blobs는 가우시안 정규분포를 따름.)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis')
plt.title("softmax 예제 데이터 (4-class blobs)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# ----- 기본 방식: 마지막 레이어에 softmax activation 사용 -----
model = Sequential([
    Dense(units=25, activation='relu'),
    Dense(units=15, activation='relu'),
    Dense(units=4, activation='softmax')
])

model.compile(
    loss=SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(X_train, y_train, epochs=10)

p_nonpreferred = model.predict(X_train)
print("\n[기본 방식] 예측 확률 예시:", p_nonpreferred[:2])
print("[기본 방식] 확률 합:", np.sum(p_nonpreferred[0]))
print("[기본 방식] 예측 클래스:", np.argmax(p_nonpreferred[0]))

# ----- 개선된(수치적으로 안정적인) 방식 -----
# 마지막 레이어를 linear로 두고 loss에서 from_logits=True로 softmax를 대신 계산하게 하면
# 지수함수 계산이 내부적으로 더 안정적으로 처리됨(Andrew Ng 강좌 권장 패턴).
# 대신 model.predict()의 출력은 확률이 아니라 logit이므로, 마지막에 tf.nn.softmax()를 따로 적용해야 함.
preferred_model = Sequential([
    Dense(units=25, activation='relu'),
    Dense(units=15, activation='relu'),
    Dense(units=4, activation='linear')
])

preferred_model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

preferred_model.fit(X_train, y_train, epochs=10)

p_preferred = preferred_model.predict(X_train)
print("\n[개선 방식] 학습 직후 출력은 확률이 아니라 logit:", p_preferred[:2])

sm_preferred = tf.nn.softmax(p_preferred).numpy()
print("[개선 방식] softmax 적용 후 확률:", sm_preferred[:2])
print("[개선 방식] 확률 합:", np.sum(sm_preferred[0]))
print("[개선 방식] 예측 클래스:", np.argmax(sm_preferred[0]))

# ----- 두 방식 비교 -----
print("\n[비교] 기본 방식 예측 클래스:", np.argmax(p_nonpreferred, axis=1)[:10])
print("[비교] 개선 방식 예측 클래스:", np.argmax(sm_preferred, axis=1)[:10])
