import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

RANDOM_STATE = 55

# ----- ReLU 함수 직접 구현 -----
# ReLU(z) = max(0, z). 음수는 다 0으로 죽이고 양수는 그대로 통과시키는 아주 단순한 함수.
def my_relu(z):
    return np.maximum(0, z)

z_example = np.array([-3., -1., 0., 2., 5.])
print("z:", z_example)
print("relu(z):", my_relu(z_example))

# ----- sigmoid와 비교 -----
# sigmoid는 z가 커지거나 작아지면 기울기가 거의 0이 되어버림(saturate) -> 학습이 느려짐(gradient vanishing).
# relu는 z>0 구간에서 기울기가 항상 1이라 학습이 더 잘 되는 경우가 많아서 hidden layer에서 기본값처럼 쓰임.
def my_sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 200)
plt.plot(z, my_sigmoid(z), label='sigmoid')
plt.plot(z, my_relu(z), label='relu')
plt.title("sigmoid vs relu")
plt.xlabel("z")
plt.ylabel("activation(z)")
plt.legend()
plt.show()

# ----- activation function이 왜 필요한가: linear만 쌓으면 결국 linear함 -----
# Dense layer를 activation='linear'로만 여러 개 쌓으면, 아무리 layer가 많아져도
# 전체 네트워크는 결국 하나의 linear function(W_eff @ x + b_eff)과 동일해짐.
# activation(비선형 함수)이 있어야 layer를 쌓는 의미가 생김.
linear_only_model = Sequential([
    Dense(units=4, activation='linear'),
    Dense(units=3, activation='linear'),
    Dense(units=1, activation='linear'),
])

X_demo = np.random.default_rng(RANDOM_STATE).normal(size=(5, 2))
linear_only_model.predict(X_demo)  # 최초 예측을 한 번 해줘야 weight가 초기화됨(build)

W1, b1 = linear_only_model.layers[0].get_weights()
W2, b2 = linear_only_model.layers[1].get_weights()
W3, b3 = linear_only_model.layers[2].get_weights()

# 3개 layer를 하나로 합친 "유효(effective)" 가중치/절편
W_eff = W1 @ W2 @ W3
b_eff = b1 @ W2 @ W3 + b2 @ W3 + b3

pred_stacked = linear_only_model.predict(X_demo)
pred_collapsed = X_demo @ W_eff + b_eff

print("\n[Prediction by only using linear model]:\n", pred_stacked)
print("[Linear regression combined ]:\n", pred_collapsed)
print("-> 두 결과가 (거의) 같음. linear activation만으로는 layer를 아무리 쌓아도 표현력이 늘지 않음.")

# ----- ReLU로 non-linear(꺾인) 함수 근사하기 -----
# ReLU 유닛 여러 개를 합치면 꺾인 지점(kink)이 여러 개인 piecewise-linear 함수를 표현할 수 있음.
# x=1에서 기울기가 바뀌고, x=4에서 한 번 더 바뀌는 함수를 만들어서 근사해봄.
def target_function(x):
    y = np.where(x < 1, 0.0, x - 1.0)
    y = np.where(x < 4, y, y + 2.0 * (x - 4.0))
    return y

x_train = np.linspace(-2, 8, 200).reshape(-1, 1)
y_train = target_function(x_train.flatten())

relu_model = Sequential([
    Dense(units=8, activation='relu'),
    Dense(units=1, activation='linear'),
])

relu_model.compile(
    loss=MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(0.05),
)

relu_model.fit(x_train, y_train, epochs=200, verbose=0)

y_pred = relu_model.predict(x_train).flatten()

plt.figure()
plt.plot(x_train, y_train, label='target (piecewise-linear)')
plt.plot(x_train, y_pred, label='relu network의 근사')
plt.title("Using relu to show non linear plot")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

print("\n[relu 근사] target 첫 5개:", y_train[:5])
print("[relu 근사] 예측 첫 5개:", y_pred[:5])
