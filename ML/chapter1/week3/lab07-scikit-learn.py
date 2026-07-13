import numpy as np

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])


from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X, y)
"""
    fit: 기존에 10,000번씩 돌리던 그 run_gradient_descent를 대신함. 
        사이킷런 내부에서 자동으로 데이터 스케일링, 손실 함수 최적화, 최적의 w와 b 탐색을 눈 깜짝할 사이에 끝냄. 
        학습이 끝나면 lr_model 안에는 최적의 w와 b가 저장됨
"""


y_pred = lr_model.predict(X)

print("Prediction on training set:", y_pred)

print("Accuracy on training set:", lr_model.score(X, y))

print(f"w: {lr_model.coef_}")
"""
사이킷런은 파이썬에서 머신러닝을 할 때 가장 널리 쓰이는 표준 라이브러리. 
왜 쓰나요?: 우리가 직접 짰던 run_gradient_descent 함수, 손실 함수 미분, 시그모이드 연산 등을 이미 수학적으로 가장 완벽하고 빠르게 최적화해서 만들어 둠
일관된 철학: 사이킷런은 모든 알고리즘이 fit()(학습해라)과 predict()(예측해라)라는 동일한 인터페이스를 가집니다. 
          그래서 로지스틱 회귀를 배우다가 다른 모델(SVM, 랜덤 포레스트 등)로 넘어가도 코드가 거의 똑같습니다.
"""