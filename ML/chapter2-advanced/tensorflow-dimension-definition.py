import numpy as np

# 기존 넘파이에서 정의하던 방식 
a = np.array([200,17]) # 1-D.  행/열 개념이 없는 1차원 배열임 
print(a)
print(a.shape) # (2, ) -> 요소 2개를 가지는 1차원 배열 

b = a.reshape(-1,1)
print(b)
print(b.shape)



# Tensorflow에서 정의하는 방식 

x = np.array([[200,17]]) # [[]]을 두번 사용함! . 1 * 2 행벡터 
print(x)
print(x.shape)

x = np.array([[200], [17]]) # 2 * 1 열벡터 
print(x)
print(x.shape)

#allowing empty commit sorry mom and dad