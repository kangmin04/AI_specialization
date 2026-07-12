import numpy as np
a = np.array([[1], [2], [3]])
print(a.shape)
print(a)

arr1d = np.array([10, 20, 30, 40])
print(arr1d.shape)  # 결과: (4,) -> 데이터가 4개 1줄로 있음

arr2d = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print(arr2d.shape)  # 결과: (2, 3) -> 2개의 행, 각 행마다 3개의 열


arr3d = np.array([
    [[1, 2], [3, 4]],   # 첫 번째 2차원 면 (Page 0)
    [[5, 6], [7, 8]],   # 두 번째 2차원 면 (Page 1)
    [[9, 0], [1, 2]]    # 세 번째 2차원 면 (Page 2)
])
print(arr3d.shape)  # 결과: (3, 2, 2) -> 2x2 짜리 면이 총 3장 겹쳐 있음


# 4차원 배열 생성 (예: 2x2x2 짜리 3차원 입체가 총 2개 묶여 있는 상태)
arr4d = np.array([
    [ [[1, 2], [3, 4]], [[5, 6], [7, 8]] ],  # 3차원 큐브 0번
    [ [[9, 1], [2, 3]], [[4, 5], [6, 7]] ]   # 3차원 큐브 1번
])
print(arr4d.shape)  # 결과: (2, 2, 2, 2)