import numpy as np
import matplotlib.pyplot as plt
from utils import *

# ** 아래의 예제는 utils.py의 loaddata() 함수에서 명시한 파일이 없음. coursera 측의 오류인듯 -> K-means가 작동하는 코드 로직에 중점을 둘것!! 


# UNQ_C1
# GRADED FUNCTION: find_closest_centroids

def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example
    
    Args:
        X (ndarray): (m, n) Input values      
        centroids (ndarray): (K, n) centroids
    
    Returns:
        idx (array_like): (m,) closest centroids
    
    """
    # Set K
    K = centroids.shape[0]
    
    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)
    
    ### START CODE HERE ###
    for i in range(X.shape[0]):
        #  첫번째 데이터가 어디 centroids와 더 가까운지 계산
        shortest_centroids = float('inf')
        shortest_centroids_idx = 0
#         print(f'test X[{i}]: ', X[i])
        for j in range(K):
            cost = np.linalg.norm(centroids[j] - X[i])
#             print(f'test cost print: {cost}')
            if(shortest_centroids > cost ):
                shortest_centroids = cost
                shortest_centroids_idx = j
        idx[i] = shortest_centroids_idx    
        
     ### END CODE HERE ###
    
    return idx


# Load an example dataset that we will be using
X = load_data()

print("First five elements of X are:\n", X[:5]) 
print('The shape of X is:', X.shape)

# Select an initial set of centroids (3 Centroids)
initial_centroids = np.array([[3,3], [6,2], [8,5]])

# Find closest centroids using initial_centroids
idx = find_closest_centroids(X, initial_centroids)

# Print closest centroids for the first three elements
print("First three elements in idx are:", idx[:3])

# UNIT TEST
from public_tests import *

find_closest_centroids_test(find_closest_centroids)



# ------------------------------------------------------------------------------------------

# UNQ_C2
# GRADED FUNCTION: compute_centroids

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    
    # Useful variables
    m, n = X.shape # [300,2]
    
    # You need to return the following variables correctly
    centroids = np.zeros((K, n))
    
    ### START CODE HERE ###
    for i in range(K):
        arr = X[idx == i]
        # print(f'arr mean: {np.mean(arr, axis = 0)}')
        centroids[i] = np.mean(arr, axis = 0)
    ### END CODE HERE ## 
    
    return centroids

# 방식 B: NumPy 마스킹(Boolean Indexing) 활용 (파이썬다운 효율적인 방법)
# if문과 반복문 없이 NumPy의 기능을 이용하면 한 줄로 해당 클러스터의 데이터만 뽑아낼 수 있습니다.
# X[idx == k] 구문은 idx가 k인 X의 데이터 행들만 슬라이싱해 줍니다.
# 이 배열에 np.mean(..., axis=0)을 적용하면 해당 클러스터의 중심(평균)을 바로 계산할 수 있습니다.

# 원리 : 
# == 비교하며 원본 배열과 동일한 크기의 True/False 배열을 생성
# 생성된 불리언 배열을 데이터 배열 X의 인덱스로 전달합니다. 이때 NumPy는 True에 해당하는 위치의 행(Row)만 남기고, False 위치의 행은 버립니다.
# X의 모양이 (m, n)이고 idx의 모양이 (m,)일 때:
# NumPy는 불리언 배열의 개수(m)를 X의 첫 번째 차원(행의 개수 m)과 1:1로 대응시킵니다.
# 만약 해당 클러스터 k에 배정된 데이터가 하나도 없다면(idx == k 결과가 모두 False), X[idx == k]는 빈 배열이 되어 np.mean() 계산 시 NaN 경고가 발생할 수 있습니다.
# 따라서 실제 구현 시에는 선택된 데이터가 존재하는지(len(X[idx == k]) > 0) 확인하는 조건처리를 함께 고려하는 것이 좋습니다.





def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """
    
    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))

    # Run K-Means
    for i in range(max_iters):
        
        #Output progress
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)
        
        # Optionally plot progress
        if plot_progress:
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            
        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    plt.show() 
    return centroids, idx


# Load an example dataset
X = load_data()

# Set initial centroids
initial_centroids = np.array([[3,3],[6,2],[8,5]])

# Number of iterations
max_iters = 10

# Run K-Means
centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)




# K-means 첫 시작 시 아예 random한 곳에서 하는 방법도 있으나, 훈련 데이터들에서 하나를 골라서 해당 부분을 initialization point로 설정하고 하고 됨!  
def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be 
    used in K-Means on the dataset X
    
    Args:
        X (ndarray): Data points 
        K (int):     number of centroids/clusters
    
    Returns:
        centroids (ndarray): Initialized centroids
    """
    
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])  # permutaion : 순열.  (원본 배열과는 다른 랜덤한 순서의 배열을 만드ㄴㄴ 것 )
    
    # Take the first K examples as centroids
    centroids = X[randidx[:K]]
    
    return centroids


# Run this cell repeatedly to see different outcomes.

# Set number of centroids and max number of iterations
K = 3
max_iters = 10

# Set initial centroids by picking random examples from the dataset
initial_centroids = kMeans_init_centroids(X, K)

# Run K-Means
centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)