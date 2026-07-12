import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import zscore_normalize_features, run_gradient_descent_feng
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

'''
    What if your features/data are non-linear or are combinations of features?
    For example, Housing prices do not tend to be linear with living area but penalize very small or very large houses resulting in the curves shown in the graphic above. 
    How can we use the machinery of linear regression to fit this curve? 
    Recall, the 'machinery' we have is the ability to modify the parameters  𝐰 ,  𝐛  in (1) to 'fit' the equation to the training data. 
    However, no amount of adjusting of  𝐰 , 𝐛  in (1) will achieve a fit to a non-linear curve.
'''

# create target data
x = np.arange(0, 20, 1) # 0~19까지의 배열 
y = 1 + x**2
X = x.reshape(-1, 1) # 열을 1행으로 맞추는 것 -> 행은 20행 20 * 1 열벡터 

model_w,model_b = run_gradient_descent_feng(X,y,iterations=1000, alpha = 1e-2)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("no feature engineering")
plt.plot(x,X@model_w + model_b, label="Predicted Value");  plt.xlabel("X"); plt.ylabel("y"); 
plt.legend(); 
plt.show()


"""
    𝑦=𝑤0𝑥20+𝑏 , or a polynomial feature. 
    To accomplish this, you can modify the input data to engineer the needed features. 
    If you swap the original data with a version that squares the  𝑥  value, then you can achieve  𝑦=𝑤0𝑥20+𝑏 . Let's try it. 
"""

# create target data
x = np.arange(0, 20, 1)
y = 1 + x**2

# Engineer features 
X = x**2      #<-- added engineered feature
X = X.reshape(-1, 1)  #X should be a 2-D Matrix
model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha = 1e-5)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Added x**2 feature")
plt.plot(x, np.dot(X,model_w) + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()


"""
    Above, we knew that an  𝑥2  term was required. It may not always be obvious which features are required. One could add a variety of potential features to try and find the most useful. 
    For example, what if we had instead tried :  𝑦=𝑤0𝑥0+𝑤1𝑥21+𝑤2𝑥32+𝑏  ?
"""
# create target data
x = np.arange(0, 20, 1)
y = x**2

# engineer features .
X = np.c_[x, x**2, x**3]   #<-- added engineered feature

model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha=1e-7)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("x, x**2, x**3 features")
plt.plot(x, X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()


# create target data
x = np.arange(0, 20, 1)
y = x**2

# engineer features .
X = np.c_[x, x**2, x**3]   #<-- added engineered feature
X_features = ['x','x^2','x^3']

fig,ax=plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X[:,i],y)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("y")
plt.show()


#Scaling Features
"""
    if the data set has features with significantly different scales, one should apply feature scaling to speed gradient descent. 
    In the example above, there is  𝑥 ,  𝑥2  and  𝑥3  which will naturally have very different scales.
    현재 x는 0 ~ 19고 x^3은 0 ~ 6859 => 매우 다른 scale 상태 => 정규화 ! 
"""
# create target data
x = np.arange(0,20,1)
X = np.c_[x, x**2, x**3]
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X,axis=0)}")

# add mean_normalization 
X = zscore_normalize_features(X)     
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X,axis=0)}")

x = np.arange(0,20,1)
y = x**2

X = np.c_[x, x**2, x**3]
X = zscore_normalize_features(X) 

model_w, model_b = run_gradient_descent_feng(X, y, iterations=100000, alpha=1e-1) # 정규화를 통해, 훨씬 더 공격적인(큰) learning rate를 적용 가능하다 

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show() # numpy에서 @는 행렬의 곱셈을 뜻함 


# Complex Functions
x = np.arange(0,20,1)
y = np.cos(x/2)

X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
X = zscore_normalize_features(X) 

model_w,model_b = run_gradient_descent_feng(X, y, iterations=1000000, alpha = 1e-1)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
