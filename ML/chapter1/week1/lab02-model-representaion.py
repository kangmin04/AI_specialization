import numpy as np
import matplotlib.pyplot as plt
plt.style.use('/Users/kangmin/Desktop/personal-study/AI/ML/chapter1/deeplearning.mplstyle')

#Lab2. Linear Regression with one variable. 

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")

# m is the number of training examples
m = len(x_train)
print(f"Number of training examples is: {m}")

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='b') 
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()


w = 200
b = 100
print(f"w: {w}")
print(f"b: {b}")


def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model(한 변수를 대상으로 하는 선형 회귀 모델 )
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m) # m개의 0으로 가득찬 배열(벡터)를 만듦 
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb
    
# Note: The argument description (ndarray (m,)) describes a Numpy n-dimensional array of shape (m,). 
# (scalar) describes an argument without dimensions, just a magnitude.
# Note: np.zero(n) will return a one-dimensional numpy array with  𝑛  entries
 
tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()


w = 200                         
b = 100    
x_i = 1.2
cost_1200sqft = w * x_i + b    

print(f"${cost_1200sqft:.0f} thousand dollars")