import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('C:/Users/User/Desktop/Machine learning exercise/data1/data/ex2data1.txt', header=None)
x = data.iloc[:, :-1].to_numpy()
y = data.iloc[:, -1].to_numpy().reshape(-1, 1)

# Feature Scaling (FIXED)
def feature_scaling(x):
    mu = np.mean(x, axis=0)        # column-wise mean
    std = np.std(x, axis=0)        # column-wise std
    x_norm = (x - mu) / std
    return x_norm, mu, std

x, x_mean, std = feature_scaling(x)

# Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Add bias term
m, n = x.shape
x = np.append(np.ones((m, 1)), x, axis=1)
theta = np.zeros((n + 1, 1))

# Cost Function (FIXED)
def cost_function(x, theta, y):
    m = len(y)
    z = np.dot(x, theta)
    h = sigmoid(z)
    # avoid log(0)
    h = np.clip(h, 1e-10, 1 - 1e-10)
    cost = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    grad = np.dot(x.T, (h - y)) / m
    return cost, grad

# Gradient Descent
def gradient_descent(x, y, theta, alpha, iterations):
    j_history = []
    for i in range(iterations):
        cost, grad = cost_function(x, theta, y)
        theta = theta - alpha * grad
        j_history.append(cost)
    return theta, j_history

theta, j_history = gradient_descent(x, y, theta, 0.1, 400)
print(theta)
print(j_history)

# Plot Decision Boundary (FIXED)
pos = (y == 1).ravel()
neg = (y == 0).ravel()
plt.scatter(x[pos, 1], x[pos, 2], c='r', marker='+', label='Admitted')
plt.scatter(x[neg, 1], x[neg, 2], c='b', marker='o', label='Not Admitted')
x_value = np.array([np.min(x[:,1]), np.max(x[:,1])])
y_value = -(theta[0] + theta[1]*x_value) / theta[2]
plt.plot(x_value, y_value, 'g')
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend()
plt.show()

x_test=np.array([45,85])
x_test=(x_test - x_mean) / np.std(x_test, axis=0)
x_test=np.append(np.ones(1),x_test)
x_test=x_test.reshape(3,1)
print(x_test)
print(x_test.shape)

z = np.dot(theta.T, x_test)
h = sigmoid(z)
probability = h.item() * 100 # scalar ပြောင်း
print(f"For a student with scores 45 & 85, we predict an admission probability of: {probability:.2f}%")

def predict(theta,x):
    prd = x.dot(theta)

    return prd > 0
    # return np.argmax(prd)
    # return prd > probability

p = predict(theta,x)
# print(p)
print(p.shape)
print(y.shape)

#checking
result=sum(p==y)
print("Training Accuracy:", result[0],"%")

