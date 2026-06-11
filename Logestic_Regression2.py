import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('C:/Users/User/Desktop/Machine learning exercise/data1/data/ex2data2.txt')

x_original = df.iloc[:, :-1].to_numpy()
y = df.iloc[:, -1].to_numpy()

# reshape y
y = y.reshape(-1, 1)

# positive / negative
pos = (y == 1).reshape(-1)
neg = (y == 0).reshape(-1)

# plot data
plt.scatter(x_original[pos, 0], x_original[pos, 1], c="r", marker="+", label="Accepted")
plt.scatter(x_original[neg, 0], x_original[neg, 1], c="b", marker="x", label="Rejected")


# =========================
# Feature Mapping
# =========================
def mapFeature(x1, x2, degree):
    out = np.ones((len(x1), 1))
    for i in range(1, degree + 1):
        for j in range(i + 1):
            term = (x1 ** (i - j) * x2 ** j).reshape(-1, 1)
            out = np.hstack((out, term))
    return out


# map features
x = mapFeature(x_original[:, 0], x_original[:, 1], 6)


# =========================
# Sigmoid
# =========================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# =========================
# Cost Function + Gradient
# =========================
def costFunctionRe(theta, x, y, Lambda):
    m = len(y)

    h = sigmoid(x @ theta)

    # cost
    cost = (-y * np.log(h)) - ((1 - y) * np.log(1 - h))
    cost = np.sum(cost) / m

    # regularization
    reg = (Lambda / (2 * m)) * np.sum(theta[1:] ** 2)

    J = cost + reg

    # gradient
    grad = (1 / m) * (x.T @ (h - y))
    grad[1:] = grad[1:] + (Lambda / m) * theta[1:]

    return J, grad


# =========================
# Gradient Descent
# =========================
def gradientDescentRe(theta, x, y, alpha, iterations, Lambda):
    j_history = []

    for i in range(iterations):
        cost, grad = costFunctionRe(theta, x, y, Lambda)
        theta = theta - alpha * grad
        j_history.append(cost)

    return theta, j_history


# =========================
# Training
# =========================
theta = np.zeros((x.shape[1], 1))

theta, j_history = gradientDescentRe(theta, x, y, alpha=1, iterations=800, Lambda=1)

print("Theta:\n", theta)
print("Shape:", theta.shape)


# =========================
# Decision Boundary
# =========================
u_vals = np.linspace(-1, 1.5, 50)
v_vals = np.linspace(-1, 1.5, 50)

z = np.zeros((len(u_vals), len(v_vals)))
for i in range(len(u_vals)):
    for j in range(len(v_vals)):
        mapped = mapFeature(np.array([u_vals[i]]), np.array([v_vals[j]]), 6)
        z[i, j] = (mapped @ theta)[0, 0]

# contour
plt.contour(u_vals, v_vals, z.T, levels=[0])

plt.xlabel("Test 1")
plt.ylabel("Test 2")
plt.legend(loc=0)
plt.show()