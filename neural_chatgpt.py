import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# =========================
# 1. Data Load
# =========================
data = loadmat('C:/Users/User/Desktop/Machine learning exercise/data1/data/ex3data1.mat')

x = data['X']   # (5000, 400) -> pixel features
y = data['y']   # (5000, 1) -> labels (1-10)

m, n = x.shape  # m = 5000, n = 400

# =========================
# 2. Sigmoid Function
# =========================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# =========================
# 3. Logistic Regression Cost + Gradient
# =========================
def cost_function(x, y, theta, Lambda):
    m = len(y)

    z = x @ theta              # (m,1)
    h = sigmoid(z)             # prediction

    # Cost (log loss)
    cost = np.sum(-y*np.log(h) - (1-y)*np.log(1-h)) / m

    # Regularization (theta0 မပါ)
    reg = (Lambda/(2*m)) * np.sum(theta[1:]**2)

    # Total cost
    total_cost = cost + reg

    # Gradient (vectorized)
    grad = (1/m) * (x.T @ (h - y))

    # Regularization gradient (theta0 မပါ)
    grad[1:] += (Lambda/m) * theta[1:]

    return total_cost, grad


# =========================
# 4. Gradient Descent
# =========================
def gradient_descent(x, y, theta, alpha, iterations, Lambda):
    j_history = []

    for i in range(iterations):
        cost, grad = cost_function(x, y, theta, Lambda)

        # theta update
        theta = theta - alpha * grad

        j_history.append(cost)

    return theta, j_history


# =========================
# 5. One-vs-All (Multiclass)
# =========================
def one_vs_all(x, y, num_labels, Lambda):
    m, n = x.shape

    # bias column ထည့်
    x = np.hstack((np.ones((m,1)), x))   # (5000,401)

    all_theta = np.zeros((num_labels, n+1))

    for i in range(1, num_labels+1):

        # label ကို binary ပြောင်း (i vs all)
        y_i = (y == i).astype(int)

        # initial theta
        theta = np.zeros((n+1,1))

        theta, _ = gradient_descent(x, y_i, theta, alpha=0.1, iterations=300, Lambda=Lambda)

        all_theta[i-1,:] = theta.ravel()

    return all_theta


# =========================
# 6. Train One-vs-All
# =========================
all_theta = one_vs_all(x, y, num_labels=10, Lambda=0.1)


# =========================
# 7. Prediction (Logistic)
# =========================
def predict_one_vs_all(all_theta, x):
    m = x.shape[0]

    x = np.hstack((np.ones((m,1)), x))  # bias

    probs = x @ all_theta.T             # (m,10)

    return np.argmax(probs, axis=1) + 1


pred = predict_one_vs_all(all_theta, x)

accuracy = np.mean(pred.reshape(-1,1) == y) * 100
print("Logistic Regression Accuracy:", accuracy, "%")


# =========================
# 8. Load Pretrained NN
# =========================
mat = loadmat("C:/Users/User/Desktop/Machine learning exercise/data1/data/ex3weights.mat")

Theta1 = mat['Theta1']  # (25,401)
Theta2 = mat['Theta2']  # (10,26)


# =========================
# 9. Neural Network Prediction
# =========================
def predict_nn(theta1, theta2, x):
    m = x.shape[0]

    # input layer
    a1 = np.hstack((np.ones((m,1)), x))

    # hidden layer
    a2 = sigmoid(a1 @ theta1.T)
    a2 = np.hstack((np.ones((m,1)), a2))

    # output layer
    h = sigmoid(a2 @ theta2.T)

    return np.argmax(h, axis=1) + 1


pred2 = predict_nn(Theta1, Theta2, x)

accuracy2 = np.mean(pred2.reshape(-1,1) == y) * 100
print("Pretrained NN Accuracy:", accuracy2, "%")


# =========================
# 10. Sigmoid Gradient
# =========================
def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)


# =========================
# 11. Neural Network Cost + Backprop
# =========================
def nn_cost_function(nn_params, input_size, hidden_size, num_labels, x, y, Lambda):

    # theta reshape
    theta1 = nn_params[:hidden_size*(input_size+1)].reshape(hidden_size, input_size+1)
    theta2 = nn_params[hidden_size*(input_size+1):].reshape(num_labels, hidden_size+1)

    m = x.shape[0]

    # Forward Propagation
    a1 = np.hstack((np.ones((m,1)), x))

    z2 = a1 @ theta1.T
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m,1)), a2))

    z3 = a2 @ theta2.T
    h = sigmoid(z3)

    # One-hot encoding
    y_matrix = np.zeros((m, num_labels))
    for i in range(num_labels):
        y_matrix[:,i] = (y.flatten() == (i+1)).astype(int)

    # Cost
    cost = np.sum(-y_matrix*np.log(h) - (1-y_matrix)*np.log(1-h)) / m

    # Regularization
    reg = (Lambda/(2*m)) * (
        np.sum(theta1[:,1:]**2) +
        np.sum(theta2[:,1:]**2)
    )

    total_cost = cost + reg

    # ======================
    # Backpropagation
    # ======================
    delta3 = h - y_matrix
    delta2 = (delta3 @ theta2)[:,1:] * sigmoid_grad(z2)

    grad1 = (delta2.T @ a1) / m
    grad2 = (delta3.T @ a2) / m

    # Regularization gradient
    grad1[:,1:] += (Lambda/m) * theta1[:,1:]
    grad2[:,1:] += (Lambda/m) * theta2[:,1:]

    return total_cost, grad1, grad2


# =========================
# 12. Random Initialization
# =========================
def random_init(L_in, L_out):
    epsilon = np.sqrt(6) / np.sqrt(L_in + L_out)
    return np.random.rand(L_out, L_in+1) * 2*epsilon - epsilon


initial_theta1 = random_init(n, 25)
initial_theta2 = random_init(25, 10)

initial_nn_params = np.concatenate([initial_theta1.ravel(), initial_theta2.ravel()])


# =========================
# 13. Neural Network Training
# =========================
def nn_gradient_descent(x, y, nn_params, alpha, iterations, Lambda, input_size, hidden_size, num_labels):

    theta1 = nn_params[:hidden_size*(input_size+1)].reshape(hidden_size, input_size+1)
    theta2 = nn_params[hidden_size*(input_size+1):].reshape(num_labels, hidden_size+1)

    j_history = []

    for i in range(iterations):

        nn_params = np.concatenate([theta1.ravel(), theta2.ravel()])

        cost, grad1, grad2 = nn_cost_function(
            nn_params, input_size, hidden_size, num_labels, x, y, Lambda
        )

        theta1 -= alpha * grad1
        theta2 -= alpha * grad2

        j_history.append(cost)

    final_params = np.concatenate([theta1.ravel(), theta2.ravel()])

    return final_params, j_history


# =========================
# 14. Train NN
# =========================
nn_params, history = nn_gradient_descent(
    x, y,
    initial_nn_params,
    alpha=0.1,
    iterations=300,
    Lambda=1,
    input_size=400,
    hidden_size=25,
    num_labels=10
)

plt.plot(history)
plt.title("NN Cost")
plt.show()


# =========================
# 15. Final Prediction
# =========================
Theta1 = nn_params[:25*(400+1)].reshape(25,401)
Theta2 = nn_params[25*(400+1):].reshape(10,26)

pred3 = predict_nn(Theta1, Theta2, x)

accuracy3 = np.mean(pred3.reshape(-1,1) == y) * 100
print("Trained NN Accuracy:", accuracy3, "%")