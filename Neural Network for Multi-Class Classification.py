import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


# DATA LOADER CLASS
class DataLoader:
    def __init__(self, path_x, path_theta=None):
        self.path_x = path_x
        self.path_theta = path_theta

    def load_data(self):
        data = loadmat(self.path_x)
        X = data["X"]
        y = data["y"]
        return X, y

    def load_weights(self):
        if self.path_theta:
            mat = loadmat(self.path_theta)
            return mat["Theta1"], mat["Theta2"]
        return None, None

# CORE MODEL CLASS
class NeuralNetworkML:
    def __init__(self, input_size=400, hidden_size=25, num_labels=10, Lambda=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.Lambda = Lambda

        self.Theta1 = None
        self.Theta2 = None
        self.cost_history = []

    # -------- Activation --------
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_grad(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    # -------- Init weights --------
    def random_init(self, L_in, L_out):
        epi = np.sqrt(6) / np.sqrt(L_in + L_out)
        return np.random.rand(L_out, L_in + 1) * (2 * epi) - epi

    # -------- Forward Propagation --------
    def forward(self, X, Theta1, Theta2):
        m = X.shape[0]
        a1 = np.hstack((np.ones((m, 1)), X))
        z2 = a1 @ Theta1.T
        a2 = self.sigmoid(z2)
        a2 = np.hstack((np.ones((m, 1)), a2))
        z3 = a2 @ Theta2.T
        h = self.sigmoid(z3)
        return a1, a2, h, z2

    # -------- Cost + Gradient --------
    def nn_cost_grad(self, nn_params, X, y):
        Theta1 = nn_params[:(self.input_size + 1) * self.hidden_size] \
            .reshape(self.hidden_size, self.input_size + 1)
        Theta2 = nn_params[(self.input_size + 1) * self.hidden_size:] \
            .reshape(self.num_labels, self.hidden_size + 1)
        m = X.shape[0]
        a1, a2, h, z2 = self.forward(X, Theta1, Theta2)
        # one-hot
        y_matrix = np.eye(self.num_labels)[y.flatten() - 1]
        # cost
        cost = (-1 / m) * np.sum(
            y_matrix * np.log(h) + (1 - y_matrix) * np.log(1 - h))
        reg = (self.Lambda / (2 * m)) * (
            np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2))
        cost += reg

        # backprop
        delta3 = h - y_matrix
        delta2 = (delta3 @ Theta2)[:, 1:] * self.sigmoid_grad(z2)
        grad1 = (delta2.T @ a1) / m
        grad2 = (delta3.T @ a2) / m
        # regularization
        grad1[:, 1:] += (self.Lambda / m) * Theta1[:, 1:]
        grad2[:, 1:] += (self.Lambda / m) * Theta2[:, 1:]
        return cost, grad1, grad2

    # -------- Training --------
    def train(self, X, y, alpha=0.8, epochs=800):
        self.Theta1 = self.random_init(self.input_size, self.hidden_size)
        self.Theta2 = self.random_init(self.hidden_size, self.num_labels)
        for i in range(epochs):
            nn_params = np.append(self.Theta1.flatten(), self.Theta2.flatten())
            cost, grad1, grad2 = self.nn_cost_grad(nn_params, X, y)
            self.Theta1 -= alpha * grad1
            self.Theta2 -= alpha * grad2
            self.cost_history.append(cost)
        return self

    # -------- Prediction --------
    def predict(self, X):
        _, _, h, _ = self.forward(X, self.Theta1, self.Theta2)
        return np.argmax(h, axis=1) + 1

    # -------- Accuracy --------
    def accuracy(self, X, y):
        pred = self.predict(X).reshape(-1, 1)
        return np.mean(pred == y) * 100

    # -------- Advanced Plot --------
    def plot_training(self):
        plt.style.use("seaborn-v0_8-darkgrid")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.cost_history, color="royalblue", linewidth=2)
        ax.set_title("Neural Network Training Loss Curve", fontsize=14)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Cost J(θ)")
        ax.fill_between(range(len(self.cost_history)), self.cost_history, alpha=0.2)
        plt.tight_layout()
        plt.show()

# LOGISTIC ONE-VS-ALL (optional)
class OneVsAll:
    def __init__(self, Lambda=0.1):
        self.Lambda = Lambda
        self.all_theta = None
        self.history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost_grad(self, X, y, theta):
        m = len(X)
        h = self.sigmoid(X @ theta)
        cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        reg = (self.Lambda / (2 * m)) * np.sum(theta[1:] ** 2)
        cost += reg
        grad = (1 / m) * (X.T @ (h - y))
        grad[1:] += (self.Lambda / m) * theta[1:]
        return cost, grad

    def train(self, X, y, num_labels=10, alpha=1, epochs=300):
        m, n = X.shape
        X = np.hstack((np.ones((m, 1)), X))
        self.all_theta = np.zeros((num_labels, n + 1))
        for i in range(num_labels):
            theta = np.zeros((n + 1, 1))
            y_i = (y == (i + 1)).astype(int)
            for _ in range(epochs):
                cost, grad = self.cost_grad(X, y_i, theta)
                theta -= alpha * grad.reshape(-1, 1)
                self.history.append(cost)
            self.all_theta[i] = theta.flatten()
        return self

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return np.argmax(X @ self.all_theta.T, axis=1) + 1

# MAIN (NO DATA LOADING HERE)
if __name__ == "__main__":
    # just run pipeline
    loader = DataLoader(
        "C:/Users/User/Desktop/Machine learning exercise/data1/data/ex3data1.mat")
    X, y = loader.load_data()
    nn = NeuralNetworkML()
    nn.train(X, y)
    print("Accuracy:", nn.accuracy(X, y), "%")
    nn.plot_training()