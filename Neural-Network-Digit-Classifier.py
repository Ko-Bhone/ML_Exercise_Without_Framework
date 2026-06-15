import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.gridspec import GridSpec

class DigitClassifier:
    def __init__(self, data_path, weight_path, alpha=0.1, iterations=300, Lambda=1):
        self.data_path = data_path
        self.weight_path = weight_path
        self.alpha = alpha
        self.iterations = iterations
        self.Lambda = Lambda

        self.x = None
        self.y = None

        self.all_theta = None

        self.theta1 = None
        self.theta2 = None

        self.log_cost_history = []
        self.nn_cost_history = []

    # Activation Functions
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_grad(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    # Data Loading
    def load_data(self):
        data = loadmat(self.data_path)
        self.x = data['X']
        self.y = data['y']
        print(f"Dataset Loaded: {self.x.shape}")

    def load_pretrained_weights(self):
        mat = loadmat(self.weight_path)
        self.theta1 = mat['Theta1']
        self.theta2 = mat['Theta2']

  # Logistic Regression
    def logistic_cost(self, x, y, theta):
        m = len(y)
        z = x @ theta
        h = self.sigmoid(z)
        h = np.clip(h, 1e-10, 1 - 1e-10)
        cost = np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h)) / m
        reg = (self.Lambda / (2 * m)) * np.sum(theta[1:] ** 2)
        total_cost = cost + reg
        grad = (1 / m) * (x.T @ (h - y))
        grad[1:] += (self.Lambda / m) * theta[1:]
        return total_cost, grad

    def gradient_descent(self, x, y, theta):
        history = []
        for i in range(self.iterations):
            cost, grad = self.logistic_cost(x, y, theta)
            theta -= self.alpha * grad
            history.append(cost)
        return theta, history

    def train_logistic(self):
        m, n = self.x.shape
        x_bias = np.hstack((np.ones((m, 1)), self.x))
        self.all_theta = np.zeros((10, n + 1))
        for i in range(1, 11):
            y_i = (self.y == i).astype(int)
            theta = np.zeros((n + 1, 1))
            theta, history = self.gradient_descent(x_bias, y_i, theta)
            self.all_theta[i - 1, :] = theta.ravel()
            self.log_cost_history.append(history)
        print("Logistic Training Done!")

    def predict_logistic(self):
        m = self.x.shape[0]
        x_bias = np.hstack((np.ones((m, 1)), self.x))
        probs = x_bias @ self.all_theta.T
        pred = np.argmax(probs, axis=1) + 1
        accuracy = np.mean(pred.reshape(-1, 1) == self.y) * 100
        print(f"Logistic Accuracy: {accuracy:.2f}%")
        return pred

    # Neural Network
    def random_init(self, L_in, L_out):
        epsilon = np.sqrt(6) / np.sqrt(L_in + L_out)
        return np.random.rand(L_out, L_in + 1) * 2 * epsilon - epsilon

    def one_hot(self):
        m = self.y.shape[0]
        y_matrix = np.zeros((m, 10))
        for i in range(10):
            y_matrix[:, i] = (self.y.flatten() == i + 1).astype(int)
        return y_matrix

    def forward(self, x):
        m = x.shape[0]
        a1 = np.hstack((np.ones((m, 1)), x))
        z2 = a1 @ self.theta1.T
        a2 = self.sigmoid(z2)
        a2 = np.hstack((np.ones((m, 1)), a2))
        z3 = a2 @ self.theta2.T
        h = self.sigmoid(z3)
        return a1, z2, a2, h

    def train_nn(self):
        self.theta1 = self.random_init(400, 25)
        self.theta2 = self.random_init(25, 10)
        y_matrix = self.one_hot()
        m = self.x.shape[0]
        for i in range(self.iterations):
            a1, z2, a2, h = self.forward(self.x)
            h = np.clip(h, 1e-10, 1 - 1e-10)
            cost = np.sum(-y_matrix * np.log(h) - (1 - y_matrix) * np.log(1 - h)) / m
            reg = (self.Lambda / (2 * m)) * (np.sum(self.theta1[:, 1:] ** 2) + np.sum(self.theta2[:, 1:] ** 2))
            cost += reg
            delta3 = h - y_matrix
            delta2 = (delta3 @ self.theta2)[:, 1:] * self.sigmoid_grad(z2)
            grad1 = (delta2.T @ a1) / m
            grad2 = (delta3.T @ a2) / m
            grad1[:, 1:] += (self.Lambda / m) * self.theta1[:, 1:]
            grad2[:, 1:] += (self.Lambda / m) * self.theta2[:, 1:]
            self.theta1 -= self.alpha * grad1
            self.theta2 -= self.alpha * grad2
            self.nn_cost_history.append(cost)
        print("Neural Network Training Done!")

    def predict_nn(self):
        _, _, _, h = self.forward(self.x)
        pred = np.argmax(h, axis=1) + 1
        accuracy = np.mean(pred.reshape(-1, 1) == self.y) * 100
        print(f"NN Accuracy: {accuracy:.2f}%")
        return pred

    # Visualization
    def visualize_training(self):
        fig = plt.figure(figsize=(15, 6))
        gs = GridSpec(1, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax1.plot(np.mean(self.log_cost_history, axis=0), linewidth=3)
        ax1.set_title("Logistic Cost")
        ax1.grid(True)
        ax2.plot(self.nn_cost_history, linewidth=3)
        ax2.set_title("Neural Network Cost")
        ax2.grid(True)
        plt.tight_layout()
        plt.show()

    def show_random_prediction(self, pred):
        idx = np.random.randint(0, self.x.shape[0])
        image = self.x[idx].reshape(20, 20).T
        plt.figure(figsize=(5, 5))
        plt.imshow(image, cmap="viridis")
        plt.title(f"Actual: {self.y[idx][0]} | Pred: {pred[idx]}")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    model = DigitClassifier(data_path="C:/Users/User/Desktop/Machine learning exercise/data1/data/ex3data1.mat",
        weight_path="C:/Users/User/Desktop/Machine learning exercise/data1/data/ex3weights.mat",
        alpha=0.1,
        iterations=300,
        Lambda=1)
    model.load_data()
    model.train_logistic()
    pred1 = model.predict_logistic()
    model.train_nn()
    pred2 = model.predict_nn()
    model.visualize_training()
    model.show_random_prediction(pred2)