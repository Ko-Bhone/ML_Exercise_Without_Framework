import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RegularizedLogisticRegression:
    def __init__(self, file_path, degree=6, alpha=1, iterations=800, Lambda=1):
        self.file_path = file_path
        self.degree = degree
        self.alpha = alpha
        self.iterations = iterations
        self.Lambda = Lambda

        # Data
        self.df = None
        self.x_original = None
        self.x = None
        self.y = None

        # Model
        self.theta = None
        self.j_history = []

    # Load Dat
    def load_data(self):
        self.df = pd.read_csv(self.file_path, header=None)
        self.x_original = self.df.iloc[:, :-1].to_numpy()
        self.y = self.df.iloc[:, -1].to_numpy().reshape(-1, 1)

    # Feature Mapping
    def map_feature(self, x1, x2):
        out = np.ones((len(x1), 1))
        for i in range(1, self.degree + 1):
            for j in range(i + 1):
                term = (x1 ** (i - j) * x2 ** j).reshape(-1, 1)
                out = np.hstack((out, term))
        return out

    def prepare_data(self):
        self.x = self.map_feature(self.x_original[:, 0], self.x_original[:, 1])

    # PCA for Visualization
    def pca_2d(self, X):
        X_norm = X - np.mean(X, axis=0)
        cov = (X_norm.T @ X_norm) / len(X_norm)
        U, S, Vt = np.linalg.svd(cov)
        return X_norm @ U[:, :2]

    # Sigmoid
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Cost Function + Gradient
    def cost_function(self, theta):
        m = len(self.y)
        h = self.sigmoid(self.x @ theta)
        h = np.clip(h, 1e-10, 1 - 1e-10)
        cost = (-self.y * np.log(h) - (1 - self.y) * np.log(1 - h))
        cost = np.sum(cost) / m
        reg = (self.Lambda / (2 * m)) * np.sum(theta[1:] ** 2)
        J = cost + reg
        grad = (1 / m) * (self.x.T @ (h - self.y))
        grad[1:] += (self.Lambda / m) * theta[1:]
        return J, grad

    # Gradient Descent
    def gradient_descent(self):
        self.theta = np.zeros((self.x.shape[1], 1))
        for _ in range(self.iterations):
            cost, grad = self.cost_function(self.theta)
            self.theta -= self.alpha * grad
            self.j_history.append(cost)

    # Dashboard Plot
    def plot_dashboard(self):
        pos = (self.y == 1).reshape(-1)
        neg = (self.y == 0).reshape(-1)
        fig, ax = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Regularized Logistic Regression Workflow", fontsize=18, fontweight='bold', y=0.97)

        # 1. Raw Data
        ax[0, 0].scatter(self.x_original[pos, 0], self.x_original[pos, 1],
                          c='green', s=80, edgecolors='black', label="Accepted")
        ax[0, 0].scatter(self.x_original[neg, 0], self.x_original[neg, 1],
                          c='red', marker='X', s=80, edgecolors='black', label="Rejected")
        ax[0, 0].set_title("1. Raw Data")
        ax[0, 0].set_xlabel("Test 1")
        ax[0, 0].set_ylabel("Test 2")
        ax[0, 0].grid(alpha=0.3, linestyle='--')
        ax[0, 0].legend()

        # 2. After Mapping (PCA)
        X_pca = self.pca_2d(self.x[:, 1:])
        ax[0, 1].scatter(X_pca[pos, 0], X_pca[pos, 1],
                          c='green', s=80, edgecolors='black', label="Accepted")
        ax[0, 1].scatter(X_pca[neg, 0], X_pca[neg, 1],
                          c='red', marker='X', s=80, edgecolors='black', label="Rejected")
        ax[0, 1].set_title("2. After Mapping (PCA)")
        ax[0, 1].grid(alpha=0.3, linestyle='--')
        ax[0, 1].legend()

        # 3. Cost Convergence
        ax[1, 0].plot(self.j_history, linewidth=3)
        ax[1, 0].set_title("3. Cost Convergence")
        ax[1, 0].set_xlabel("Iterations")
        ax[1, 0].set_ylabel("Cost")
        ax[1, 0].grid(alpha=0.3, linestyle='--')

        # 4. Decision Boundary
        ax[1, 1].scatter(self.x_original[pos, 0], self.x_original[pos, 1],
                          c='green', s=80, edgecolors='black', label="Accepted")
        ax[1, 1].scatter(self.x_original[neg, 0], self.x_original[neg, 1],
                          c='red', marker='X', s=80, edgecolors='black', label="Rejected")
        u = np.linspace(-1, 1.5, 300)
        v = np.linspace(-1, 1.5, 300)
        z = np.zeros((len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                mapped = self.map_feature(np.array([u[i]]), np.array([v[j]]))
                z[i, j] = (mapped @ self.theta)[0, 0]
        ax[1, 1].contour(u, v, z.T, levels=[0], linewidths=3)
        ax[1, 1].set_title("4. Decision Boundary")
        ax[1, 1].grid(alpha=0.3, linestyle='--')
        ax[1, 1].legend()
        fig.subplots_adjust(top=0.88)
        plt.show()

    # Run Pipeline
    def run(self):
        self.load_data()
        print("Raw Data Shape:", self.x_original.shape)
        self.prepare_data()
        print("Mapped Data Shape:", self.x.shape)
        self.gradient_descent()
        print("Theta Shape:", self.theta.shape)
        self.plot_dashboard()

if __name__ == "__main__":
    model = RegularizedLogisticRegression(
        "C:/Users/User/Desktop/Machine learning exercise/data1/data/ex2data2.txt")

    model.run()