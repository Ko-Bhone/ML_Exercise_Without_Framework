import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegressionModel:
    def __init__(self, file_path,alpha=0.1, iterations=400):
        self.file_path = file_path
        self.alpha = alpha
        self.iterations = iterations
        self.theta = None
        self.mean = None

        self.std = None
        self.cost_history = []
        self.x = None
        self.y = None

    # Load Data
    def load_data(self):
        data = pd.read_csv(self.file_path, header=None)
        self.x = data.iloc[:, :-1].to_numpy()
        self.y = data.iloc[:, -1].to_numpy().reshape(-1, 1)

    # Plot Raw Data
    def plot_raw_data(self):
        pos = (self.y == 1).ravel()
        neg = (self.y == 0).ravel()
        plt.figure(figsize=(8, 6))
        plt.scatter(self.x[pos, 0], self.x[pos, 1], marker="+", s=100, linewidths=2, label="Admitted")
        plt.scatter(self.x[neg, 0], self.x[neg, 1], marker="o", s=60, alpha=0.7, label="Not Admitted")
        plt.title("Raw Training Data")
        plt.xlabel("Exam 1 Score")
        plt.ylabel("Exam 2 Score")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Feature Scaling
    def feature_scaling(self):
        self.mean = np.mean(self.x, axis=0)
        self.std = np.std(self.x, axis=0)
        self.x = (self.x - self.mean) / self.std


   # Sigmoid
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # Cost Function
    def compute_cost_gradient(self):
        m = len(self.y)
        z = np.dot(self.x,self.theta)
        h = self.sigmoid(z)
        h = np.clip(h,1e-10, 1 - 1e-10)
        cost = (-self.y * np.log(h) - (1 - self.y) * np.log(1 - h)).mean()
        gradient = (np.dot(self.x.T, (h - self.y)) / m)
        return cost, gradient


    # Train Model
    def fit(self):
        self.load_data()
        # Visualize raw data before preprocessing
        self.plot_raw_data()
        # Feature Scaling
        self.feature_scaling()
        m, n = self.x.shape
        # Add Bias
        self.x = np.c_[np.ones((m, 1)), self.x]
        self.theta = np.zeros((n + 1, 1))
        for i in range(self.iterations):
            cost, grad = self.compute_cost_gradient()
            self.theta -= self.alpha * grad
            self.cost_history.append(cost)

    # Predict Probability
    def predict_probability(self, score):
        score = (score - self.mean) / self.std
        score = np.insert(score,0,1)
        z = np.dot(score,self.theta)
        return self.sigmoid(z).item()

    # Predict Class
    def predict(self):
        z = np.dot(self.x, self.theta)
        h = self.sigmoid(z)
        return h >= 0.5

    # Accuracy
    def accuracy(self):
        prediction = self.predict()
        acc = np.mean(prediction == self.y) * 100
        return acc

    # Plot Cost
    def plot_cost(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.cost_history, linewidth=3)
        plt.title("Cost Convergence")
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.grid(True)
        plt.show()

    # Plot Decision Boundary
    def plot_decision_boundary(self):
        pos = (self.y == 1).ravel()
        neg = (self.y == 0).ravel()
        plt.figure(figsize=(10, 7))
        plt.scatter(self.x[pos, 1], self.x[pos, 2], marker="+", s=120, label="Admitted")
        plt.scatter(self.x[neg, 1], self.x[neg, 2], s=80, alpha=0.7, label="Not Admitted")
        x_value = np.array([np.min(self.x[:, 1]), np.max(self.x[:, 1])])
        y_value = -(self.theta[0] + self.theta[1] * x_value) / self.theta[2]
        plt.plot(x_value, y_value.flatten(), linewidth=3, label="Boundary")
        plt.title("Decision Boundary")
        plt.xlabel("Exam 1")
        plt.ylabel("Exam 2")
        plt.legend()
        plt.grid(True)
        plt.show()


    # Run Entire Project
    def run(self):
        self.fit()
        print("\nTheta:")
        print(self.theta)
        probability = (self.predict_probability(np.array([45, 85])) * 100)
        print(f"\nAdmission Probability: {probability:.2f}%")
        print(f"\nAccuracy: {self.accuracy():.2f}%")
        self.plot_cost()
        self.plot_decision_boundary()

# Main Entry Point
def main():
    model = LogisticRegressionModel(
        file_path="C:/Users/User/Desktop/Machine learning exercise/data1/data/ex2data1.txt")
    model.run()

if __name__ == "__main__":
    main()