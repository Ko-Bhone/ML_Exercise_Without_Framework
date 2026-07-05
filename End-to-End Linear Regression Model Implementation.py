import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegressionModel:
    def __init__(self, file_path):
        self.file_path = file_path

        # Data
        self.x = None
        self.y = None
        self.m = None
        # Model parameters
        self.theta = None
        # Training history
        self.J_history = []

    # Load and prepare data
    def load_data(self):
        data = pd.read_csv(self.file_path, header=None)
        data = data.to_numpy()
        x = data[:, 0]
        y = data[:, 1]
        self.m = y.size

        # Add bias term
        self.x = np.stack([np.ones(self.m), x], axis=1)
        self.y = y.reshape(self.m, 1)
        # Initialize theta
        self.theta = np.zeros((2, 1))

    # Plot training data
    def plot_training_data(self):
        plt.plot(self.x[:, 1], self.y.ravel(), 'ro', ms=8, mec='k')
        plt.ylabel('Profit in 10,000 $')
        plt.xlabel('Population of City in 10,000s')

    # Cost function
    def compute_cost(self):
        h = np.dot(self.x, self.theta)
        d = h - self.y
        J = (1 / (2 * self.m)) * np.dot(d.T, d)
        return J[0, 0]

    # Gradient Descent
    def gradient_descent(self, alpha=0.01, num_iter=1500):
        for i in range(num_iter):
            h = np.dot(self.x, self.theta)
            d = h - self.y
            self.theta = self.theta - ((alpha / self.m) * np.dot(self.x.T, d))
            self.J_history.append(self.compute_cost())
        return self.theta

    # Train model
    def train(self, alpha=0.01, num_iter=1500):
        initial_cost = self.compute_cost()
        print("Initial Cost =", initial_cost)
        self.gradient_descent(alpha, num_iter)
        print("\nTheta:")
        print(self.theta)
        print(self.J_history)

    # Cost History Plot
    def plot_cost_history(self):
        plt.figure(figsize=(7,5))
        plt.plot(self.J_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Cost History')
        plt.grid(True)
        plt.show()

    # Plot regression line
    def plot_regression_line(self):
        plt.figure()
        self.plot_training_data()
        plt.plot(self.x[:, 1], np.dot(self.x, self.theta), '-')
        plt.legend(['Training Data', 'Linear Regression'])
        plt.show()

    # Prediction
    def predict(self, population):
        xnew = np.array([1, population]).reshape(2, 1)
        prediction = np.dot(self.theta.T,xnew)[0, 0]
        return prediction * 10000

# Main
if __name__ == "__main__":
    model = LinearRegressionModel("C:/Users/User/Desktop/Machine learning exercise/data1/data/ex1data1.txt")
    model.load_data()
    model.train(alpha=0.01, num_iter=1500)
    model.plot_cost_history()
    model.plot_regression_line()
    prediction = model.predict(3.5)
    print("\nPrediction for population 35,000 =", prediction)