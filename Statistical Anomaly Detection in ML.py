import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

class GaussianAnomalyDetector:
    def __init__(self):
        self.file_path = ("C:/Users/User/Desktop/Machine learning exercise/data1/data/ex8/ex8data1.mat")
        self.x = None
        self.x_val = None
        self.y_val = None

        # MODEL PARAMETERS
        self.mu = None
        self.sigma2 = None
        self.epsilon = None
        self.best_f1 = None

        # Auto Run
        self.load_data()
        self.preprocess()

    # LOAD DATA
    def load_data(self):
        try:
            mat = loadmat(self.file_path)
            self.x = mat["X"]
            self.x_val = mat["Xval"]
            self.y_val = mat["yval"]

            print("=" * 50)
            print("Dataset Loaded Successfully")
            print("=" * 50)

            print(f"Training Shape   : {self.x.shape}")
            print(f"Validation Shape : {self.x_val.shape}")
            print(f"Label Shape      : {self.y_val.shape}")
        except Exception as e:
            print(f"Error Loading Dataset: {e}")

    # PREPROCESSING
    def preprocess(self):
        print("\nChecking Missing Values...")
        self.x = self.fill_missing_values(self.x)
        self.x_val = self.fill_missing_values(self.x_val)
        print("Preprocessing Completed")

    def fill_missing_values(self, data):
        if np.isnan(data).sum() > 0:
            print("Missing values detected")
            for col in range(data.shape[1]):
                median_value = np.nanmedian(data[:, col])
                data[:, col] = np.where(np.isnan(data[:, col]), median_value, data[:, col])
            print("Missing values filled")
        else:
            print("No missing values found")
        return data

    # DATA VISUALIZATION
    def plot_dataset(self):
        plt.figure(figsize=(10, 7))
        plt.scatter(self.x[:, 0], self.x[:, 1], marker='x', s=80, alpha=0.8, label="Training Data")
        plt.xlabel("Latency (ms)", fontsize=12)
        plt.ylabel("Throughput (mb/s)", fontsize=12)
        plt.title("Training Dataset", fontsize=15, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # TRAIN MODEL
    def fit(self):
        m = self.x.shape[0]
        self.mu = (1 / m) * np.sum(self.x, axis=0)
        self.sigma2 = (1 / m) * np.sum((self.x - self.mu) ** 2, axis=0)
        print("\n" + "=" * 50)
        print("Training Completed")
        print("=" * 50)
        print("Mu:")
        print(self.mu)
        print("\nSigma²:")
        print(self.sigma2)

    # MULTIVARIATE GAUSSIAN
    def multivariate_gaussian(self, x):
        k = len(self.mu)
        sigma_matrix = np.diag(self.sigma2)
        centered_X = x - self.mu.T
        coefficient = 1 / (((2 * np.pi) ** (k / 2)) * (np.linalg.det(sigma_matrix) ** 0.5))
        exponent = np.exp(-0.5 * np.sum(centered_X @ np.linalg.pinv(sigma_matrix) * centered_X, axis=1))
        probability = (coefficient * exponent)
        return probability

    # VISUALIZE GAUSSIAN FIT
    def visualize_fit(self):
        x1, x2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
        grid_points = np.stack([x1.ravel(), x2.ravel()], axis=1)
        z = self.multivariate_gaussian(grid_points)
        z = z.reshape(x1.shape)
        plt.figure(figsize=(10, 7))
        plt.scatter(self.x[:, 0], self.x[:, 1], marker='x', s=70, alpha=0.7, label="Training Data")
        contour = plt.contour(x1, x2, z, levels=10 ** (np.arange(-20., 1, 3)), colors='red', linewidths=1.5)
        plt.clabel(contour, inline=True, fontsize=8)
        plt.xlabel("Latency (ms)", fontsize=12)
        plt.ylabel("Throughput (mb/s)", fontsize=12)
        plt.title("Gaussian Distribution Fit", fontsize=15, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # THRESHOLD SELECTION
    def select_threshold(self):
        p_val = self.multivariate_gaussian(
            self.x_val)
        best_epsilon = 0
        best_f1 = 0
        step_size = (p_val.max() - p_val.min()) / 1000
        epsilon_range = np.arange(p_val.min(), p_val.max(), step_size)
        for epsilon in epsilon_range:
            prediction = (p_val < epsilon)
            tp = np.sum((prediction == self.y_val) & (self.y_val == 1))
            fp = np.sum((prediction == 1) & (self.y_val == 0))
            fn = np.sum((prediction == 0) & (self.y_val == 1))
            precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0
            recall = (tp / (tp + fn))\
                if (tp + fn) > 0 \
                else 0
            f1 = (2 * precision * recall) / (precision + recall)\
                if (precision + recall) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_epsilon = epsilon
        self.epsilon = best_epsilon
        self.best_f1 = best_f1

        print("\n" + "=" * 50)
        print("Threshold Selection Completed")
        print("=" * 50)

        print(f"Best Epsilon : ",f"{self.epsilon}")
        print(f"Best F1 Score: ",f"{self.best_f1:.4f}")

    # DETECT OUTLIERS
    def detect_outliers(self):
        probabilities = (self.multivariate_gaussian(self.x))
        outliers = (probabilities < self.epsilon)
        print(f"\nOutliers Found: ",f"{np.sum(outliers)}")
        return outliers

    # PLOT OUTLIERS
    def plot_outliers(self):
        outliers = self.detect_outliers()
        x1, x2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
        grid_points = np.stack([x1.ravel(), x2.ravel()], axis=1)
        z = self.multivariate_gaussian(grid_points)
        z = z.reshape(x1.shape)
        plt.figure(figsize=(10, 7))
        # Normal Data
        plt.scatter(self.x[:, 0], self.x[:, 1], marker='x', s=70, alpha=0.6, label='Normal Data')

        # Outliers
        plt.scatter(self.x[outliers, 0], self.x[outliers, 1], s=220, facecolors='none', linewidths=2, label='Anomaly')

        # Gaussian Contour
        plt.contour(x1, x2, z,levels=10 ** (np.arange(-20., 1, 3)))
        plt.xlabel("Latency (ms)", fontsize=12)
        plt.ylabel("Throughput (mb/s)", fontsize=12)
        plt.title("Anomaly Detection Result", fontsize=15, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # RUN COMPLETE PIPELINE
    def run(self):
        self.plot_dataset()
        self.fit()
        self.visualize_fit()
        self.select_threshold()
        self.plot_outliers()

if __name__ == "__main__":

    detector = GaussianAnomalyDetector()
    detector.run()
