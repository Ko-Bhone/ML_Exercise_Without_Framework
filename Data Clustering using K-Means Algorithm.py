import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

class KMeansClustering:
    def __init__(self, file_path, K=3):
        self.file_path = file_path
        self.K = K
        self.X = None
        self.centroids = None
        self.idx = None

    def load_data(self):
        mat = loadmat(self.file_path)
        self.X = mat["X"]
        print("Dataset Shape:", self.X.shape)

    def set_initial_centroids(self, centroids):
        self.centroids = np.array(centroids, dtype=float)

    def random_initialize_centroids(self):
        m = self.X.shape[0]
        rand_idx = np.random.choice(m, self.K, replace=False)
        self.centroids = self.X[rand_idx].copy()

    def find_closest_centroid(self):
        m = self.X.shape[0]
        self.idx = np.zeros(m, dtype=int)
        for i in range(m):
            distances = np.sum((self.X[i] - self.centroids) ** 2, axis=1)
            self.idx[i] = np.argmin(distances)

    def compute_centroids(self):
        new_centroids = np.zeros_like(self.centroids)
        for k in range(self.K):
            points = self.X[self.idx == k]
            if len(points) > 0:
                new_centroids[k] = np.mean(points, axis=0)
            else:
                new_centroids[k] = self.centroids[k]
        self.centroids = new_centroids

    def plot_iteration(self, iteration):
        colors = ["r","g","b","c","m","y","k"]
        plt.figure(figsize=(6,6))
        for k in range(self.K):
            pts = self.X[self.idx==k]
            if len(pts):
                plt.scatter(pts[:,0], pts[:,1], c=colors[k%len(colors)], s=20, label=f"Cluster {k+1}")
        plt.scatter(self.centroids[:,0], self.centroids[:,1],
                    c="black", marker="x", s=150, linewidths=3, label="Centroids")
        plt.title(f"Iteration {iteration}")
        plt.legend()
        plt.show()

    def fit(self, tolerance=1e-6, max_iters=100, plot=True):
        if self.centroids is None:
            self.random_initialize_centroids()

        iteration = 0
        while True:
            iteration += 1
            old_centroids = self.centroids.copy()
            self.find_closest_centroid()
            self.compute_centroids()

            print(f"Iteration {iteration}")
            print(self.centroids)

            if plot:
                self.plot_iteration(iteration)

            if np.allclose(old_centroids, self.centroids, atol=tolerance):
                print(f"\nConverged after {iteration} iterations.")
                break

            if iteration >= max_iters:
                print("\nReached maximum iterations.")
                break

        print("\nFinal Centroids:")
        print(self.centroids)

    def predict(self, new_points):
        new_points = np.array(new_points)
        labels = []
        for p in new_points:
            distances = np.sum((p - self.centroids)**2, axis=1)
            labels.append(np.argmin(distances))
        return np.array(labels)

if __name__ == "__main__":
    model = KMeansClustering(
        file_path="C:/Users/User/Desktop/Machine learning exercise/data1/data/ex7/ex7data2.mat",
        K=3
    )

    model.load_data()

    model.set_initial_centroids([
        [3,3],
        [6,5],
        [8,5]
    ])
    # model.random_initialize_centroids()

    model.fit(tolerance=1e-5, max_iters=100, plot=True)

    sample = [[5.5,4.8],[3.2,2.9]]
    print("Predicted Clusters:", model.predict(sample))
