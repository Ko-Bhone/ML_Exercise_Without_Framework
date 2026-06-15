import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

class KMeansClustering:
    def __init__(self, file_path, K=3):
        self.file_path = file_path
        self.K = K
        self.x = None
        self.centroids = None
        self.idx = None

    # Load Data
    def load_data(self):
        mat = loadmat(self.file_path)
        self.x = mat["X"]
        print("Dataset Shape:", self.x.shape)

    # Manual centroid initialization
    def set_initial_centroids(self, centroids):
        self.centroids = np.array(centroids)

    def random_initialize_centroids(self):
        m, n = self.x.shape
        self.centroids = np.zeros((self.K, n))
        for i in range(self.K):
            random_index = np.random.randint(0, m)
            self.centroids[i] = self.x[random_index]

    # Find closest centroid
    def find_closest_centroid(self):
        idx = np.zeros((self.x.shape[0], 1), dtype=int)
        temp = np.zeros((self.K, 1))
        for i in range(self.x.shape[0]):
            for j in range(self.K):
                dist = (self.x[i, :] - self.centroids[j, :])
                length = np.sum(dist ** 2)
                temp[j] = length
            idx[i] = np.argmin(temp) + 1
        self.idx = idx

    # Compute new centroids
    def compute_centroids(self):
        m, n = self.x.shape
        centroids = np.zeros((self.K, n))
        count = np.zeros((self.K, 1))
        for i in range(m):
            index = int((self.idx[i] - 1)[0])
            centroids[index, :] += self.x[i, :]
            count[index] += 1
        self.centroids = (centroids / count)

    # Plot iterations
    def plot_kmeans(
            self, num_iters=10):
        m, n = self.x.shape
        fig, ax = plt.subplots(nrows=num_iters, ncols=1, figsize=(6, 36))
        for i in range(num_iters):
            colors = "rgb"
            for k in range(1, self.K + 1):
                grp = (self.idx == k).reshape(m, 1)
                ax[i].scatter(self.x[grp[:, 0], 0], self.x[grp[:, 0], 1], c=colors[k - 1], s=15)

            ax[i].scatter(self.centroids[:, 0], self.centroids[:, 1], s=120, marker="x", c="black", linewidth=3)
            ax[i].set_title(f"Iteration {i + 1}")
            self.compute_centroids()
            self.find_closest_centroid()
        plt.tight_layout()
        plt.show()

    # Train model
    def fit(self, num_iters=10):
        self.find_closest_centroid()
        self.plot_kmeans(num_iters)
        print("Final Centroids:")
        print(self.centroids)

# MAIN PROGRAM
if __name__ == "__main__":
    kmeans = KMeansClustering(file_path="C:/Users/User/Desktop/Machine learning exercise/data1/data/ex7/ex7data2.mat",
                              K=3)
    kmeans.load_data()
    # Manual centroid
    kmeans.set_initial_centroids([
        [3, 3],
        [6, 5],
        [8, 5]
    ])

    # Random centroid
    # kmeans.random_initialize_centroids()

    kmeans.fit(num_iters=10)
