import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

class KMeansClustering():
    def __init__(self, file_path,K=3):
        self.file_path = file_path
        self.k = K
        self.x = None
        self.centroids = None
        self.idx = None

    def load_data(self):
        mat = loadmat(self.file_path)
        self.x = mat['X']
        print("Dataset Shape:",self.x.shape)

    def initial_centroids(self,centroids):
        self.centroids = np.array(centroids)

    def random_initialize_centroids(self):
        m,n = self.x.shape
        self.centroids = np.zeros((self.k,n))
        for i in range(self.k):
            random_idex = np.random.randint(0,m)
            self.centroids[i] = self.x[random_idex]

    def find_closest_centroid(self):
        idx = np.zeros((self.x.shape[0],1),dtype=int)
        temp = np.zeros((self.k,1))
        for i in range(self.x.shape[0]):
            for j in range(self.k):
                dist = (self.x[i,:] - self.centroids[j,:])
                length = np.sum(dist**2)
                temp [j] =length
            idx[i] = np.argmin(temp) + 1
        self.idx = idx

    def compute_centroids(self):
        m,n = self.x.shape
        centroids = np.zeros((self.k,n))
        count = np.zeros((self.k,1))
        for i in range(m):
            idex = int((self.idx[i] - 1)[0])
            centroids[idex] += self.x[i,:]
            count[idex] += 1
        self.centroids = (centroids / count)

    def plot_kmeans(self,num_iters=10):
        m,n = self.x.shapef
        fig,ax = plt.subplots(nrows=num_iters,ncols=1,figsize=(6,36))
        for i in range(num_iters):
            colors = "rgb"
            for k in range(1,self.k+1):
                grp = (self.idx == k).reshape(m,1)
                ax[i].scatter(self.x[grp[:,0],0],self.x[grp[:,0],1],c=colors[k-1],s=15)
            ax[i].scatter(self.centroids[:,0],self.centroids[:,1],s=120,marker="x",c="black",linewidth=3)
            ax[i].set_title(f"Iteration {i + 1}")
            self.compute_centroids()
            self.find_closest_centroid()
        plt.tight_layout()
        plt.show()