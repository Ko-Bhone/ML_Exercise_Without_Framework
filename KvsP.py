import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

mat = loadmat("C:/Users/User/Desktop/Machine learning exercise/data1/data/ex7/ex7data2.mat")
x=mat["X"]
print(x.shape)


initialcentroid=np.array([[3,3],[6,5],[8,5]])
print(initialcentroid.shape)
K=initialcentroid.shape[0]

def findCloseCentroid(x,centroids):
    K = centroids.shape[0]
    idx = np.zeros((x.shape[0],1),dtype=int)
    temp = np.zeros((centroids.shape[0],1))

    for i in range(x.shape[0]):
        for j in range(K):
            dist = x[i,:] - centroids[j,:]
            length = np.sum(dist**2)
            temp[j] = length
        idx[i] = np.argmin(temp)+1
    return idx

idx = findCloseCentroid(x,initialcentroid)
print(idx.shape)
print(idx)

def conputeCentroid(x,idx,K):
    m,n=x.shape[0],x.shape[1]
    centroids = np.zeros((K,n))
    count = np.zeros((K,1))

    for i in range(m):
        index = int((idx[i]-1)[0])
        centroids[index,:] +=x[i,:]
        count[index]+=1

    return centroids / count

centroid = conputeCentroid(x,idx,3)

print("Centroid after initial",centroid)

def plotKmean(X,centroids,idx,K,num_iters):
    m,n = x.shape[0],x.shape[1]
    fig, ax = plt.subplots(nrows=num_iters,ncols=1,figsize=(6,36))

    for i in range(num_iters):
        color = "rgb"
        for K in range(1,K+1):
            grp=(idx==K).reshape(m,1)
            ax[i].scatter(x[grp[:,0],0],x[grp[:,0],1],c=color[K-1],s=15)

        ax[i].scatter(centroids[:,0],centroids[:,1],s=120,marker="x",c="black",linewidth=3)
        title = "Iteration Number" + str(i)
        ax[i].set_title(title)

        centroids = conputeCentroid(x,idx,K)
        idx=findCloseCentroid(x,centroids)
    plt.tight_layout()
    plt.show()


m,n = x.shape[0],x.shape[1]
plotKmean(x,initialcentroid,idx,K,10)


def kmeansInitCentroids(x,k):
    m,n = x.shape[0],x.shape[1]
    centroids=np.zeros((k,n))

    for i in range(k):
        centroids[i] = x[np.random.randint(0,m+1),:]

    return centroids

centroids=kmeansInitCentroids(x,K)
idx = findCloseCentroid(x,centroids)
plotKmean(x,centroids,idx,K,10)

