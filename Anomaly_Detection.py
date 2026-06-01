import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

mat = loadmat("C:/Users/User/Desktop/Machine learning exercise/data1/data/ex8/ex8data1.mat")
x = mat["X"]

x_val = mat["Xval"]
y_val = mat["yval"]

plt.scatter(x[:,0],x[:,1],marker="x")
plt.xlim(0,30)
plt.ylim(0,30)
plt.xlabel("Latency (ms)")
plt.ylabel("Throughput (mb/s)")
plt.show()

def estimateCussian(x):
    m = x.shape[0]
    mu = (1/m) * np.sum(x,axis=0)
    sigma2 = (1/m) * np.sum((x-mu)**2,axis=0)
    return mu, sigma2

def estimateCussian(x):
    m = x.shape[0]
    mu = (1/m) * np.sum(x,axis=0)

mu, sigma2 = estimateCussian(x)
print(mu)
print(sigma2)


def multivariateGaussian(x,mu,sigma2):
    k = len(mu)
    sigma2 = np.diag(sigma2)
    x = x - mu.T
    p = 1 / ((2*np.pi)**(k/2)*(np.linalg.det(sigma2)**0.5)) * np.exp(-0.5 * np.sum(x @ np.linalg.pinv(sigma2) * x, axis=1))
    return p
mu, sigma2 = estimateCussian(x)
p = multivariateGaussian(x,mu,sigma2)
print(p.shape)

def visualizefit(x,mu,sigma2):
    x1,x2 = np.meshgrid(np.arange(0,35.5,0.5),np.arange(0,35.5,0.5))
    z = multivariateGaussian(np.stack([x1.ravel(),x2.ravel()],axis=1), mu, sigma2)
    z = z.reshape(x1.shape)
    plt.plot(x[:,0],x[:,1],'bx',mec='b',mew=2,ms=8)
    if np.all(abs(z) != np.inf):
        plt.contour(x1,x2,z,levels=10 ** (np.arange(-20.,1,3)),zorder=100)

visualizefit(x,mu,sigma2)
plt.xlabel("Latency (ms)")
plt.ylabel("Throughput (mb/s)")
plt.tight_layout()
plt.show()

def selectThreadshold(y_val,p_val):
    best_epi = 0
    best_F1 = 0

    stepsize = (max(p_val) - min(p_val)) / 1000
    epi_range = np.arange(p_val.min(),p_val.max(),stepsize)

    for epi in epi_range:
        prdiction = (p_val < epi)[:,np.newaxis]
        tp = np.sum((prdiction == y_val) & (y_val == 1))
        fp = np.sum((prdiction == 1) & (y_val == 0))
        fn = np.sum((prdiction == 0) & (y_val == 1))

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)

        F1 = (2*prec*rec) / (prec+rec)

        if F1 > best_F1:
            best_F1 = F1
            best_epi = epi

    return best_F1,best_epi

p_val = multivariateGaussian(x_val,mu,sigma2)
print(p_val.shape)

epsilon,F1 = selectThreadshold(y_val,p_val)
outlier = p < epsilon
print(outlier.shape)
print(sum(outlier))

visualizefit(x,mu,sigma2)
plt.xlabel("Latency (ms)")
plt.ylabel("Throughput (mb/s)")
plt.tight_layout()
plt.plot(x[outlier,0],x[outlier,1],'ro',ms=10,mfc="None")
plt.show()



