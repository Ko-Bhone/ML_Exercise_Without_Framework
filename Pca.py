import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy.linalg import svd

mat = loadmat("C:/Users/User/Desktop/Machine learning exercise/data1/data/ex7/ex7data1.mat")
x=mat["X"]
print(x.shape)

plt.scatter(x[:,0],x[:,1],marker='o',facecolors="none",edgecolors="b")
plt.show()

def FeatureNOrmalize(x):
    mu=np.mean(x,axis=0)
    sigma = np.std(x,axis=0)
    x_nom=(x-mu) / sigma
    return x_nom,mu,sigma

def pca(x):
    m = x.shape[0]
    sigma = (1/m) * (x.T @ x)
    u,s,v = svd(sigma)
    return u,s,v

x_nom,mu,std=FeatureNOrmalize(x)
u,s =  pca(x_nom)[:2]
# print(x_nom)
# print(mu)
# print(std)
plt.scatter(x[:,0],x[:,1],marker="o",facecolors="none",edgecolors="b")
plt.plot([mu[0],(mu+1.5*s[0]*u[:,0].T)[0]],[mu[1],(mu+1.5*s[0]*u[:,0].T)[1]],color="black",linewidth=3)
plt.plot([mu[0],(mu+1.5*s[1]*u[:,1].T)[0]],[mu[1],(mu+1.5*s[1]*u[:,1].T)[1]],color="black",linewidth=3)
plt.xlim(-1,7)
plt.ylim(2,8)
plt.show()

print("Top eigenvector u(:,1) = :",u[:,0])

def projectData(x,u,k):
    m  = x.shape[0]
    u_reduced = u[:,:k]
    z= np.zeros((m,k))
    z = x @ u_reduced
    return z

k = 1
z = projectData(x_nom, u, k)
print("Projection of the first example:",z[0][0])


def recoverData(z,u,k):
    m,n = z.shape[0],u.shape[0]
    x_rec = np.zeros((m,n))
    u_reduce = u[:,:k]
    x_rec = z @ u_reduce.T

    return x_rec

x_rec = recoverData(z,u,k)
print("Approximation of the first example:",x_rec[0,:])

plt.scatter(x_nom[:,0],x_nom[:,1],marker="o",label="Original",facecolors="none",edgecolors="b",s=15)
plt.scatter(x_rec[:,0],x_rec[:,1],marker="o",label="Approximation",facecolors="none",edgecolors="r",s=15)
plt.title("The Normalized and Projected Data after PCA")
plt.legend()
plt.show()


mat2=loadmat("C:/Users/User/Desktop/Machine learning exercise/data1/data/ex7/ex7faces.mat")
x2=mat2["X"]
print(x2.shape)

fig,ax = plt.subplots(nrows=10,ncols=10,figsize=(8,8))
for i in range(0,100,10):
    for j in range(10):
        ax[int(i/10),j].imshow(x2[i+j,:].reshape(32,32,order="F"),cmap="gray")
        ax[int(i/10),j].axis("off")
        plt.show()

x_nom2=FeatureNOrmalize(x2)[0]
u2 = pca(x_nom2)[0]

u_reduce = u2[:,:36].T
fig2,ax2 = plt.subplots(6,6,figsize=(8,8))
for i in range(0,36,6):
    for j in range(6):
        ax2[int(i/6),j].ishow(u_reduce[i+j,:].reshape(32,32,order="F"),cmap="gray")
        ax2[int(i/6),j].axis("off")
        plt.show()

k2 = 100
z2=projectData(x_nom2,k2)
x_rec2 = recoverData(z2,u2,k2)
fig3,ax3 = plt.subplots(10,10,figsize=(8,8))
for i in range(0,100,100):
    for j in range(10):
        ax3=[int(i/10),j].imshow(x_rec2[i+j,:].reshape(32,32,order="F"),cmap="gray")
        ax3=[int(i/10),j].axis("Off")
        plt.show()