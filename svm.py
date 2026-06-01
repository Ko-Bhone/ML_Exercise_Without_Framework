import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.svm import SVC

mat = loadmat("C:/Users/User/Desktop/Machine learning exercise/data1/data/ex6/ex6data1.mat")
x=mat["X"]
y=mat["y"]

print(x.shape)
print(y.shape)

m,n = x.shape[0],x.shape[1]
pos,neg=(y==1).reshape(m,1),(y==0).reshape(m,1)
plt.scatter(x[pos[:,0],0],x[pos[:,0],1],c="r",marker="+",s=50)
plt.scatter(x[neg[:,0],0],x[neg[:,0],1],c='y',marker="o",s=50)
plt.show()

classifier = SVC(kernel="linear")
classifier.fit(x,np.ravel(y))
plt.figure(figsize=(8,6))
plt.scatter(x[pos[:,0],0],x[pos[:,0],1],c="r",marker="+",s=50)
plt.scatter(x[neg[:,0],0],x[neg[:,0],1],c="y",marker="o",s=50)
x_1,x_2 = np.meshgrid(np.linspace(x[:,0].min(),x[:,1].max(),num=100),np.linspace(x[:,1].min(),x[:,1].max(),num=100))
plt.contour(x_1,x_2,classifier.predict(np.array([x_1.ravel(),x_2.ravel()]).T).reshape(x_1.shape),1,colors="b")
plt.xlim(0,4.5)
plt.ylim(1.5,5)
plt.show()

classifier2=SVC(C=100,kernel="linear")
classifier2.fit(x,np.ravel(y))
plt.figure(figsize=(8,6))
plt.scatter(x[pos[:,0],0],x[pos[:,0],1],c="r",marker="+",s=50)
plt.scatter(x[neg[:,0],0],x[neg[:,0],1],c="y",marker="o",s=50)
x_3,x_4 = np.meshgrid(np.linspace(x[:,0].min(),x[:,1].max(),num=100),np.linspace(x[:,1].min(),x[:,1].max(),num=100))
plt.contour(x_3,x_4,classifier2.predict(np.array([x_3.ravel(),x_4.ravel()]).T).reshape(x_3.shape),1,colors="b")
plt.xlim(0,4.5)
plt.ylim(1.5,5)
plt.show()


mat2=loadmat("C:/Users/User/Desktop/Machine learning exercise/data1/data/ex6/ex6data2.mat")
x2=mat2["X"]
y2=mat2["y"]

m2,n2 =x2.shape[0],x2.shape[1]
pos2,neg2= (y2==1).reshape(m2,1),(y2==0).reshape(m2,1)
plt.figure(figsize=(8,6))
plt.scatter(x2[pos2[:,0],0],x2[pos2[:,0],1],c="r",marker="+")
plt.scatter(x2[neg2[:,0],0],x2[neg2[:,0],1],c="y",marker="o")
plt.xlim(0,1)
plt.ylim(0.4,1)
plt.show()


classifier3=SVC(kernel="rbf",gamma=30)
classifier3.fit(x2,y2.ravel())

plt.figure(figsize=(8,6))
plt.scatter(x2[pos2[:,0],0],x2[pos2[:,0],1],c="r",marker="+")
plt.scatter(x2[neg2[:,0],0],x2[neg2[:,0],1],c="y",marker="o")
x_5,x_6 = np.meshgrid(np.linspace(x2[:,0].min(),x2[:,1].max(),num=100),np.linspace(x2[:,1].min(),x2[:,1].max(),num=100))
plt.contour(x_5,x_6,classifier3.predict(np.array([x_5.ravel(),x_6.ravel()]).T).reshape(x_5.shape),1,colors="b")
plt.xlim(0,1)
plt.ylim(0.4,1)
plt.show()


mat3=loadmat("C:/Users/User/Desktop/Machine learning exercise/data1/data/ex6/ex6data3.mat")
x3=mat3["X"]
y3=mat3["y"]

x_val = mat3["Xval"]
y_val = mat3["yval"]

m3,n3 =x3.shape[0],x3.shape[1]
pos3,neg3 = (y3==1).reshape(m3,1),(y3==0).reshape(m3,1)
plt.figure(figsize=(8,6))
plt.scatter(x3[pos3[:,0],0],x3[pos3[:,0],1],c="r",marker='+',s=50)
plt.scatter(x3[neg3[:,0],0],x3[neg3[:,0],1],c="y",marker='o',s=50)
plt.show()

def dataset3Parms(x,y,x_val,y_val,vals):
    acc=0
    best_c=0
    best_gamma=0

    for i in vals:
        C = i
        for j in vals:
            gamma =1/j
            classifier =SVC(C=C,gamma=gamma)
            classifier.fit(x,y)
            prediction=classifier.predict(x_val)
            score = classifier.score(x_val,y_val)

            if score > acc:
                 acc = score
                 best_c = C
                 best_gamma = gamma

    return best_c, best_gamma

vals =[0.001,0.03,0.1,0.3,1,3,10,30]

C,gamma =dataset3Parms(x3,y3.ravel(),x_val,y_val.ravel(),vals)

classifier4 =SVC(C=C,gamma=gamma)
classifier4.fit(x3,y3.ravel())
SVC(C=0.3,gamma=100.0)
plt.figure(figsize=(8,6))
plt.scatter(x3[pos3[:,0],0],x3[pos3[:,0],1],c="r",marker="+",s=50)
plt.scatter(x3[neg3[:,0],0],x3[neg3[:,0],1],c="y",marker="o",s=50)

x_7,x_8=np.meshgrid(np.linspace(x3[:,0].min(),x3[:,1].max(),num=100),np.linspace(x3[:,1].min(),x3[:,1].max(),num=100))
plt.contour(x_7,x_8,classifier4.predict(np.array([x_7.ravel(),x_8.ravel()]).T).reshape(x_7.shape),1,colors="b")
plt.xlim(-0.6,0.3)
plt.ylim(-0.7,0.5)
plt.show()