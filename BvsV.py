import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

mat = loadmat("C:/Users/User/Desktop/Machine learning exercise/data1/data/ex5data1.mat")

x=mat['X']
y=mat['y']

xv=mat['Xval']
yv=mat['yval']

xt=mat['Xtest']
yt=mat['ytest']

plt.scatter(x,y,marker='*',color='r')
plt.xlabel("Change in Water level")
plt.ylim(0,40)
plt.ylabel("Water flowing out of the dam")
plt.show()

def LinRegCostFunction(x,y,theta,Lambda):
    m = len(x)
    h= x @ theta
    cost = (1/(2*m) * np.sum((h-y)**2))
    regcost = cost +(((Lambda/(2*m)) * (np.sum(theta[1:]**2))))

    grad1= 1/m * x.T @ (h-y)
    grad2= 1/m * x.T @ (h-y) + (Lambda/m * theta)
    grad = np.vstack((grad1[0],grad2[1:]))

    return regcost, grad

m = x.shape[0]
x_1 =np.hstack((np.ones((m,1)),x))
theta = np.ones((2,1))
cost,grad = LinRegCostFunction(x_1,y,theta,1)
print("Cost at theta = [1;1] :",cost)
print("Gradient at theta = [1;1] :",grad)

def gradientDescent(x,y,theta,alpha,num_iters,Lambda):
    j_history = []
    for i in range(num_iters):
        cost,grad=LinRegCostFunction(x,y,theta,Lambda)
        theta = theta - (alpha * grad)
        j_history.append(cost)

    return theta, j_history

Lambda = 0
theta,j_history= gradientDescent(x_1,y,np.zeros((2,1)),0.001,4000,Lambda)
# print(theta)
# print(j_history)

plt.plot(j_history)
plt.xlabel(r"Number of Iterations")
plt.ylabel(r"$J(\Theta)$")
plt.title(r"Cost function using Gradient Descent")
plt.show()

plt.scatter(x,y,marker='*',color='r')
plt.scatter(x,y,marker="x",color="r")
x_value = [x for x in range(-50,40)]
y_value = [theta[0] + x*theta[1] for x in x_value]
plt.plot(x_value,y_value,color="b")
plt.xlabel("Change in water level")
plt.ylabel("Water flowing out of the dam")
plt.ylim(-5,40)
plt.xlim(-50,40)
plt.title("High bias problem")
plt.show()


def learningCurves(x,y,xv,yv,Lambda):
    m = len(x)
    n = x.shape[1]
    err_train, err_val = [],[]

    for i in range(1,m+1):
        theta = gradientDescent(x[0:i,:],y[0:i,:],np.zeros((n,1)),0.001,3000,Lambda)[0]
        err_train.append(LinRegCostFunction(x[0:i,:],y[0:i,:],theta,Lambda)[0])
        err_val.append(LinRegCostFunction(xv,yv,theta,Lambda)[0])

    return err_train, err_val

xval_1=np.hstack((np.ones((21,1)),xv))
err_train,err_val = learningCurves(x_1,y,xval_1,yv,Lambda)

plt.plot(err_train, label="Train")
plt.plot(err_val, label="Cross Validation", color="r")
plt.xlabel("Number of training examples")
plt.ylabel("Error")
plt.legend()
plt.show()

print("# Training Examples\t Train Error \t\t Cross Validation Error")
for i in range(1,13):
    print("\t",i,"\t\t",err_train[i-1],"\t",err_val[i-1],"\n")

def polyFeature(x,p):
    for i in range(2,p+1):
        x=np.hstack((x,(x[:,0]**i)[:,np.newaxis]))
    return x

p=8
x_poly=polyFeature(x,p)
print(x_poly)
print(x_poly.shape)

sc_x=StandardScaler()
x_poly=sc_x.fit_transform(x_poly)

x_poly=np.hstack((np.ones((x_poly.shape[0],1)),x_poly))
print(x_poly.shape)

x_poly_val=polyFeature(xv,p)
x_poly_val=sc_x.transform(x_poly_val)
x_poly_val= np.hstack((np.ones((x_poly_val.shape[0],1)),x_poly_val))

theta_ply, j_history_ply = gradientDescent(x_poly,y,np.zeros((9,1)),0.03,20000,Lambda)
# print(theta_ply)
# print(j_history_val)

plt.scatter(x,y,marker='*',color='r')
x_value=np.linspace(-55,65,2400)
x_value_poly=polyFeature(x_value[:,np.newaxis],p)
x_value_poly=sc_x.transform(x_value_poly)
x_value_poly=np.hstack((np.ones((x_value_poly.shape[0],1)),x_value_poly))
y_value = x_value_poly @ theta_ply
plt.plot(x_value,y_value,"--",color="b")
plt.xlabel("Change in water level")
plt.ylabel("Water flowing out of the dam")
plt.show()

err_train,err_val = learningCurves(x_poly,y,x_poly_val,yv,Lambda)
plt.plot(range(12),err_train,label="Train")
plt.plot(range(12),err_val,label="Cross Validation",color="r")
plt.title("Learning Curves for Linear Regression")
plt.xlabel("Number of training examples")
plt.ylabel("Error")
plt.legend()
plt.show()

print("# Training Examples\t Train Error \t\t Cross Validation Error")
for i in range(1,13):
    print("\t",i,"\t\t",err_train[i-1],"\t",err_val[i-1],"\n")

Lambda = 100
theta_poly, j_history_ply = gradientDescent(x_poly,y,np.zeros((9,1)),0.01,20000,Lambda)
plt.scatter(x,y,marker="x",color="r")
x_value=np.linspace(-55,65,2400)
x_value_poly = polyFeature(x_value[:,np.newaxis], p)
x_value_poly = sc_x.transform(x_value_poly)
x_value_poly = np.hstack((np.ones((x_value_poly.shape[0],1)),x_value_poly))
y_value= x_value_poly @ theta_poly
plt.plot(x_value,y_value,"--",color="b")
plt.xlabel("Change in water level")
plt.ylabel("Water flowing out of the dam")
plt.show()

err_train,err_val=learningCurves(x_poly,y,x_poly_val,yv,Lambda)
plt.plot(range(12),err_train,label="Train")
plt.plot(range(12),err_val,label="Cross Validation",color="r")
plt.title("Learning Curves for Polynomial Regression")
plt.xlabel("Number of training examples")
plt.ylabel("Error")
plt.legend()
plt.show()

def validationCurve(x,y,xv,yv):
    lambda_vec=[0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]
    m = len(x)
    n = x.shape[1]
    err_train, err_val = [], []

    for i in range(len(lambda_vec)):
        Lambda_try = lambda_vec[i]
        theta = gradientDescent(x,y,np.zeros((n,1)),0.001,3000,Lambda_try)[0]
        err_train.append(LinRegCostFunction(x,y,theta,0)[0])
        err_val.append(LinRegCostFunction(xv,yv,theta,0)[0])

    return lambda_vec, err_train, err_val

lambda_vec, err_train, err_val = validationCurve(x_poly,y,x_poly_val,yv)
print('lambda\t\tTrain Error\tValidation Error')
for i in range(len(lambda_vec)):
    print(' %f\t%f\t%f' % (lambda_vec[i], err_train[i], err_val[i]))


plt.plot(lambda_vec,err_train,'--',lambda_vec,err_val,'--',lw=2)
plt.legend(['Train','Cross Validation'])
plt.xlabel("lambda")
plt.ylabel("Error")
plt.show()




