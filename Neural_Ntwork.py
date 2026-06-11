import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat('C:/Users/User/Desktop/Machine learning exercise/data1/data/ex3data1.mat')
x=data['X']
y=data['y']

# m=len(x)
# # x=np.hstack((np.ones((m,1)),x))
# theta = np.zeros((x.shape[1] +1,1)

def sigmoid(z):
    return 1 / (1+np.exp(-z))

def cost_function(x,y,theta,Lambda):
    m = len(x)
    z = x @ theta
    h = sigmoid(z)
    l = ((-y * np.log(h)) - ((1-y) * np.log(1-h)))
    cost = np.sum(l) / m
    reg = Lambda / (2*m) * np.dot(theta[1:].T,theta[1:])
    reg_cost = cost + reg

    j_0 =(1/m) * (np.dot(x.T,(h-y)))[0]
    j_1 =(1/m) * (np.dot(x.T,(h-y)))[1] + (Lambda/m) * theta[1:]

    grad = np.vstack((j_0[:np.newaxis],j_1))

    return reg_cost.item(), grad


theta_t = np.array([-2,-1,1,2]).reshape(4,1)
x_t =np.array([np.linspace(0.1,1.5,15)]).reshape(3,5).T
x_t = np.hstack((np.ones((5,1)), x_t))
y_t = np.array([1,0,1,0,1]).reshape(5,1)
J, grad = cost_function(x_t,y_t,theta_t,3)
# print("Cost:",J,"Expected cost: 2.534819")
# print("Gradients:\n",grad,"\nExpected gradients:\n 0.146561\n -0.548558\n 0.724722\n 1.398003")

def gradient_Descent(x,y,theta,alpha,num_itera,Lambda):
    m = len(x)
    j_history = []
    for i in range(num_itera):
        cost, grad = cost_function(x,y,theta,Lambda)
        theta = theta - (alpha * grad)
        j_history.append(cost)

    return theta, j_history

def one_vs_all(x,y,num_label,Lambda):
    m = x.shape[0]
    n = x.shape[1]
    init_theta = np.zeros((n+1,1))
    all_theta =[]
    all_j = []
    x = np.hstack((np.ones((m,1)),x))

    for i in range(1,num_label+1):
         theta , j_history = gradient_Descent(x,np.where(y==i,1,0),init_theta,1,300,Lambda)
         all_theta.extend(theta)
         all_j.extend(j_history)

    return np.array(all_theta).reshape(num_label,n+1),all_j

all_theta,all_j = one_vs_all(x,y,10,0.1)


plt.plot(all_j[0:300])
plt.xlabel('Iteration')
plt.ylabel(r"$J(\theta)$")
plt.title("Cost Function Using Gradient Descent")
plt.show()

def predictonevsall(all_theta,x):
    m = x.shape[0]
    x = np.hstack((np.ones((m,1)),x))
    prediction = np.dot(x,all_theta.T)
    return np.argmax(prediction,axis=1)+1

pred = predictonevsall(all_theta,x)
pred = pred.reshape(5000,1)
np.sum(pred==y)
np.sum(pred == y) / 5000
print("Training Set Accuracy:",sum(pred == y)[0]/5000*100,"%")

mat = loadmat("C:/Users/User/Desktop/Machine learning exercise/data1/data/ex3weights.mat")
theta1=mat['Theta1']
theta2=mat['Theta2']
print(theta1.shape)
print(theta2.shape)

def predict(theta1,theta2,x):
    m = x.shape[0]
    a1 = np.hstack((np.ones((m,1)),x))
    a2 = sigmoid(a1 @ theta1.T)
    a2 = np.hstack((np.ones((m,1)),a2))
    h = sigmoid(a2 @ theta2.T)

    return np.argmax(h,axis=1)+1

pred2 = predict(theta1,theta2,x)
print(pred2.shape)
pred2 = pred2.reshape(5000,1)
print("Training Set Accuracy:",sum(pred2 == y)[0] / 5000*100,"%")


def sigmoic_grad(z):
    s = sigmoid(z)
    return s * (1-s)

print(x.shape)
print(y.shape)

def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,x,y,Lambda):
    theta1 =nn_params[:((input_layer_size+1) * hidden_layer_size)].reshape(hidden_layer_size,input_layer_size + 1)
    theta2 =nn_params[((input_layer_size+1) * hidden_layer_size):].reshape(num_labels,hidden_layer_size+1)

    m = x.shape[0]
    j = 0
    a1 = np.hstack((np.ones((m,1)),x))
    y10 = np.zeros((m,num_labels))
    a2 =sigmoid(a1 @ theta1.T)
    a2 = np.hstack((np.ones((m,1)),a2))

    h = sigmoid(a2 @ theta2.T)

    for i in range(1,num_labels+1):
        y10[:,i-1][:,np.newaxis] = np.where(y==i,1,0)

    cost = np.sum(np.multiply(-y10,np.log(h)) - np.multiply(1-y10,np.log(1-h))) / m
    reg =Lambda/(2*m) * np.sum(theta1[:,1:]**2) + np.sum(theta2[:,1:]**2)
    reg_cost = cost + reg

    grad1=np.zeros((theta1.shape))
    grad2=np.zeros((theta2.shape))

    z2 = np.dot(a1,theta1.T)
    delta3 = h - y10
    delta2 = np.multiply(np.dot(delta3,theta2)[:,1:], sigmoic_grad(z2))

    grad2 =np.dot(delta3.T,a2)/m
    grad1 =np.dot(delta2.T,a1)/m

    grad1_reg = grad1 + (Lambda/m) * np.hstack((np.zeros((theta1.shape[0],1)),theta1[:,1:]))
    grad2_reg = grad2 + (Lambda/m) * np.hstack((np.zeros((theta2.shape[0],1)),theta2[:,1:]))

    return cost, grad1, grad2, reg_cost, grad1_reg, grad2_reg


input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

nn_params = np.append(theta1.flatten(),theta2.flatten())

J, reg_j =nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,x,y,1)[0:4:3]

print("Cost at parameters (non-regularized):",J,"\nCost at parameters (Regularized):",reg_j)


def random_initialize_weight(L_in,L_out):
    epi = (6**1/2) /(L_in + L_out)**1/2
    w = np.random.rand(L_out,L_in + 1) * (2*epi) - epi
    return w

initial_theta1 = random_initialize_weight(input_layer_size,hidden_layer_size)
initial_theta2 = random_initialize_weight(hidden_layer_size,num_labels)

initial_nn_params = np.append(initial_theta1.flatten(),initial_theta2.flatten())

def gradientDescent(x,y,initial_nn_params,alpha,num_ite,Lambda,input_layer_size,hidden_layer_size,num_labels):
    theta1 =initial_nn_params[:((input_layer_size+1) * hidden_layer_size)].reshape(hidden_layer_size,input_layer_size + 1)
    theta2 =initial_nn_params[((input_layer_size+1) * hidden_layer_size):].reshape(num_labels,hidden_layer_size+1)

    m = len(y)
    j_history = []

    for i in range(num_ite):
        nn_params = np.append(theta1.flatten(),theta2.flatten())
        cost, grad1, grad2 =nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,x,y,Lambda) [3:]

        theta1 = theta1 - (alpha*grad1)
        theta2 = theta2 - (alpha*grad2)

        j_history.append(cost)

    nn_params_final = np.append(theta1.flatten(),theta2.flatten())
    return nn_params_final, j_history

nn_theta, nn_j_history = gradientDescent(x,y,initial_nn_params,0.8,800,1,input_layer_size,hidden_layer_size,num_labels)
print(nn_theta.shape)
# print(nn_j_history.shpae)

plt.plot(nn_j_history[0:800])
plt.show()

Theta1 = nn_theta[:((input_layer_size+1) * hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)
Theta2 = nn_theta[((input_layer_size +1)* hidden_layer_size ):].reshape(num_labels,hidden_layer_size+1)

pred3 = predict(Theta1,Theta2,x)
print("Training Set Accuracy:",sum(pred3[:,np.newaxis]==y)[0]/5000*100,"%")