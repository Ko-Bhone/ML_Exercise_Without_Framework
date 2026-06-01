import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat("C:/Users/User/Desktop/Machine learning exercise/data1/data/ex8/ex8_movies.mat")
Y,R = data['Y'], data['R']
print(Y.shape)
print(R.shape)

data=loadmat("C:/Users/User/Desktop/Machine learning exercise/data1/data/ex8/ex8_movieParams.mat")


x, Theta, num_users, num_movies, num_features = data["X"], data["Theta"], data["num_users"], data['num_movies'], data["num_features"]

print("X"+str(x))
print("\n Theta" + str(Theta))
print("Users" + str(num_users))
print("movies" + str(num_movies))
print("Features" + str(num_features))

print(x.shape)
print(Theta.shape)

def colfiCostfunction(params,Y,R,num_users,num_movies,num_features,Lambda=0.0):
    x=params[:num_movies * num_features].reshape(num_movies,num_features)
    Theta = params[num_movies * num_features:].reshape(num_users,num_features)

    J = 0
    X_grad = np.zeros(x.shape)
    Theta_grad = np.zeros(Theta.shape)

    reg_cost = (Lambda/2) * np.sum(np.square(Theta)) + (Lambda/2) * np.sum(np.square(x))
    J = 0.5 * np.sum(np.square((x.dot(Theta.T) - Y) * R)) + reg_cost

    x_grad = ((x.dot(Theta.T) - Y) * R) @ Theta + (Lambda @ x)
    Theta_grad = ((x.dot(Theta.T) - Y) * R).T @ x + (Lambda @ Theta)

    grad = np.concatenate([x_grad.ravel(),Theta_grad.ravel()])
    return J, grad


X, Theta, num_users, num_movies, num_features = data['X'],\
        data['Theta'], data['num_users'], data['num_movies'], data['num_features']

num_users = 4
num_movies = 5
num_features = 3

X = X[:num_movies, :num_features]
Theta = Theta[:num_users, :num_features]
Y = Y[:num_movies, 0:num_users]
R = R[:num_movies, 0:num_users]

J , _ = colfiCostfunction(np.concatenate([X.ravel(), Theta.ravel()]), Y,R, num_users, num_movies, num_features)

def computeNumericalGradient(J, theta, e=1e-4):
    numgrad = np.zeros(theta.shape)
    perturb = np.diag(e * np.ones(theta.shape))
    for i in range(theta.size):
        loss1, _ = J(theta - perturb[:, i])
        loss2, _ = J(theta + perturb[:, i])
        numgrad[i] = (loss2 - loss1)/(2*e)
    return numgrad

def checkCostFunction(cofiCostFunc, lambda_=0.):
    # Create small problem
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)

    # Zap out most entries
    Y = np.dot(X_t, Theta_t.T)
    Y[np.random.rand(*Y.shape) > 0.5] = 0
    R = np.zeros(Y.shape)
    R[Y != 0] = 1

    # Run Gradient Checking
    X = np.random.randn(*X_t.shape)
    Theta = np.random.randn(*Theta_t.shape)
    num_movies, num_users = Y.shape
    num_features = Theta_t.shape[1]

    params = np.concatenate([X.ravel(), Theta.ravel()])
    numgrad = computeNumericalGradient(
        lambda x: cofiCostFunc(x, Y, R, num_users, num_movies, num_features, lambda_), params)

    cost, grad = cofiCostFunc(params, Y, R, num_users,num_movies, num_features, lambda_)

    print(np.stack([numgrad, grad], axis=1))
    print('\nThe above two columns you get should be very similar.'
          '(Left-Your Numerical Gradient, Right-Analytical Gradient)')

    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print('If your cost function implementation is correct, then '
          'the relative difference will be small (less than 1e-9).')
    print('\nRelative Difference: %g' % diff)