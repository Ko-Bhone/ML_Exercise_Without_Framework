import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

class DataLoader:
    def __init__(self, path_x, path_theta):
        self.path_x = path_x
        self.path_theta = path_theta

    def data_loader(self):
        data = loadmat(self.path_x)
        x = data["X"]
        y = data["y"]
        return x,y

    def weight_loader(self):
        if self.path_theta:
            mat = loadmat(self.path_theta)
            return mat["Theta1"], mat["Theta2"]
        return None, None

class NeuralNetworkML:
    def __init__(self,input_size=400, hidden_size=25,num_labels=10,Lambda=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.Lambda = Lambda

        self.theta1 = None
        self.theta2 = None
        self.cost_history = []

    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_grad(self,z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def random_init(self,L_in,L_out):
        epi = np.sqrt(6) / np.sqrt(L_in + L_out)
        return np.random.rand(L_in,L_out) * (2 * epi) - epi

    def forward(self,x,theta1,theta2):
        m = x.shape[0]
        a1 = np.hstack((np.ones((m, 1)), x))
        z2 = a1 @ theta1.T
        a2 = self.sigmoid(z2)
        a2 = np.hstack((np.ones((m, 1)), a2))
        z3 = a2 @ theta2.T
        h = self.sigmoid(z3)
        return a1, a2, h , z2

    def nn_cost_grad(self,nn_params,x,y):
        Theta1 = nn_params