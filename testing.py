import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.gridspec import GridSpec

class DigitClassifier:
    def __init__(self, data_path, alpha=0.1, iterations=300, Lambda=1):
        self.data_path = data_path
        self.alpha = alpha
        self.iterations = iterations
        self.Lambda = Lambda
        # Data
        self.X = None
        self.y = None
        # Logistic
        self.all_theta = None
        self.log_cost_history = []
        # Neural Network
        self.theta1 = None
        self.theta2 = None
        self.nn_cost_history = []

    # Activation Function
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def sigmoid_grad(self,z):
        s = self.sigmoid(z)
        return s*(1-s)

    # Load Data
    def load_data(self):
        data = loadmat(self.data_path)
        self.X = data["X"]
        self.y = data["y"]
        print("Dataset Loaded:", self.X.shape)

    # Logistic Regression
    def logistic_cost(self, X, y, theta):
        m = len(y)
        z = X @ theta
        h = self.sigmoid(z)
        h = np.clip(h,1e-10, 1-1e-10)
        # Error
        cost = np.sum(-y*np.log(h) - (1-y)*np.log(1-h) ) / m
        reg = (self.Lambda/(2*m)) * np.sum(theta[1:]**2)
        total_cost = cost + reg
        grad = (X.T @ (h-y)) / m
        grad[1:] += (self.Lambda/m) * theta[1:]
        return total_cost,grad

    def gradient_descent(self, X, y, theta ):
        history=[]
        for i in range(self.iterations):
            cost,grad = self.logistic_cost(X, y, theta)
            theta -= self.alpha*grad
            history.append(cost)
        return theta,history

    def train_logistic(self):
        m,n = self.X.shape
        X = np.hstack((np.ones((m,1)),self.X))
        self.all_theta = np.zeros((10,n+1))
        for digit in range(10):
            y_digit = (self.y == digit+1).astype(int)
            theta = np.zeros((n+1,1))
            theta,history = self.gradient_descent(X,y_digit,theta)
            self.all_theta[digit] = (theta.ravel())
            self.log_cost_history.append(history)
        print("Logistic Training Finished")

    def predict_logistic(self):
        m = self.X.shape[0]
        X = np.hstack((np.ones((m,1)), self.X))
        score = (X @ self.all_theta.T)
        pred = (np.argmax(score,axis=1)+1)
        accuracy = np.mean(pred.reshape(-1,1)==self.y)*100
        print(f"Logistic Accuracy: {accuracy:.2f}%")
        return pred

    # Neural Network
    def random_init(self,L_in,L_out):
        epsilon = (np.sqrt(6)/np.sqrt(L_in+L_out))
        return (np.random.rand(L_out,L_in+1) * 2 * epsilon - epsilon)

    def one_hot(self):
        m = self.y.shape[0]
        y_matrix = np.zeros((m,10))
        for i in range(10):
            y_matrix[:,i] = (self.y.flatten()==i+1).astype(int)
        return y_matrix

    def forward(self,X):
        m = X.shape[0]
        # Layer 1
        a1 = np.hstack((np.ones((m,1)),X))
        # Layer 2
        z2 = (a1 @ self.theta1.T)
        a2 = self.sigmoid(z2)
        a2 = np.hstack((np.ones((m,1)),a2))
        # Output
        z3 = (a2 @ self.theta2.T)
        h = self.sigmoid(z3)
        return a1,z2,a2,h

    def nn_cost(self,h, y):
        m = y.shape[0]
        h = np.clip(h,1e-10,1-1e-10)
        cost = np.sum(-y*np.log(h)-(1-y)*np.log(1-h))/m
        reg = (self.Lambda/(2*m)) * (np.sum(self.theta1[:,1:]**2)+np.sum(self.theta2[:,1:]**2))
        return cost+reg

    def backprop(self,a1,z2,a2,h,y):
        m = y.shape[0]
        delta3 = h-y
        delta2 = (delta3 @ self.theta2)[:,1:] * self.sigmoid_grad(z2)
        grad1 = (delta2.T @ a1)/m
        grad2 = (delta3.T @ a2)/m
        return grad1,grad2

    def train_nn(self):
        self.theta1 = self.random_init(400,25)
        self.theta2 = self.random_init(25,10)
        y_matrix = self.one_hot()
        m = self.X.shape[0]
        for i in range(self.iterations):
            a1,z2,a2,h = self.forward(self.X)
            cost = self.nn_cost(h,y_matrix)
            grad1,grad2 = self.backprop(a1, z2, a2, h, y_matrix)
            grad1[:,1:] += (self.Lambda/m) * self.theta1[:,1:]
            grad2[:,1:] += (self.Lambda/m) * self.theta2[:,1:]
            self.theta1 -= (self.alpha*grad1)
            self.theta2 -= (self.alpha*grad2)
            self.nn_cost_history.append(cost)
        print("Neural Network Training Finished")

    def predict_nn(self):
        _,_,_,h = self.forward(self.X)
        pred = (np.argmax(h,axis=1)+1)
        accuracy = np.mean(pred.reshape(-1,1)==self.y)*100
        print(f"NN Accuracy: {accuracy:.2f}%")
        return pred

    # Visualization
    def show_cost(self):
        plt.figure(figsize=(12,5))
        plt.plot(np.mean(self.log_cost_history,axis=0),label="Logistic")
        plt.plot(self.nn_cost_history,label="Neural Network")
        plt.legend()
        plt.grid()
        plt.show()

    def show_random_prediction(self,pred):
        index = np.random.randint(0,len(self.X))
        image = self.X[index].reshape(20,20).T
        plt.imshow(image,cmap="viridis")
        plt.title(f"Actual:{self.y[index][0]} | Pred:{pred[index]}")
        plt.axis("off")
        plt.show()

# Main Program
if __name__ == "__main__":
    model = DigitClassifier(
        data_path="C:/Users/User/Desktop/Machine learning exercise/data1/data/ex3data1.mat", alpha=0.1, iterations=300, Lambda=1)
    # Load
    model.load_data()
    # Logistic Regression
    model.train_logistic()
    logistic_pred = model.predict_logistic()
    # Neural Network
    model.train_nn()
    nn_pred = model.predict_nn()
    # Graph
    model.show_cost()
    # Test Image
    model.show_random_prediction(nn_pred)