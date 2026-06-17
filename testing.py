import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegressionModel:
    def __init__(self,file_path):
        self.file_path=file_path
        self.x=None
        self.y=None
        self.m=None
        self.theta=None
        self.J_history=[]


    def load_data(self):
        data=pd.read_csv(self.file_path,header=None)
        data=data.to_numpy()
        x=data[:,0]
        y=data[:,1]
        self.m=y.size

        self.x=np.stack([np.ones(self.m),x],axis=1)
        self.y=y.reshape(self.m,1)
        self.theta=np.zeros((2,1))

    def plot_data(self):
        plt.plot(self.x[:,1],self.y,'ro',ms=8,mec='k')
        plt.ylabel("Profit in 10,000 $")
        plt.xlabel("Population of City in 10,000 $")

if __name__ == "__main__":
    main=LinearRegressionModel("C:/Users/User/Desktop/Machine learning exercise/data1/data/ex1data1.txt")
    main.load_data()
    main.plot_data()
