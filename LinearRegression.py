import numpy as np

class LinearRegression:
    def __init__(self, data, target, learning_rate, epoch):
        self.data=data
        self.m=len(data)
        self.target=target
        self.theta=np.random.rand(data.shape[1])
        print(self.theta)
        self.lr=learning_rate
        self.epoch=epoch
        
    def adjust_theta(self,theta):
        pred=self.prediction(self.data,theta)
        gd=np.matmul(self.target-pred,self.data)
        return theta+(self.lr*gd)

    def mean_square_error(self,pred,target):
        return np.mean((target-pred)**2)

    def prediction(self,data,theta):
        return np.matmul(data,theta)
        
    def train(self):
        # first predict
        pred=self.prediction(self.data, self.theta)
        print("predict",pred)
        
        for i in range(self.epoch):
            # adjust theta
            self.theta=self.adjust_theta(self.theta)
            pred=self.prediction(self.data,self.theta) 
            mse=self.mean_square_error(self.target,pred)
            print("epoch",i)
            print("theta",self.theta)
            print("predict",pred)
            print("loss",mse)
            print("---------------")

        return self.theta,mse
    
data=np.array([[1,2],[2,3],[4,5]])
target=np.array([1,2,4])
l=LinearRegression(data,target,0.01,10)
l.train()