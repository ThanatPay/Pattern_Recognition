import numpy as np
import math

# logistic for binary class
class LogisticRegression:
    def __init__(self,data,target):
        self.data=data
        self.m=len(data)
        self.target=target
        self.theta=np.random.rand(data.shape[1])

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-np.array(x,dtype=float)))
        
    def adjust_theta(self,theta,learning_rate):
        prob=self.probability(self.data,theta)
        gd=np.matmul(self.target-prob,self.data)
        return theta+(learning_rate*gd)

    def log_likelihood_error(self,pred,target):
        epsilon = 1e-15
        pred = np.clip(pred, epsilon, 1 - epsilon)
        return np.mean((-target*np.log(pred))-((1-target)*np.log(1-pred)))

    def probability(self,data,theta):
        return self.sigmoid(np.matmul(data,theta))
    
    def prediction(self,data,theta,threshold):
        prob=self.probability(data,theta)
        pred=np.array(prob>=threshold,dtype=int)
        return pred
        
    def train(self, epoch, learning_rate):
        # first predict
        best_mle=math.inf
        
        for i in range(epoch):
            # adjust theta
            self.theta=self.adjust_theta(self.theta,learning_rate)
            prob=self.probability(self.data,self.theta) 
            mle=self.log_likelihood_error(self.target,prob)
            if mle < best_mle:
                best_mle=mle
                best_theta=self.theta
            print("epoch",i)
            print("loss",mle)
            print("---------------")

        return best_theta,best_mle