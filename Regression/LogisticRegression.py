import numpy as np
import math

# logistic for binary class
class LogisticRegression:
    def __init__(self):
        self.check_weight=False

    def adjust_weight(self,X,y,learning_rate):
        y_prob=self.probability(X)
        d_theta=np.matmul(y-y_prob,X)
        d_bias=np.sum(y-y_prob)
        return self.theta+(learning_rate*d_theta),self.bias+(learning_rate*d_bias)

    def log_likelihood_error(self,y,y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return np.mean((-y*np.log(y_pred))-((1-y)*np.log(1-y_pred)))
    
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-np.array(x,dtype=float)))

    def probability(self,X):
        return self.sigmoid(np.matmul(X,self.theta)+self.bias)
    
    def predict(self,X,threshold):
        prob=self.probability(X)
        return np.array(prob>=threshold,dtype=int)
        
    def train(self, X, y, epoch, learning_rate, batch_size):
        # Convert to NumPy arrays if not already
        X=np.array(X)
        y=np.array(y)

        if not self.check_weight:
            # random first theta and bias
            self.theta=np.random.rand(X.shape[1])
            self.bias=np.random.rand(1)
            self.check_weight=True
        # set batch
        num_batch=X.shape[0]//batch_size
        X_batch=np.reshape(X,(num_batch,batch_size,X.shape[1]))
        y_batch=np.reshape(y,(num_batch,batch_size))
        # for select best loss
        self.best_mle=math.inf
        
        for i in range(epoch):
            print("epoch",i)
            for j in range(num_batch):
                # adjust theta
                self.theta,self.bias=self.adjust_weight(X_batch[j],y_batch[j],learning_rate)
            # calaulate loss
            y_pred=self.probability(X) 
            mle=self.log_likelihood_error(y,y_pred)
            if mle < self.best_mle:
                self.best_mle=mle
                self.best_theta=self.theta
                self.best_bias=self.bias
            print("loss",mle)
            print("---------------")

        return self.best_theta,self.best_bias,self.best_mle
    
    def set_weight(self,theta,bias):
        self.theta=np.array(theta)
        self.bias=np.array(bias)
        self.check_weight=True
        
    
# X=np.array([[1,1],[0,0],[3,3],[2,1],[4,3],[4,4]])
# y=np.array([1,1,0,1,0,0])
# L2=LogisticRegression()
# theta,bias,_=L2.train(X,y,1000,0.01,6)
# print(theta,bias)
# print(L2.probability(X))
# print(L2.predict(X,0.5))
