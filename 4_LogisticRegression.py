import numpy as np
import math

# logistic for binary class
class LogisticRegression:
    def adjust_weight(self,X,y,theta,bias,learning_rate):
        y_prob=self.probability(X,theta,bias)
        d_theta=np.matmul(y-y_prob,X)
        d_bias=np.sum(y-y_prob)
        return theta+(learning_rate*d_theta),bias+(learning_rate*d_bias)

    def log_likelihood_error(self,y,y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return np.mean((-y*np.log(y_pred))-((1-y)*np.log(1-y_pred)))
    
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-np.array(x,dtype=float)))

    def probability(self,X,theta,bias):
        return self.sigmoid(np.matmul(X,theta)+bias)
    
    def predict(self,X,theta,bias,threshold):
        prob=self.probability(X,theta,bias)
        return np.array(prob>=threshold,dtype=int)
        
    def train(self, X, y, epoch, learning_rate, batch_size):
        # Convert to NumPy arrays if not already
        X=np.array(X)
        y=np.array(y)
        # random first theta and bias
        theta=np.random.rand(X.shape[1])
        bias=np.random.rand(1)
        # set batch
        num_batch=X.shape[0]//batch_size
        X_batch=np.reshape(X,(num_batch,batch_size,X.shape[1]))
        y_batch=np.reshape(y,(num_batch,batch_size))
        # for select best loss
        best_mle=math.inf
        
        for i in range(epoch):
            for j in range(num_batch):
                # adjust theta
                theta,bias=self.adjust_weight(X_batch[j],y_batch[j],theta,bias,learning_rate)
            # calaulate loss
            y_pred=self.probability(X,theta,bias) 
            mle=self.log_likelihood_error(y,y_pred)
            if mle < best_mle:
                best_mle=mle
                best_theta=theta
                best_bias=bias
            print("epoch",i)
            print("loss",mle)
            print("---------------")

        return best_theta,best_bias,best_mle
    
X=np.array([[1,1],[0,0],[3,3],[2,1],[4,3],[4,4]])
y=np.array([1,1,0,1,0,0])
L2=LogisticRegression()
theta,bias,_=L2.train(X,y,1000,0.01,6)
print(theta,bias)
print(L2.probability(X,theta,bias))
print(L2.predict(X,theta,bias,0.5))
