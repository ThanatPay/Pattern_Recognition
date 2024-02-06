import numpy as np
import math

class LinearRegression:
    def adjust_weight(self,X,y,theta,bias,learning_rate):
        y_pred=self.predict(X,theta,bias)
        d_theta=np.matmul(y-y_pred,X)
        d_bias=np.sum(y-y_pred)
        return theta+(learning_rate*d_theta),bias+(learning_rate*d_bias)

    def mean_square_error(self,y,y_pred):
        return np.mean((y-y_pred)**2)

    def predict(self,X,theta,bias):
        return np.matmul(X,theta)+bias
        
    def train(self, X, y, epoch, learning_rate, batch_size):
        # Convert to NumPy arrays if not already
        X=np.array(X)
        y=np.array(y)
        # random first theta
        theta=np.random.rand(X.shape[1])
        bias=np.random.rand(1)
        # set batch
        num_batch=X.shape[0]//batch_size
        X_batch=np.reshape(X,(num_batch,batch_size,X.shape[1]))
        y_batch=np.reshape(y,(num_batch,batch_size))
        # for select best loss
        best_mse=math.inf
        
        for i in range(epoch):
            for j in range(num_batch):
                # adjust theta
                theta,bias=self.adjust_weight(X_batch[j],y_batch[j],theta,bias,learning_rate)
            # calaulate loss
            y_pred=self.predict(X,theta,bias) 
            mse=self.mean_square_error(y,y_pred)
            # select best loss
            if mse < best_mse:
                best_mse=mse
                best_theta=theta
                best_bias=bias
            print("epoch",i)
            print("loss",mse)
            print("---------------")

        return best_theta,best_bias,best_mse
    
X=np.array([[1,1],[2,1],[3,2],[4,2]])
y=np.array([1,2,3,4])
L1=LinearRegression()
theta,bias,_=L1.train(X,y,10,0.1,2)
print(theta,bias)
print(L1.predict(X,theta,bias))
