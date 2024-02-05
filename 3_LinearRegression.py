import numpy as np
import math

class LinearRegression:
    def adjust_theta(self,X,y,theta,learning_rate):
        y_pred=self.predict(X,theta)
        gd=np.matmul(y-y_pred,X)
        return theta+(learning_rate*gd)

    def mean_square_error(self,y,y_pred):
        return np.mean((y-y_pred)**2)

    def predict(self,X,theta):
        return np.matmul(X,theta)
        
    def train(self, X, y, epoch, learning_rate, batch):
        # Convert to NumPy arrays if not already
        X=np.array(X)
        y=np.array(y)
        # random first theta
        theta=np.random.rand(X.shape[1])
        # set batch
        X_batch=np.reshape(X,(batch,X.shape[0]//batch,X.shape[1]))
        y_batch=np.reshape(y,(batch,y.shape[0]//batch))
        # for select best loss
        best_mse=math.inf
        
        for i in range(epoch):

            for j in range(batch):
                # adjust theta
                theta=self.adjust_theta(X_batch[j],y_batch[j],theta,learning_rate)
            # calaulate loss
            y_pred=self.predict(X,theta) 
            mse=self.mean_square_error(y,y_pred)
            # select best loss
            if mse < best_mse:
                best_mse=mse
                best_theta=theta
            print("epoch",i)
            print("loss",mse)
            print("---------------")

        return best_theta,best_mse
    
# X=np.array([[1,1],[2,1],[3,2],[4,2]])
# y=np.array([1,2,3,4])
# L1=LinearRegression()
# theta,_=L1.train(X,y,10,0.1,2)
# print(L1.predict(X,theta))
