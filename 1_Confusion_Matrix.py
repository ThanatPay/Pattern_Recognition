import numpy as np
    
class Confusion_Matrix:
    def __init__(self,y_true,y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        self.acc=(tp+tn)/(tp+fp+tn+fn)
        self.pre=tp/(tp+fp)
        self.rec=tp/(tp+fn)
        self.f1=(2*self.pre*self.rec)/(self.pre+self.rec)

    def accuracy(self):
        return self.acc

    def precision(self):
        return self.pre

    def recall(self):
        return self.rec
    
    def F1(self):
        return self.f1

