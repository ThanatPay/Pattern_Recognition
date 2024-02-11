import numpy as np
    
class Evaluation:
    def __init__(self,y_true,y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.tp = np.sum((y_true == 1) & (y_pred == 1))
        self.fn = np.sum((y_true == 1) & (y_pred == 0))
        self.fp = np.sum((y_true == 0) & (y_pred == 1))
        self.tn = np.sum((y_true == 0) & (y_pred == 0))
        self.acc=(self.tp+self.tn)/(self.tp+self.fp+self.tn+self.fn)
        self.pre=self.tp/(self.tp+self.fp) if (self.tp+self.fp)>0 else 0
        self.rec=self.tp/(self.tp+self.fn) if (self.tp+self.fn)>0 else 0
        self.f1=(2*self.pre*self.rec)/(self.pre+self.rec) if self.pre+self.rec>0 else 0

    def confusion_matrix(self):
        return self.tp,self.fn,self.fp,self.tp

    def accuracy(self):
        return self.acc

    def precision(self):
        return self.pre

    def recall(self):
        return self.rec
    
    def F1(self):
        return self.f1

