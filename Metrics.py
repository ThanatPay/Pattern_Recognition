class Metrics:
    def __init__(self,tp,fn,fp,tn):
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
    
dog=Metrics(30,20,10,40)
print("Dog")
print("accuracy:",dog.accuracy())
print("precision:",dog.precision())
print("recall:",dog.recall())
print("F1 score:",dog.F1())

cat=Metrics(20,30,40,10)
print("Cat")
print("accuracy:",cat.accuracy())
print("precision:",cat.precision())
print("recall:",cat.recall())
print("F1 score:",cat.F1())

dog_lopsided=Metrics(30*20,20*20,10*80,40*80)
print("Dog_lopsided")
print("accuracy:",dog_lopsided.accuracy())
print("precision:",dog_lopsided.precision())
print("recall:",dog_lopsided.recall())
print("F1 score:",dog_lopsided.F1())

