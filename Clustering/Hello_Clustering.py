import numpy as np
import math

def Euclidean_distance(o1,o2):
    Ed = np.sqrt(np.sum((o1-o2)**2))
    return Ed

def Cosine_similarity(o1,o2):
    dot = np.dot(o1, o2)
    sso1 = np.sqrt(np.sum(o1**2))
    sso2 = np.sqrt(np.sum(o2**2))
    return dot/(sso1 + sso2)

class hello_clustering:
    def __init__(self,data, k, distance, tolerance):
        self.n=len(data)
        self.dim=len(data[0])
        self.data=data.astype(np.float32)
        self.k=k
        self.cluster=np.zeros(self.n)
        self.old_centroid=np.zeros((k,self.dim))
        self.distance=distance
        self.tolerance=tolerance

    def Random_centroid(self):
        # random k centroid
        index=np.random.choice(self.data.shape[0], self.k, replace=False)
        self.centroid=self.data[index]
        # print(self.centroid)
        # print("-----------")

    def Assign_centroid(self):
        # Assign centroid
        for i in range(self.n):
            dis=math.inf
            for j in range(self.k):
                dis_c = self.distance(self.data[i],self.centroid[j]) 
                if dis_c<dis:
                    dis=dis_c
                    self.cluster[i]=j

    def Update_centroid(self):
        # Update centroid
        self.old_centroid=np.copy(self.centroid)
        for j in range(self.k):
            so=np.zeros(self.dim)
            nc=0
            for i in range(self.n):
                if self.cluster[i] == j:
                    so+=np.array(self.data[i])
                    nc+=1
            self.centroid[j]=np.array(so/nc)
        # print(self.centroid)
        # print("-----------")

    def forward(self):
        self.Random_centroid()
        self.Assign_centroid()
        while not(np.allclose(self.centroid, self.old_centroid, atol=self.tolerance)):
            self.Update_centroid()
            self.Assign_centroid()
        # print(self.centroid)
        # print(self.cluster)
        # print("-----------")
        return self.cluster,self.centroid

class Elbow_method:
    def __init__(self,data,N,distance):
        self.data=data
        self.n=len(data)
        self.M=np.sum(data,axis=0)/len(data)
        self.N=N
        self.distance=distance
    
    def between_cluster_variance(self,centroid,cluster):
        bcv=0
        for j in range(len(centroid)):
            dis=self.distance(centroid[j],self.M)**2
            n=(cluster==j).sum()
            bcv+=dis*n
        return bcv/(self.n-1)

    def all_data_variance(self,data):
        adv=0
        for i in range(self.n):
            dis=self.distance(data[i],self.M)**2
            adv+=dis
        return adv/(self.n-1)

    def forword(self):
        fraction_variance=np.zeros(self.N)
        for i in range(self.N):
            clustering=hello_clustering(self.data,i+1,self.distance,1e-5)  
            cluster,centroid=clustering.forward()
            bcv=self.between_cluster_variance(centroid,cluster)
            adv=self.all_data_variance(self.data)
            # print(bcv)
            # print(adv)
            # print("-----------")
            fraction_variance[i]=bcv/adv
        return fraction_variance

data=np.array([[1,2],[3,3],[2,2],[8,8],[6,6],[7,7],[-3,-3],[-2,-4],[-7,-7]])
elbow=Elbow_method(data,5,Euclidean_distance)
print(elbow.forword())

        