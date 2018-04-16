import dist_metrics as metrics
import transform
import numpy as np
class KNNClassifier:
    k = None
    metric = None
    p= None
    train_features = None
    train_labels = None

    def __init__(self,k=5, metric='minkowski', p=2):
       self.k = k
       self.p = p
       self.metric = metric
    def get_neighbors(self, test_row):
       if(hasattr(metrics,self.metric+"_distance")):
            method = getattr(metrics,self.metric+"_distance")
            distances = method(self.train_features,test_row,self.p)
            idx_sort = np.argsort(distances)
            return idx_sort[1:self.k+1]

    def fit(self,X,y):
        self.train_features = X
        self.train_labels = y

    def predict(self, test_row):
        idx_sort = self.get_neighbors(test_row)
        output_values = self.train_labels[idx_sort]
        counts = np.unique(output_values, return_counts=True)
        indx_max = np.argmax(counts[1])
        prediction = counts[0][indx_max]
        return prediction



class KNNPredictor:
    k = None
    metric = None
    p= None
    train_features = None
    train_labels = None

    def __init__(self,k=5, metric='minkowski', p=2):
       self.k = k
       self.p = p
       self.metric = metric

    def get_neighbors(self,test_row):
       if(hasattr(metrics,self.metric+"_distance")):
            method = getattr(metrics,self.metric+"_distance")
            distances =  method(self.train_features,test_row,self.p)
            idx_sort = np.argsort(distances)
            return idx_sort[1:self.k+1]

    def fit(self,X,y):
        self.train_features = X
        self.train_labels = y

    def predict(self, test_row):
        idx_sort = self.get_neighbors(test_row)
        output_values = self.train_labels[idx_sort]
        prediction = np.sum(output_values) / output_values.shape[0]
        return prediction
        