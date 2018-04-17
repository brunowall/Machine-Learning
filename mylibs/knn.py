import dist_metrics as metrics
import numpy as np
class KNNClassifier:
    k = None
    metric = None
    p = None
    train_features = None
    train_labels = None

    def __init__(self,k=5, metric='minkowski', p=2):
       self.k = k
       self.p = p
       self.metric = metric
    def get_neighbors(self, test_row):
       if(hasattr(metrics,self.metric+"_distance")):
            method = getattr(metrics,self.metric+"_distance")
            if(self.metric=='minkowski'):
                distances = method(self.train_features,test_row,self.p)
            else:     
                distances = method(self.train_features,test_row)
            idx_sort = np.argsort(distances)
            return idx_sort[1:self.k+1]

    def fit(self,X,y):
        self.train_features = X
        self.train_labels = y

    def predict(self, test_rows):
        predictions = np.empty((test_rows.shape[0],0))
        for i in range(test_rows.shape[0]):
            idx_sort = self.get_neighbors(test_rows[i,:])
            output_values = self.train_labels[idx_sort]
            counts = np.unique(output_values, return_counts=True)
            indx_max = np.argmax(counts[1])
            prediction = counts[0][indx_max]
            predictions.put(i,prediction)
        return predictions

class KNNPredictor:
    k = None
    metric = None
    p= None
    train_features = None
    train_labels = None

    def __init__(self,k=5, metric='minkowski', p = 2):
       self.k = k
       self.p = p
       self.metric = metric
    def get_neighbors(self, test_row):
       if(hasattr(metrics,self.metric+"_distance")):
            method = getattr(metrics,self.metric+"_distance")
            if(self.metric=='minkowski'):
                distances = method(self.train_features,test_row,self.p)
            else:     
                distances = method(self.train_features,test_row,self.p)
            idx_sort = np.argsort(distances)
            return idx_sort[1:self.k+1]
    def fit(self,X,y):
        self.train_features = X
        self.train_labels = y

    def predict(self, test_rows):
        predictions = np.empty((1,0))
        for i in range(test_rows.shape[0]):
            idx_sort = self.get_neighbors(test_rows[i,:])
            output_values = self.train_labels[idx_sort]
            prediction = np.sum(output_values) / output_values.shape[0]
            predictions.append(prediction)
        return predictions