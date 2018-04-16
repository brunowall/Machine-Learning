import dist_metrics as metrics
import transform
class KNNClassifier:
    k = None
    metric = None
    p= None

    def __init__(self,k=5, metric='minkowski', p=2):
       self.k = k
       self.p = p
       self.metric = metric
    def get_neighbors(self,X_train, test_row):
       if(hasattr(metrics,self.metric+"_distance")):
            method = getattr(metrics,self.metric+"_distance")
            distances = method()
            idx_sort = np.argsort(distances)
            return idx_sort[1:self.k+1]

    def predict_classification(self,X, y, test_row):
        idx_sort = self.get_neighbors(X,test_row)
        output_values = y[idx_sort]
        counts = np.unique(output_values, return_counts=True)
        indx_max = np.argmax(counts[1])
        prediction = counts[0][indx_max]
        return prediction



class KNNPredictor:
    k = None
    metric = None
    p= None
    
    def __init__(self,k=5, metric='minkowski', p=2):
       self.k = k
       self.p = p
       self.metric = metric


    def get_neighbors(self,X_train, test_row):
       if(hasattr(metrics,self.metric+"_distance")):
            method = getattr(metrics,self.metric+"_distance")
            distances = method()
            idx_sort = np.argsort(distances)
            return idx_sort[1:self.k+1]

    def predict(self,X, y, test_row):
        idx_sort = self.get_neighbors(X,test_row)
        output_values = y[idx_sort]
        prediction = np.sum(output_values) / output_values.shape[0]
        return prediction