import math
import numpy as np
import stats
class SimpleLinearRegression:
    b0 = None
    b1 = None

    def fit(self,X,y):
       b1 = np.sum((X - np.mean(X)) * (y - np.mean(y))) / np.sum((X - np.mean(X)) ** 2) 
       b0 = np.mean(y) - b1 * np.mean(X)

    def predict(self,X):
        y_pred = []
        for x in X:
            y_pred.append((self.b0 + self.b1*x[0]))
            print(x[0])

        return y_pred