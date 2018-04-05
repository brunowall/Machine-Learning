import math
import numpy as np

def mse(y_true, y_pred):
    count = sum(((y_true - y_pred)**2))
    return count/len(y_true)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    count = sum(np.abs((y_true - y_pred)))
    return count/len(y_true)

def accuracy(y_true,y_pred):
    cont = 0
    for i in range (y_pred.shape[0]):
        y_true[i] == y_pred[i] ? cont+=1:continue
    return (cont/y_pred.shape[0])

