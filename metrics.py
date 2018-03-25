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