import math
import numpy as np
from sklearn import metrics
def mse(y_true, y_pred):
    count = sum(((y_true - y_pred)**2))
    return count/len(y_true)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    count = sum(np.abs((y_true - y_pred)))
    return count/len(y_true)

def accuracy(y_true,y_pred):
    num = np.where(y_true == y_pred)
    return num[0].shape[0]/y_pred.shape[0]

def precision(y_true, y_pred):
    precision = []
    matriz = metrics.confusion_matrix(y, y_pred, labels=np.unique(y_true))
    for i in range(matriz.shape[1]):
        precision.append(matriz[i,i]/(sum(matriz[:,i])))

    return precision
def f1_measure(y, y_pred):
    mat =  metrics.confusion_matrix(y, y_pred, labels=np.unique(y_true))
    return 2*((precision(y, y_pred)*recall(y, y_pred))/(precision(y, y_pred)+recall(y, y_pred)))


def recall(y_true, y_pred):
    precision = []
    matriz = metrics.confusion_matrix(y, y_pred, labels=np.unique(y_true))
    print (matriz)
    for i in range(matriz.shape[1]):
        precision.append(matriz[i,i]/(sum(matriz[i,:])))
        
    return precision


y = ['cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat',  'dog', 'dog', 
'dog', 'dog', 'dog', 'dog', 'rabbit', 'rabbit', 'rabbit', 'rabbit', 'rabbit',
 'rabbit', 'rabbit', 'rabbit', 'rabbit', 'rabbit', 'rabbit', 'rabbit', 'rabbit']
y_pred = ['cat', 'cat', 'cat', 'cat', 'cat', 'dog', 'dog', 'dog',  'cat', 
'cat', 'dog', 'dog', 'dog', 'rabbit', 'dog', 'dog', 'rabbit', 'rabbit', 'rabbit', 'rabbit', 
'rabbit', 'rabbit', 'rabbit', 'rabbit', 'rabbit', 'rabbit', 'rabbit']
print(accuracy(np.array(y),np.array(y_pred)))
print(precision(np.array(y),np.array(y_pred)))
print(recall(np.array(y),np.array(y_pred)))

