import numpy as np

def minkowski_distance (X,row,p):
    sum = np.abs(X-row)**p
    return np.sum(sum,axis=1)**(1/p)

def euclidian_distance(X,row):
    sum  = minkowski_distance(X,row,2)
    return sum 

def manhattan_distance(X, row):
    sum  = minkowski_distance(X,row,1)
    return sum

def chebyshev_distance(X, row):
    vetor = (X - row)
    return np.max(vetor)