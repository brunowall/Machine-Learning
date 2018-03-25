import math
import numpy as np
def mean(x):
    soma = sum(x)
    return soma/len(x)

def stdev (x):
    soma = sum((x - mean(x))**2)
    var = soma/len(x) 
    return math.sqrt(var)

def var(y):
    return stdev(y)**2
