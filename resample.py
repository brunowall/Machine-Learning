import math
import numpy as np

def split_train_test(n_elem, perc_train, seed):
    total = [ i for i in range(n_elem)]
    elem_train = math.floor(perc_train*n_elem) #numero de elementos da funcao de treino
    np.random.seed(seed)
    np.random.shuffle(total)
    train = total[0:elem_train]
    test = total[elem_train:]
    return train,test

