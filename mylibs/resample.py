import math
import numpy as np

def split_train_test(n_elem, perc_train, seed):
    total = [ i for i in range(n_elem)]
    elem_train = math.floor(perc_train*n_elem) # numero de elementos da funcao de treino
    np.random.seed(seed)
    np.random.shuffle(total)
    train = total[0:elem_train]
    test = total[elem_train:]
    return train,test

def __folds(n_elem, n_splits=2, shuffle=True, seed=0):
    total = np.arange(n_elem)
    if shuffle:
        np.random.shuffle(total)
    if seed:
        np.random.seed(seed)
    
    fold_size = n_elem // n_splits
    split_fold = np.arange(0, n_elem, fold_size)[1:n_splits]
    return np.split(total, split_fold)
    
def split_k_fold(n_elem, n_splits=2, shuffle=True, seed=0):
    train = []
    test  = []
    total = __folds(n_elem, n_splits, shuffle, seed)
    for i in range(n_splits):
        x = np.delete(total, [i])
        if type(x[0]) == np.ndarray:
            x = np.concatenate(x)
        train.append(x)
        test.append(total[i])
    return train, test