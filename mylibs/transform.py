import numpy as np

def standardize(X):
    X_std = np.copy(X)
    n_cols = X.shape[1]
    for i in range(n_cols):
        X_std[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
    return X_std


def normalize(X):
    X_norm = np.copy(X)
    n_cols = X.shape[1]
    for i in range(n_cols):
        minn = np.min(X[:, i])
        X_norm[:, i] = (X[:, i] - minn) /  (np.max(X[:, i]) - minn)
    return X_norm

