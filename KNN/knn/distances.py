import numpy as np
import numpy.linalg as LA


def euclidean_distance(X, Y):
    X_norm = np.sum(X * X, axis=1)[:, None]
    Y_norm = np.sum(Y * Y, axis=1)[None, :]
    return np.sqrt(X_norm + Y_norm - 2 * np.dot(X, Y.T))


def cosine_distance(X, Y):
    X_norm = np.sum(X * X, axis=1)[:, None]
    Y_norm = np.sum(Y * Y, axis=1)[None, :]

    X_norm[X_norm == 0] = 1e-5
    Y_norm[Y_norm == 0] = 1e-5
    return 1.0 - np.dot(X, Y.T) / np.sqrt((X_norm  * Y_norm))
