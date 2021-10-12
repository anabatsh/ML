import numpy as np
import multiprocessing
from joblib import parallel_backend, Parallel, delayed
from sklearn.neighbors import NearestNeighbors  # only for specific strategies

from knn.distances import euclidean_distance, cosine_distance


class KNNClassifier:
    def __init__(self, k: int, strategy: str = 'my_own', metric: str = 'eucledean',
                 weights: bool = False, test_block_size: int = 0, n_jobs: int = 1):
        self.k = k

        self.metric = metric
        if metric == "euclidean":
            self._metric_func = euclidean_distance
        elif metric == "cosine":
            self._metric_func = cosine_distance
        else:
            raise TypeError("Metric <{}> is not supported!".format(metric))

        # only for parallel computation
        num_cores = multiprocessing.cpu_count()
        self.n_jobs = n_jobs if 0 < n_jobs < num_cores else num_cores

        self.strategy = strategy
        if strategy in ['brute', 'kd_tree', 'ball_tree']:
            self.NearestNeighbours = NearestNeighbors(
                n_neighbors=k,
                algorithm=strategy,
                metric=metric,
                n_jobs=self.n_jobs
            )

        elif strategy != 'my_own':
            raise TypeError("Strategy <{}> is not supported!".format(strategy))

        self.weights = weights
        self.test_block_size = test_block_size

        self.X_train = None
        self.y_train = None
        self.labels = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.strategy != 'my_own':
            self.NearestNeighbours.fit(X)
        else:
            self.X_train = X

        self.y_train = y
        self.labels = np.unique(y)

    def _find_kneighbors(self, X: np.ndarray, return_distance: bool):
        chunks_borders = np.linspace(0, self.X_train.shape[0], self.n_jobs + 1).astype(int)
        chunks_borders = zip(chunks_borders[:-1], chunks_borders[1:])

        def chunk_find_kneighbors(left, right):
            chunk_distances = self._metric_func(X, self.X_train[left:right])
            chunk_indices = np.argpartition(-chunk_distances, range(-self.k, 0))[:, -self.k:]
            return np.take_along_axis(chunk_distances, chunk_indices, axis=1), chunk_indices + left

        with parallel_backend('threading', n_jobs=self.n_jobs):
            chunks_neighbors = Parallel()(
                delayed(chunk_find_kneighbors)(left, right)
                for (left, right) in chunks_borders
            )

        distances, indices = zip(*chunks_neighbors)
        distances, indices = np.hstack(distances), np.hstack(indices)

        inner_indices = np.argpartition(-distances, range(-self.k, 0))[:, -self.k:]

        if return_distance:
            return (
                np.flip(np.take_along_axis(distances, inner_indices, axis=1), axis=1),
                np.flip(np.take_along_axis(indices, inner_indices, axis=1), axis=1)
            )
        else:
            return np.flip(np.take_along_axis(indices, inner_indices, axis=1), axis=1)

    def find_kneighbors(self, X: np.ndarray, return_distance: bool = True):
        if self.strategy != 'my_own':
            return self.NearestNeighbours.kneighbors(
                X,
                return_distance=return_distance
            )
        else:
            return self._find_kneighbors(X, return_distance)

    def estimate(self, indices: np.ndarray, distances: np.ndarray = None):
        # partial predictions without re-evaluating find_kneighbors()
        weights = 1.0 / (1e-5 + distances) if self.weights else np.ones(indices.shape)
        predictions = np.empty((indices.shape[0], self.labels.shape[0]))
        for i, label in enumerate(self.labels):
            predictions[:, i] = np.mean(weights * (self.y_train[indices] == label), axis=1)

        return self.labels[np.argmax(predictions, axis=1)]

    def predict(self, X: np.ndarray):
        distances, indices = self.find_kneighbors(X, True)
        return self.estimate(indices, distances)
