import numpy as np
from math import ceil
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar
from joblib import parallel_backend, Parallel, delayed
import multiprocessing


class RandomForestMSE:
    def __init__(self, n_estimators, max_depth=None, feature_subsample_size=None, n_jobs=1,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size

        num_cores = multiprocessing.cpu_count()
        self.n_jobs = n_jobs if 0 < n_jobs < num_cores else num_cores

        self.trees_parameters = trees_parameters

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """

        if self.feature_subsample_size is None:
            self.feature_subsample_size = ceil(X.shape[1] / 3) # рекомендация для задачи регрессии

        def Decision_Tree(i):
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         max_features=self.feature_subsample_size,
                                         **self.trees_parameters)
            indices = np.random.randint(X.shape[0], size=X.shape[0])
            tree.fit(X[indices], y[indices])
            return tree

        self.tree_list = Parallel(n_jobs=self.n_jobs)(delayed(Decision_Tree)(i) for i in range(self.n_estimators))


    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        #result = np.zeros(X.shape[0])
        #for i in range(self.n_estimators):
        #    result += self.tree_list[i].predict(X)
        #return result / self.n_estimators
        def predict(X, tree):
            return tree.predict(X)

        pred = Parallel(n_jobs=self.n_jobs)(delayed(predict)(X, tree) for tree in self.tree_list)
        return np.mean(pred, axis=0)



class GradientBoostingMSE:
    def __init__(self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use learning_rate * gamma instead of gamma

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters

    def _MSE(self, y, z):
        return ((y - z) ** 2).mean()

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
                X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """
        self.tree_list = []
        self.coef = []
        answers = np.zeros(X.shape[0])

        if self.feature_subsample_size is None:
            self.feature_subsample_size = ceil(X.shape[1] / 3) # рекомендация для задачи регрессии

        for i in range(self.n_estimators):
            mdl = DecisionTreeRegressor(max_depth=self.max_depth, max_features=self.feature_subsample_size, **self.trees_parameters)
            mdl.fit(X, 2 * (answers - y) / X.shape[0])
            current_predict = mdl.predict(X)
            self.coef.append(minimize_scalar(lambda x: self._MSE(y, answers + x * current_predict)).x)
            self.tree_list.append(mdl)
            answers += self.lr * self.coef[i] * current_predict

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        result = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            result += self.coef[i] * self.lr * self.tree_list[i].predict(X)
        return result