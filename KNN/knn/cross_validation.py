import numpy as np
from collections import defaultdict

from knn.nearest_neighbors import KNNClassifier


def kfold(n, n_folds):
    index = np.arange(n)
    folds = np.array_split(index, n_folds)

    result = []
    for fold in folds:
        mask = np.ones(n).astype(bool)
        mask[fold] = False
        result.append((index[mask], index[~mask]))

    return result


def knn_cross_val_score(X: np.ndarray, y: np.ndarray, k_list, score: str = 'accuracy', cv=None, **kwargs):
    if score == 'accuracy':
        score_func = lambda y_pred, y_true: np.mean(np.abs(y_pred == y_true))
    elif score == 'mae':
        score_func = lambda y_pred, y_true: np.mean(np.abs(y_pred != y_true))
    else:
        raise TypeError("Score <{}> is not supported!\nUse <accuracy>.".format(score))

    # KFold splits
    if not cv:
        cv = kfold(X.shape[0], n_folds=3)

    # fix k for validation
    kwargs['k'] = k_list[-1]

    # process score_func for each fold from cv and each k from k_list
    scoring = defaultdict(list)
    for train_ind, val_ind in cv:
        model = KNNClassifier(**kwargs)
        model.fit(X[train_ind], y[train_ind])
        distances, indices = model.find_kneighbors(X[val_ind], True)
        for k in k_list:
            predictions = model.estimate(indices[:, :k], distances[:, :k])
            scoring[k].append(score_func(predictions, y[val_ind]))

        # try to free the memory
        del model

    return scoring
