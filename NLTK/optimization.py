import numpy as np
from numpy import linalg as LA

import scipy
from scipy.special import expit
from sklearn.metrics import accuracy_score

from time import time

from oracles import BinaryLogistic


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss_function, step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций

        **kwargs - аргументы, необходимые для инициализации
        """
        if loss_function == 'binary_logistic':
            self.oracle = BinaryLogistic(**kwargs)

        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter

    def fit(self, X, y, w_0=None, trace=False):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        trace - переменная типа bool

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)

        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        if w_0 is None:
            self.w = np.zeros(X.shape[1])
        else:
            self.w = w_0

        func_cur = self.oracle.func(X, y, self.w)
        history = {'time': [0], 'func': [func_cur], 'accuracy': [accuracy_score(y, self.predict(X))]}
        start = time()

        for i in range(1, self.max_iter + 1):
            grad = self.oracle.grad(X, y, self.w)
            w_new = self.w - (self.step_alpha / (i ** self.step_beta)) * grad
            func_new = self.oracle.func(X, y, w_new)

            if trace:
                history['time'].append(time() - start)
                history['func'].append(func_new)
                history['accuracy'].append(accuracy_score(y, self.predict(X)))
                start = time()

            if abs(func_cur - func_new) < self.tolerance:
                self.w = w_new
                break

            func_cur = func_new
            self.w = w_new
        if trace:
            return history

    def predict(self, X):
        """
        Получение меток ответов на выборке X

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: одномерный numpy array с предсказаниями
        """
        y_pred = (self.predict_proba(X) > 0.5).astype('int')
        y_pred[y_pred==0] = -1
        return y_pred

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """
        return expit(X.dot(self.w))

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: float
        """
        return self.oracle.func(X, y, self.w)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: numpy array, размерность зависит от задачи
        """
        return self.oracle.grad(X, y, self.w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.w


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss_function, batch_size, step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        batch_size - размер подвыборки, по которой считается градиент

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход


        max_iter - максимальное число итераций (эпох)

        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.

        **kwargs - аргументы, необходимые для инициализации
        """
        if loss_function == 'binary_logistic':
            self.oracle = BinaryLogistic(**kwargs)

        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed

    def fit(self, X, y, w_0=None, trace=False, log_freq=1):
        """
        Обучение метода по выборке X с ответами y
        ВАЖНО! Вектор y должен состоять из 1 и -1, а не 1 и 0.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}

        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.

        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        np.random.seed(self.random_seed)

        if w_0 is None:
            self.w = np.zeros(X.shape[1])
        else:
            self.w = w_0

        if isinstance(X, scipy.sparse.coo.coo_matrix):
            X = X.tocsr()

        func_cur = self.oracle.func(X, y, self.w)
        history = {'epoch_num': [0], 'time': [0], 'func': [func_cur], 'weights_diff': [0],
                   'accuracy': [accuracy_score(y, self.predict(X))]}
        start = time()
        i_d = 1
        for i in range(1, self.max_iter + 1):
            new_index = np.random.permutation(X.shape[0])
            X_new = X[new_index]
            y_new = y[new_index]

            self.w_save = np.copy(self.w)
            for batch in range(X.shape[0] // self.batch_size + 1):
                a = batch * self.batch_size
                b = (batch + 1) * self.batch_size
                grad = self.oracle.grad(X_new[a : b], y_new[a : b], self.w)
                self.w = self.w - self.step_alpha / (i_d ** self.step_beta) * grad
                i_d += 1

            func_new = self.oracle.func(X, y, self.w)

            if trace:
                history['epoch_num'].append(i)
                history['time'].append(time() - start)
                history['func'].append(func_new)
                history['accuracy'].append(accuracy_score(y, self.predict(X)))
                history['weights_diff'].append(LA.norm(self.w - self.w_save))
                start = time()

            if abs(func_cur - func_new) < self.tolerance:
                break

            func_cur = func_new

        if trace:
            return history

    def predict(self, X):
        """
        Получение меток ответов на выборке X

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: одномерный numpy array с предсказаниями
        """
        y_pred = (self.predict_proba(X) > 0.5).astype('int')
        y_pred[y_pred==0] = -1
        return y_pred

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """
        return expit(X.dot(self.w))

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.w
