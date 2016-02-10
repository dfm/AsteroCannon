# -*- coding: utf-8 -*-

__all__ = ["AsteroCannon"]

import numpy as np
from scipy.optimize import minimize


class AsteroCannon(object):

    def __init__(self, order=2):
        self.order = order

    def _get_A(self, x):
        order = self.order
        try:
            int(order)
        except TypeError:
            if len(order) != x.shape[1]:
                raise ValueError("dimension mismatch")
        else:
            order = [order for _ in range(x.shape[1])]

        # Voodoo magic for computing all the permutations.
        xs = [np.vander(x[:, i], n) for i, n in enumerate(order)]
        s0 = [slice(None)] + [None for _ in range(x.shape[1])]
        s = list(s0)
        x0 = xs[0]
        for i in range(1, x.shape[1]):
            s[i] = slice(None)
            s1 = s[:i+2]
            s2 = list(s0)[:i+2]
            s2[i+1] = slice(None)
            x0 = x0[s1] * xs[i][s2]
        x0 = x0.reshape((len(x), -1))
        return x0

    def fit(self, X, y):
        self._X_std = np.std(X, axis=0)
        self._X_mean = np.mean(X, axis=0)
        x = (X - self._X_mean[None, :]) / self._X_std[None, :]
        self._X_min = np.min(x, axis=0)
        self._X_max = np.max(x, axis=0)

        A = self._get_A(x)
        ATA = np.dot(A.T, A)
        alpha = np.dot(A.T, y - np.median(y, axis=1)[:, None])
        self.weights_ = np.linalg.solve(ATA, alpha)

    def predict(self, X):
        x = (np.atleast_2d(X) - self._X_mean[None, :]) / self._X_std[None, :]
        A = self._get_A(x)
        return np.dot(A, self.weights_)

    def _nll(self, x, y):
        A = self._get_A(np.atleast_2d(x))
        mu = np.dot(A, self.weights_)[0]
        return 0.5 * np.sum((mu - y)**2)

    def infer_one(self, y, nrestarts=10):
        yy = np.array(y)
        yy -= np.median(y)
        best = (np.inf, None)
        d = self._X_max-self._X_min
        for _ in range(nrestarts):
            pos = self._X_min + d*np.random.rand(len(self._X_min))
            r = minimize(self._nll, pos, args=(yy, ),
                         bounds=list(zip(self._X_min, self._X_max)))
            if r.fun < best[0]:
                best = (r.fun, r.x)
        return best[1] * self._X_std + self._X_mean
