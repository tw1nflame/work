import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


class LogitRecalibrator(BaseEstimator, ClassifierMixin):
    """
    Калибровка вероятностей через logistic recalibration:
        logit(p_cal) = a + b * logit(p_base)

    Где:
    - p_base = predict_proba базовой модели
    - a = intercept correction
    - b = slope correction

    Подходит, когда модель хорошо ранжирует, но занижает/завышает вероятности.
    """

    def __init__(self, base_model, eps=1e-15, C=1e6, max_iter=1000):
        self.base_model = base_model
        self.eps = eps
        self.C = C
        self.max_iter = max_iter
        self.lr_ = None

    def _clip_proba(self, p):
        return np.clip(p, self.eps, 1 - self.eps)

    def _logit(self, p):
        p = self._clip_proba(p)
        return np.log(p / (1 - p))

    def fit(self, X, y):
        # вероятности базовой модели для класса 1
        p_base = self.base_model.predict_proba(X)[:, 1]

        # переводим в logit-space
        z = self._logit(p_base).reshape(-1, 1)

        # почти без регуляризации, чтобы не мешать калибровке
        self.lr_ = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            max_iter=self.max_iter
        )
        self.lr_.fit(z, y)
        return self

    def predict_proba(self, X):
        if self.lr_ is None:
            raise ValueError("Recalibrator is not fitted yet.")

        p_base = self.base_model.predict_proba(X)[:, 1]
        z = self._logit(p_base).reshape(-1, 1)

        p_cal = self.lr_.predict_proba(z)[:, 1]
        p_cal = self._clip_proba(p_cal)

        return np.column_stack([1 - p_cal, p_cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def intercept_(self):
        if self.lr_ is None:
            raise ValueError("Recalibrator is not fitted yet.")
        return self.lr_.intercept_[0]

    @property
    def slope_(self):
        if self.lr_ is None:
            raise ValueError("Recalibrator is not fitted yet.")
        return self.lr_.coef_[0, 0]
