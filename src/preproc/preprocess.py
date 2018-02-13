import numpy as np

from preproc.errors import PreprocessingError


class Preprocess(object):
    def __init__(self, steps=()):
        self.steps = list(steps)
        self.fitted = False

    def __call__(self, data):
        return self.transform(data)

    def _check_not_fitted(self):
        if self.fitted:
            raise PreprocessingError('Preprocessing object already fitted')

    def add(self, step):
        self._check_not_fitted()
        self.steps.append(step)

    def fit(self, data):
        data = np.atleast_2d(data)
        self._check_not_fitted()
        for step in self.steps:
            data = step.fit(data)
        self.fitted = True

    def transform(self, data):
        self.fitted = True
        data = np.atleast_2d(data)
        for step in self.steps:
            data = step.transform(data)
        return data
