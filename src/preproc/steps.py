import numpy as np

from preproc.errors import PreprocessingError


class Step(object):
    def __init__(self):
        pass

    def __call__(self, data):
        return self.transform(data)

    def fit(self, data):
        return self.transform(data)

    def transform(self, data):
        raise NotImplementedError


class Log(Step):
    def transform(self, data):
        return np.log(data)


class Diff(Step):
    def transform(self, data):
        return np.diff(data, axis=1)


class SelectWavelengths(Step):
    def __init__(self, from_, to):
        self.from_ = from_
        self.to = to

    def transform(self, data):
        return np.array(data)[:, self.from_:self.to+1]


class SubtractAvg(Step):
    def __init__(self):
        self.avg = None
        self.n = 0
        self.fitted = False

    def _check_not_fitted(self):
        if self.fitted:
            raise PreprocessingError('Step already fitted')


    def _calc_avg(self):
        self._check_not_fitted()
        if self.n == 0:
            raise PreprocessingError('No data points to average')
        self.avg /= self.n
        self.fitted = True

    def transform(self, data):
        if not self.fitted:
            self._calc_avg()
        return data - self.avg

    def fit(self, data):
        self._check_not_fitted()
        data = np.array(data)
        if self.avg is None:
            self.avg = np.zeros(data.shape[1])
        self.avg += np.sum(data, axis=0)
        self.n += data.shape[0]
        return data  # XXX
