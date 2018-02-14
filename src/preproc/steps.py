import numpy as np

from preproc.errors import PreprocessingError


class Step(object):
    """Base class for defining preprocessing steps.

    A preprocessing step needs to define the method transform, which
    will apply the step to the given data and return the preprocessed
    data.  If the preprocessing step needs to learn parameters from
    training data, it should implement the fit method to learn its
    parameters from training data.

    """
    def __init__(self):
        pass

    def __call__(self, data):
        return self.transform(data)

    def fit(self, data):
        """Learn any parameters that need to be learned from the given data.

        Returns: the unchanged data

        """
        return data

    def transform(self, data):
        """Preprocess the given data according to this step."""
        raise NotImplementedError


class Log(Step):
    """Step to take the natural logarithm of the data points."""

    def transform(self, data):
        return np.log(data)


class Diff(Step):
    """Step to take the difference between successive columns.

    The resulting preprocessed data will have one fewer column than the
    input.

    """

    def transform(self, data):
        return np.diff(data, axis=1)


class SelectWavelengths(Step):
    """Step to select specific columns in the input data.

    The columns selected are controlled by the from_ and to parameters
    to the __init__ method.  All data points between from_ and to
    including the endpoints will be returned by the preprocessing step.
    Indices are 0-based.

    The resulting data will have a different number of columns,
    depending on the arguments passed to the step.

    """

    def __init__(self, from_, to):
        self.from_ = from_
        self.to = to

    def transform(self, data):
        return np.array(data)[:, self.from_ : self.to+1]


class SubtractAvg(Step):
    """Step to calculate average of each column and subtract it from the
    data to be preprocessed.

    In the fit stage the average of each column in the training data is
    calculated, and in the transform stage the learned average is
    subtracted from each column.

    """

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
        return data
