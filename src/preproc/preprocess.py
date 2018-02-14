import numpy as np

from preproc.errors import PreprocessingError


class Preprocess(object):
    """Class to encapsulate the preprocessing pipeline with variable steps.

    A Preprocess object encapsulates the preprocessing pipeline.  Steps
    should inherit from preproc.steps.Step, and can either be passed in
    to the __init__ method, or added later through the add method.  At
    each step, the output from the previous step is passed as input to
    the next step.

    A Preprocess object works in two modes, fit and transform.  fit mode
    is used to set parameters in steps which learn parameters (for
    example, the average value in the training dataset), and transform
    mode is used to run new data through the preprocessing pipeline
    after the parameters have been learned.

    """

    def __init__(self, steps=()):
        """Create a new preprocessing pipeline.

        Initial steps can be passed as an iterable.

        """
        self.steps = list(steps)
        self.fitted = False

    def __call__(self, data):
        return self.transform(data)

    def _check_not_fitted(self):
        if self.fitted:
            raise PreprocessingError('Preprocessing object already fitted')

    def add(self, step):
        """Add a step to the end of this preprocessing pipeline.

        Raises: preproc.errors.PreprocessingError if the pipeline has
        already been fitted.

        """
        self._check_not_fitted()
        self.steps.append(step)

    def fit(self, data):
        """Run with training data, allowing steps to learn their parameters.

        After training, the transform method is called with the data and
        its result is returned.

        Raises: preproc.errors.PreprocessingError if the pipeline is
        already fitted.

        Returns: numpy.ndarray of the processed data

        """
        data = np.atleast_2d(data)
        self._check_not_fitted()
        for step in self.steps:
            step.fit(data)
            data = step.transform(data)
        self.fitted = True
        return data

    def transform(self, data):
        """Run the pipeline on new data.

        The data will be transformed according to the pipeline's steps
        and the parameters learned during the fitting stage.

        Returns: numpy.ndarray of the processed data

        """
        self.fitted = True
        data = np.atleast_2d(data)
        for step in self.steps:
            data = step.transform(data)
        return data
