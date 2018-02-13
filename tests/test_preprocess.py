import pytest

from preproc.preprocess import Preprocess
from preproc.errors import PreprocessingError
from preproc import steps


class TestPreprocess(object):
    def get_preprocessor(self):
        return Preprocess([
            steps.Log(), steps.Diff(),
            steps.SelectWavelengths(from_=1, to=3),
            steps.SubtractAvg(),
        ])

    def test_create_preprocessor(self):
        preproc = self.get_preprocessor()

    def test_add_step(self):
        preproc = self.get_preprocessor()
        preproc.add(steps.Log())

    def test_fit_diff_no_data(self):
        preproc = Preprocess([steps.Diff()])
        preproc.fit([])

    def test_add_step_fitted(self):
        preproc = self.get_preprocessor()
        preproc.fit([])
        with pytest.raises(PreprocessingError):
            preproc.add(steps.Log())
