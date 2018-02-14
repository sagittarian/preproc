import numpy as np

import pytest

from preproc.preprocess import Preprocess
from preproc.errors import PreprocessingError
from preproc import steps


class TestPreprocess(object):
    def test_add_step(self, preproc_instance):
        preproc_instance.add(steps.Log())

    def test_fit_diff_no_data(self, preproc_instance):
        preproc_instance.fit([])

    def test_add_step_fitted(self, preproc_instance):
        preproc_instance.fit([])
        with pytest.raises(PreprocessingError):
            preproc_instance.add(steps.Log())

    def test_run_preprocess(self, preproc_instance):
        data = np.arange(1, 21).reshape((4, 5))
        expected = np.array([
            [ 0.23641571,  0.154272  ,  0.11094599],
            [-0.03551801, -0.01562703, -0.00683704],
            [-0.08900669, -0.0593021 , -0.04320469],
            [-0.11189099, -0.07934285, -0.06090427]
        ])
        assert np.allclose(preproc_instance.fit(data), expected)
        assert np.allclose(preproc_instance.transform(data), expected)
