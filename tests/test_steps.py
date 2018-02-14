import numpy as np

from pytest import mark, raises

from preproc.errors import PreprocessingError
from preproc import steps


class TestStep(object):
    @mark.parametrize('stepclass,args,kwds,data,expected', [
        (steps.Log, (), {},
         [[1, 2, 3], [4, 5, 6]],
         [[0, 0.69314718, 1.09861229],
          [1.38629436, 1.60943791, 1.79175947]]),
        (steps.Diff, (), {}, [[1, 2, 3], [4, 6, 9]], [[1, 1], [2, 3]]),
        (steps.SelectWavelengths, (), dict(from_=1, to=3),
         [[7, 2, 5, 4, 5], [6, 7, 8, 9, 10]],
         [[2, 5, 4], [7, 8, 9]]),
    ])
    def test_step_transform(self, stepclass, args, kwds, data, expected):
        step = stepclass(*args, **kwds)
        assert np.allclose(step.transform(data), expected)

    @mark.parametrize('stepclass,args,kwds,fit_data,data,expected', [
        (steps.SubtractAvg, (), {},
         [[1, 2, 0], [4, 5, 6]], [[5, 5, 5]],
         [[2.5, 1.5, 2]])
    ])
    def test_step_fit_transform(self, stepclass, args, kwds,
                                fit_data, data, expected):
        step = stepclass(*args, **kwds)
        step.fit(fit_data)
        assert np.allclose(step.transform(data), expected)

    def test_fitted(self):
        step = steps.SubtractAvg()
        data = [[1, 2, 0], [4, 5, 6]]
        step.fit(data)
        result = step.transform(data)
        with raises(PreprocessingError):
            step.fit(data)
