import pytest

from preproc.preprocess import Preprocess
from preproc import steps


@pytest.fixture
def preproc_instance():
    return Preprocess([
        steps.Log(), steps.Diff(),
        steps.SelectWavelengths(from_=1, to=3),
        steps.SubtractAvg(),
    ])
