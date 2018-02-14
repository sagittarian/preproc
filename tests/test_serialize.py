import json

import numpy as np
import pytest

from preproc import steps
from preproc.preprocess import Preprocess
from preproc.serialize import version


class TestSerialize(object):
    @pytest.mark.parametrize('cls,args', [
        (steps.Log, ()),
        (steps.Diff, ()),
        (steps.SelectWavelengths, (1, 1)),
    ])
    def test_roundtrip(self, cls, args):
        instance = cls(*args)
        data = instance.serialize()
        json.dumps(data)
        deserialized = version.deserialize(data)
        assert isinstance(deserialized, cls)

    @pytest.mark.parametrize('cls,data', [
        (steps.SelectWavelengths, dict(
            attributes={'from_': 1, 'to': 1},
            version=1,
            name='SelectWavelengths',
        )),
        (steps.SubtractAvg, dict(
            attributes={'avg': [1, 1, 1], 'n': 5, 'fitted': True},
            version=1,
            name='SubtractAvg',
        )),
    ])
    def test_deserialize(self, cls, data):
        instance = version.deserialize(data)
        assert isinstance(instance, cls)

    def test_subtractavg_roundtrip(self):
         fit_data = [[1, 2, 0], [4, 5, 6]]
         expected_avg = [[2.5, 3.5, 3]]

         instance = steps.SubtractAvg()
         instance.fit(fit_data)
         instance.transform([[0, 0, 0]])

         serialized = instance.serialize()
         json.dumps(serialized)
         assert np.allclose(serialized['attributes']['avg'], expected_avg)

         deserialized = version.deserialize(serialized)
         assert isinstance(deserialized, steps.SubtractAvg)
         assert np.allclose(deserialized.avg, expected_avg)

    def test_serialize_pipeline(self, preproc_instance):
        serialized = preproc_instance.serialize()
        json.dumps(serialized)

        for (step, stepdata) in zip(preproc_instance.steps,
                                    serialized['attributes']['steps']):
            assert step.name == stepdata['name']

        deserialized = version.deserialize(serialized)
        for (origstep, newstep) in zip(preproc_instance.steps,
                                       deserialized.steps):
            assert isinstance(newstep, type(origstep))

