from ..base_stage import PipelineStage, HDFFile
import numpy as np

class Stage(PipelineStage):
    name = "PipelineTestingStage"
    inputs = [("tag1", HDFFile), ("tag2", HDFFile)]
    outputs = []


def test_combine():
    s = Stage({
        'tag1':'data/testing/test1.hdf',
        'tag2':'data/testing/test2.hdf',
        'config':'examples/laptop/config.yml'
    })

    it = s.combined_iterators(10,
            'tag1', 'A', ['a', 'b'],
            'tag2', 'B', ['d'],
        )

    res = next(it)
    assert len(res) == 3
    s, e, data = res
    assert s == 0
    assert e == 10
    assert 'a' in data
    assert 'b' in data
    assert 'd' in data
    assert (data['a'] == np.arange(10)).all()
    assert (data['b'] == np.arange(100, 110)).all()
    assert (data['d'] == np.arange(200, 210)).all()
