from ..base_stage import PipelineStage, HDFFile
import numpy as np

class Stage(PipelineStage):
    name = "PipelineTestingStage"
    inputs = [("tag1", HDFFile), ("tag2", HDFFile)]
    outputs = []


class Stage2(PipelineStage):
    name = "PipelineTestingStage2"
    inputs = [("tag1", HDFFile)]
    outputs = []

def test_combine():
    s = Stage({
        'tag1':'data/testing/test1.hdf',
        'tag2':'data/testing/test2.hdf',
        'config':'examples/config/laptop_config.yml'
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


def test_combine_longest():
    s = Stage2({
        'tag1':'data/testing/mixed_lengths.hdf5',
        'config':'examples/config/laptop_config.yml'
    })

    it = s.combined_iterators(50,
            'tag1', 'g1', ['c1', 'c2'],
            'tag1', 'g2', ['c3'],
            longest=True
        )
    data = list(it)
    assert len(data) == 2
    # first
    assert len(data[0]) == 2
    se, d = data[0]
    assert len(se) == 2
    assert se[0] == (0, 50)
    assert se[1] == (0, 50)
    assert np.all(d['c1'] == np.arange(50))
    assert np.all(d['c2'] == np.arange(100, 150))
    assert np.all(d['c3'] == np.arange(200, 250))
    for col in d.values():
        assert len(col) == 50

    # second
    assert len(data[1]) == 2
    se, d = data[1]
    assert len(se) == 2
    assert se[0] == (50, 100)
    assert se[1] == (None, None)
    assert np.all(d['c1'] == np.arange(50, 100))
    assert np.all(d['c2'] == np.arange(150, 200))
    assert d['c3'] is None

