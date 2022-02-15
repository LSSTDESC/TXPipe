from ..utils.hdf_tools import BatchWriter
import numpy as np


class MockGroup:
    def __init__(self):
        self.log = []
        self.col = None
    def __getitem__(self, name):
        self.col = name
        return self
    def __setitem__(self, r, d):
        self.log.append(f"Wrote {self.col} from {r.start} to {r.stop}")

def test_basic():
    g = MockGroup()
    cols = {'x': np.float64, 'y': np.int64}
    b = BatchWriter(g, cols, 0, max_size=10)

    x = np.arange(5, dtype=np.float64)
    y = np.arange(5, dtype=np.int64)
    b.write(x=x, y=y)
    b.write(x=x, y=y)
    b.write(x=x, y=y)
    b.write(x=x, y=y)
    b.write(x=x, y=y)

    b.finish()
    print(g.log)

    assert "Wrote x from 0 to 10" in g.log
    assert "Wrote y from 0 to 10" in g.log
    assert "Wrote x from 10 to 20" in g.log
    assert "Wrote y from 10 to 20" in g.log
    assert "Wrote y from 20 to 25" in g.log
    assert "Wrote y from 20 to 25" in g.log

def test_long():
    g = MockGroup()
    cols = {'x': np.float64, 'y': np.int64}
    b = BatchWriter(g, cols, 0, max_size=10)

    x = np.arange(35, dtype=np.float64)
    y = np.arange(35, dtype=np.int64)
    b.write(x=x, y=y)
    b.finish()
    print(g.log)

    assert "Wrote x from 0 to 10" in g.log
    assert "Wrote y from 0 to 10" in g.log
    assert "Wrote x from 10 to 20" in g.log
    assert "Wrote y from 10 to 20" in g.log
    assert "Wrote y from 20 to 30" in g.log
    assert "Wrote y from 20 to 30" in g.log
    assert "Wrote y from 30 to 35" in g.log
    assert "Wrote y from 30 to 35" in g.log

def test_offset():
    g = MockGroup()
    cols = {'x': np.float64, 'y': np.int64}
    b = BatchWriter(g, cols, 2, max_size=10)

    x = np.arange(35, dtype=np.float64)
    y = np.arange(35, dtype=np.int64)
    b.write(x=x, y=y)
    b.finish()
    print(g.log)

    assert "Wrote x from 2 to 12" in g.log
    assert "Wrote y from 2 to 12" in g.log
    assert "Wrote x from 12 to 22" in g.log
    assert "Wrote y from 12 to 22" in g.log
    assert "Wrote y from 22 to 32" in g.log
    assert "Wrote y from 22 to 32" in g.log
    assert "Wrote y from 32 to 37" in g.log
    assert "Wrote y from 32 to 37" in g.log


if __name__ == '__main__':
    test_offset()