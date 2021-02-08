from ..data_types import PickleFile, DataFile
import tempfile
import os
import pytest
from io import UnsupportedOperation

class PicklableTestClass:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z



def test_pickle_file():
    extra_provenance = {"input": "something"}
    obj = PicklableTestClass(1.0, "2", 3)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'something')
        p = PickleFile(path, "w", extra_provenance=extra_provenance)
        p.write(obj)
        p.close()

        p = PickleFile(path, "r")
        assert p.provenance["input"] == "something"
        obj2 = p.read()

        assert obj == obj2

def test_mode_error():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'something')
        with pytest.raises(ValueError):
            p = DataFile(path, "q")
        with pytest.raises(ValueError):
            p = DataFile(path, "rw")

def test_unsupported():
    # TODO add a loop over file types here once
    # we have converted other File classes to raise
    # UnsupportedOperation instead of ValueError
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'something')    
        # check we get the right kind of error if
        # we try to read provenance info from a write-only
        # file
        p = PickleFile(path, "w")
        with pytest.raises(UnsupportedOperation):
            p.read_provenance()
        p.close()
        # Now that we've made the (albeit empty) file,
        # check that the write_provenance fails in the
        # same way
        p = PickleFile(path, "r")
        with pytest.raises(UnsupportedOperation):
            p.write_provenance()
        p.close()
