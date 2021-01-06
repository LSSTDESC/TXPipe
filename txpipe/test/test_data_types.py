from ..data_types import PickleFile
import tempfile
import os

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
