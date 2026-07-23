from ..utils.misc import unique_list
from ..ingest.mocks import test as run_test_mock_noise
import os
import numpy as np
import tempfile

def test_unique_list():
    x = [1, 1, 1, 1]
    assert unique_list(x) == [1]

    x = [1, 1, 2, 1, 1]
    assert unique_list(x) == [1, 2]

    x = [1, 1, 2, 1, 1, 2, 3, 3]
    assert unique_list(x) == [1, 2, 3]

    x = [-1, "cat", -1, "cat", "dog"]
    assert unique_list(x) == [-1, "cat", "dog"]

def test_mock_noise():
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as dirname:
        try:
            os.chdir(dirname)
            run_test_mock_noise()
            for b in "ugrizy":
                assert os.path.exists(f"snr_{b}.png")
                assert os.path.exists(f"mag_{b}.png")
        finally:
            os.chdir(cwd)
