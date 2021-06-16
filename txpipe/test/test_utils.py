from ..utils.misc import unique_list, hex_escape, chi2_ignoring_zeros
import numpy as np
import pytest

def test_escape():
    assert hex_escape(chr(1))=='\\x01'
    assert hex_escape(chr(7) + chr(11) + chr(12))=='\\x07\\x0b\\x0c'

    assert hex_escape("aaa\nbbb") == "aaa\nbbb"
    assert hex_escape("xxx\nyyy", replace_newlines=True) == "xxx\\x0ayyy"

def test_unique_list():

    x = [1, 1, 1, 1]
    assert unique_list(x) == [1]

    x = [1, 1, 2, 1, 1]
    assert unique_list(x) == [1, 2]

    x = [1, 1, 2, 1, 1, 2, 3, 3]
    assert unique_list(x) == [1, 2, 3]

    x = [-1, "cat", -1, "cat", "dog"]
    assert unique_list(x) == [-1, "cat", "dog"]


def test_chi2_ignoring_zeros():
    s = np.ones(5)
    C = 2 * np.eye(5)
    chi2, n = chi2_ignoring_zeros(s, C)
    assert n == 5
    assert np.isclose(chi2, 2.5)
    s[-1] = 0
    C[-1, -1] = 0
    chi2, n = chi2_ignoring_zeros(s, C)
    assert n == 4
    assert np.isclose(chi2, 2.0)
    C[-2, -2] = 0
    with pytest.raises(ValueError):
        chi2, n = chi2_ignoring_zeros(s, C)
    s[-2] = 0
    C[-2, 0] = 0.001
    with pytest.raises(ValueError):
        chi2, n = chi2_ignoring_zeros(s, C)
