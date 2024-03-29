from ..utils.misc import unique_list, hex_escape
import numpy as np


def test_escape():
    assert hex_escape(chr(1)) == "\\x01"
    assert hex_escape(chr(7) + chr(11) + chr(12)) == "\\x07\\x0b\\x0c"

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
