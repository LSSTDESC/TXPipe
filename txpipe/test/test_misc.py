from ..utils.misc import unique_list
import numpy as np


def test_unique_list():

	x = [1, 1, 1, 1]
	assert unique_list(x) == [1]

	x = [1, 1, 2, 1, 1]
	assert unique_list(x) == [1, 2]

	x = [1, 1, 2, 1, 1, 2, 3, 3]
	assert unique_list(x) == [1, 2, 3]

	x = [-1, "cat", -1, "cat", "dog"]
	assert unique_list(x) == [-1, "cat", "dog"]
