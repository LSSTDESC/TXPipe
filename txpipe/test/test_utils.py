from ..utils.misc import unique_list, hex_escape


def test_escape():
	assert hex_escape(chr(1))=='\\x01'
	assert hex_escape(chr(10) + chr(11) + chr(12))=='\\x0a\\x0b\\x0c'


def test_unique_list():
	a = unique_list([1, 2, 3, "a", "b", "a", 2, 0, 0, -1, ])
	assert a == [1, 2, 3, "a", "b", 0, -1]
