from ..utils.misc import unique_list, hex_escape


def test_escape():
    assert hex_escape(chr(1))=='\\x01'
    assert hex_escape(chr(10) + chr(11) + chr(12))=='\\x0a\\x0b\\x0c'


def test_unique_list():

    x = [1, 1, 1, 1]
    assert unique_list(x) == [1]

    x = [1, 1, 2, 1, 1]
    assert unique_list(x) == [1, 2]

    x = [1, 1, 2, 1, 1, 2, 3, 3]
    assert unique_list(x) == [1, 2, 3]

    x = [-1, "cat", -1, "cat", "dog"]
    assert unique_list(x) == [-1, "cat", "dog"]

def test_multi_where():
    x = np.random.randint(0, 10, 100)
    w = [1, 2, 3]

    a = multi_where(x, w)

    for m in w:
        assert np.array_equal(np.where(x==m)[0], a[m])

    assert a.keys() == set(w)
