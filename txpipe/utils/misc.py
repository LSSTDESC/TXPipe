import hashlib
import string
import numpy as np

def array_hash(x):
    b = x.tobytes()
    # We do not need a cryptographic hash here
    return int(hashlib.md5(b).hexdigest(), 16)


def unique_list(seq):
    """
    Find the unique elements in a list or other sequence
    while maintaining the order. (i.e., remove any duplicated
    elements but otherwise leave it the same)

    Method from:
    https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order

    Parameters
    ----------
    seq: list or sequence
        Any input object that can be iterated

    Returns
    -------
    L: list
        a new list of the unique objects
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


PRINTABLE_CHARS = string.ascii_letters + string.digits + string.punctuation + ' '

def hex_escape(s):
    return ''.join(c if c in PRINTABLE_CHARS else r'\x{0:02x}'.format(ord(c)) for c in s)


def multi_where(x, matches):
    """
    Return the equivalent of {m: np.where(x==m)[0] for m in matches}
    but requiring only a single pass through x and so faster in some
    cases
    
    Parameters
    ----------
    x: array
        Values to be matches

    matches: collection
        Values against which each item in x will be compared

    Returns
    -------
    w: dict
        
    """
    n = len(x)
    # avoids doing the hash twice per object in the bin
    index = {b: i for i, b in enumerate(matches)}

    wheres = [np.empty(n, dtype=int) for b in matches]
    counts = [0 for b in matches]
    for i, v in enumerate(x):
        j = index.get(v)
        if j is not None:
            c = counts[j]
            wheres[j][c] = i
            counts[j] += 1
    return {b: wheres[i][:counts[i]] for i, b in enumerate(matches)}
