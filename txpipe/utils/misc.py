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


PRINTABLE_CHARS = string.ascii_letters + string.digits + string.punctuation + " "
PRINTABLE_CHARS_INC_NEWLINES = PRINTABLE_CHARS + "\r\n"

def hex_escape(s, replace_newlines=False):
    """
    Replace non-printable characters in a string with hex-code equivalents
    so they can be printed or saved to a FITS file header.  Newline characters
    will be replaced if replace_newlines is set to true

    Based on:  https://stackoverflow.com/a/13935582

    Parameters
    ----------
    s: str
        The initial string
    replace_newlines: bool
    Whether to include newline characters in the replacement
    Returns
    str
        Same string with unprintable chars replaced with hex codes, e.g.
        the bell character becomes "\x07"
    """
    chars = PRINTABLE_CHARS if replace_newlines else PRINTABLE_CHARS_INC_NEWLINES
    return "".join(
        c if c in chars else r"\x{0:02x}".format(ord(c)) for c in s
    )


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
    #  avoids doing the hash twice per object in the bin
    index = {b: i for i, b in enumerate(matches)}

    # the maximum possible length of these where results
    # is the full length of the array.  at the risk of being
    # wasteful, allocate these for each bin
    wheres = [np.empty(n, dtype=int) for b in matches]
    # we will track how many results are actually found for
    # each
    counts = [0 for b in matches]
    for i, v in enumerate(x):
        j = index.get(v)
        # if this is one of the matches we are looking for,
        # record it
        if j is not None:
            # record the index in the current position
            # for this match
            c = counts[j]
            wheres[j][c] = i
            # and update so the next match goes in the
            # next cell
            counts[j] += 1
    # finally, cut down to just the valid parts
    return {b: wheres[i][: counts[i]] for i, b in enumerate(matches)}
