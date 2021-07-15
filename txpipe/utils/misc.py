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
    -------
    str
        Same string with unprintable chars replaced with hex codes, e.g.
        the bell character becomes "\x07"
    """
    chars = PRINTABLE_CHARS if replace_newlines else PRINTABLE_CHARS_INC_NEWLINES
    return "".join(
        c if c in chars else r"\x{0:02x}".format(ord(c)) for c in s
    )

def rename_iterated(it, renames):
    """
    Rename the items in dictionaries yielded by an interator.

    In several places we load data from catalogs chunk by chunk,
    yielding a dictionary of data each time. This renames columns
    in each chunk.

    Parameters
    ----------
    it: iterator
        Must yield dictionaries or equivalent
    renames: dict
        dictionary of old names to new names
    """
    for s, e, data in it:
        for old, new in renames.items():
            # rename the column
            data[new] = old
            # delete the old column
            del data[old]
            yield s, e, data
