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


def chi2_ignoring_zeros(d, C):
    """
    Compute the chi^2 value of a vector and its covariance matrix, but
    ignore elements that are zero in both the data vector and rows and
    columns of the matrix.

    Parameters
    ----------
    d: array
        1D data - model
    C: array
        2D covariance

    Returns
    -------
    chi2: float
        total chi2 value
    n: int
        number of data points included
    """
    x = C.diagonal()
    # Find elements to be cut
    cut = (x == 0)

    # If the diagonal is zero then the off-diagonals must be too, and the data vector
    if (d[cut] != 0).any() or (C[cut, :] != 0).any() or (C[:, cut] != 0).any():
        raise ValueError("In chi2_ignoring_zeros, some data points with zero "
                         "variance had non-zero data points")

    # Cut down the the good data points
    keep = ~cut
    d = d[keep]
    C = C[keep][:, keep]

    # Get the chi2 of the remaining points
    P = np.linalg.inv(C)
    return d @ P @ d, keep.sum()
