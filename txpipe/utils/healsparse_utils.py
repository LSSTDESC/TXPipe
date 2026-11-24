"""
Helpers for working with healsparse maps one coverage pixel at a time.

Calling ``hsp_map.valid_pixels`` on a high-resolution map (like a mask)
builds a single array of every valid pixel in the whole map, which can use
a lot of memory. Iterating over coverage pixels instead keeps only a small
chunk of the map in memory at once.
"""
import numpy as np


def iter_valid_pixels(hsp_map):
    """
    Iterate through the valid pixels of a healsparse map, one coverage
    pixel at a time.

    Parameters
    ----------
    hsp_map : healsparse.HealSparseMap
        The map to iterate over.

    Yields
    ------
    pixels : np.ndarray
        The valid (nest-ordered) pixel indices within one coverage pixel
    values : np.ndarray
        The map values at those pixels
    """
    for pixels in hsp_map.iter_valid_pixels_by_covpix():
        if pixels.size == 0:
            continue
        yield pixels, hsp_map[pixels]


def valid_pixel_sum(hsp_map):
    """
    Sum the values of a healsparse map over all its valid pixels, iterating
    over coverage pixels to limit memory use.

    Parameters
    ----------
    hsp_map : healsparse.HealSparseMap
        The map to sum.

    Returns
    -------
    total : float
        Sum of the map values in all valid pixels
    """
    total = 0.0
    for _, values in iter_valid_pixels(hsp_map):
        total += np.sum(values, dtype=np.float64)
    return total
