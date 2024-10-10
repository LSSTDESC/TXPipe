import numpy as np
from ..utils import import_dask
from parallel_statistics import ParallelMeanVariance

def make_dask_bright_object_map(ra, dec, mag, extended, threshold, pixel_scheme):
    """
    Create a map of bright objects using Dask.

    Parameters
    ----------
    ra : dask.array
        Right ascension values of the objects.
    dec : dask.array
        Declination values of the objects.
    mag : dask.array
        Magnitude values of the objects.
    extended : dask.array
        Extendedness flag of the objects (0 for point sources, 1 for extended sources).
    threshold : float
        Magnitude threshold to classify objects as bright.
    pixel_scheme : PixelScheme
        Pixelization scheme object with methods `npix` and `ang2pix`.

    Returns
    -------
    tuple
        A tuple containing:
        - pix (dask.array): Unique pixel indices containing bright objects.
        - bright_object_count_map (dask.array): Count map of bright objects per pixel.
    """
    _, da = import_dask()
    npix = pixel_scheme.npix
    pix = pixel_scheme.ang2pix(ra, dec)
    bright = da.where((mag < threshold) & (extended == 0), 1, 0)
    bright_object_count_map = da.bincount(pix, weights=bright, minlength=npix).astype(int)
    pix = da.unique(pix)
    return pix, bright_object_count_map

def make_dask_depth_map(ra, dec, mag, snr, threshold, delta, pixel_scheme):
    """
    Generate a depth map using Dask, by finding the mean magnitude of
    objects with a signal-to-noise ratio close to a given threshold.

    Parameters
    ----------
    ra : dask.array
        Right Ascension coordinates in degrees.
    dec : dask.array
        Declination coordinates in degrees.
    mag : dask.array
        Magnitudes of the objects, in band of user's choice
    snr : dask.array
        Signal-to-noise ratios of the objects, in the same band.
    threshold : float
        Threshold value for signal-to-noise ratio.
    delta : float
        Tolerance value for signal-to-noise ratio.
    pixel_scheme : PixelScheme
        An object that provides pixelization scheme with methods `npix` and `ang2pix`.

    Returns
    -------
    tuple
        A tuple containing:
        - pix (dask.array): Unique pixel indices.
        - count_map (dask.array): Count of objects per pixel.
        - depth_map (dask.array): Mean depth per pixel.
        - depth_var (dask.array): Variance of depth per pixel.
    """
    _, da = import_dask()
    npix = pixel_scheme.npix
    pix = pixel_scheme.ang2pix(ra, dec)
    hit = da.where(abs(snr - threshold) < delta, 1, 0)
    depth = da.where(abs(snr - threshold) < delta, mag, 0)

    # get the count and sum of the depth and depth^2
    count_map = da.bincount(pix, weights=hit, minlength=npix)
    depth_map = da.bincount(pix, weights=depth, minlength=npix)
    depth2_map = da.bincount(pix, weights=depth**2, minlength=npix)

    # convert to means
    depth_map /= count_map
    depth2_map /= count_map

    # get the variance from the mean depth
    depth_var = depth2_map - depth_map**2

    pix = da.unique(pix)
    return pix, count_map, depth_map, depth_var