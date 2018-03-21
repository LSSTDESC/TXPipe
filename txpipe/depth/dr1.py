"""
The strategy used in this file for determining depth is to take the
mean and stadard deviation of the 

"""

import numpy as np
import healpy as hp
from ..utils.stats import ParallelStatsCalculator


def dr1_values_iterator(data_iterator, nside, snr_threshold, snr_delta=1., mag_index=1):
    """Loop through any input data in
    data_iterator and computes the quantities needed for depth
    estimation - the magnitudes of any objects that have
    |SNR - SNR_threshold| < snr_delta.

    This is passed on to the stats calculator later.

    This way of writing the code makes it automatically parallel
    and never requires loading in the full columns.

    Parameters
    ----------
    data_iterator: iterable
        Iterator yielding chunks of data containing ra, dec, snr and magnitude values

    nside: int
        Resolution parameter for generated map

    snr_threshold: float
        Value of SNR to use as the depth (e.g. 5.0 for 5 sigma depth)

    snr_delta: float, optional
        Half-width of the SNR values to use for the depth estimation

    mag_index: int, optional
        Index describing which magntiude to use.  Meaning depends
        on what magnitudes were run in the shear estimation.
        This needs to be updated later.

    Yields
    -------
    pixel: int
        Index of pixel for a set of measured magnitudes
    magnitudes
        Magnitudes of objects close to the SNR threshold
    """

    # Loop through data
    for data in data_iterator:
        ra = data['ra']
        dec = data['dec']
        snr = data['mcal_s2n_r']
        mags = data['mcal_mag'][:,mag_index]
        # Get healpix pixels
        pix_nums = hp.ang2pix(nside, ra, dec, lonlat=True)

        # For each found pixel find all values hitting that pixel
        # and yield the index and their magnitudes
        for p in np.unique(pix_nums):
            mask = (pix_nums==p) & (abs(snr-snr_threshold)<snr_delta)
            yield p, mags[mask]


def dr1_depth(data_iterator, nside, snr_threshold, snr_delta=1.0, mag_index=1, sparse=False, comm=None):
    """
    Parameters
    ----------
    data_iterator: iterable
        Iterator yielding chunks of data containing ra, dec, snr and magnitude values

    nside: int
        Resolution parameter for generated map

    snr_threshold: float
        Value of SNR to use as the depth (e.g. 5.0 for 5 sigma depth)

    snr_delta: float, optional
        Half-width of the SNR values to use for the depth estimation

    mag_index: int, optional
        Index describing which magntiude to use.  Meaning depends
        on what magnitudes were run in the shear estimation.
        This needs to be updated later.

    sparse: bool, optional
        Whether to use sparse indexing for the calculation.  Faster if only a small number of pixels
        Are used.

    comm: MPI communicator, optional
        An MPI comm for parallel processing.  If None, calculation is serial.

    """
    npix = hp.nside2npix(nside)

    # Make the stats calculator object
    stats = ParallelStatsCalculator(npix, sparse=sparse)

    # Make the iterator that will supply values to the stats calc
    values_iterator = dr1_values_iterator(data_iterator, nside, snr_threshold, snr_delta)

    # run the stats
    count, depth, depth_var = stats.calculate(values_iterator, comm)

    # Generate the pixel indexing (if parallel and the master process) and 
    # convert from sparse arrays to pixel, index arrays.if sparse
    if count is None:
        pixel = None
    elif sparse:
        pixel, count = count.to_arrays()
        _, depth = depth.to_arrays()
        _, depth_var = depth_var.to_arrays()
    else:
        pixel = np.arange(len(depth))

    return pixel, count, depth, depth_var