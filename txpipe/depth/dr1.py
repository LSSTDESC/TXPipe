import numpy as np
import healpy as hp
from ..utils.stats import ParallelStatsCalculator


def dr1_values_iterator(data_iterator, nside, snr_threshold):
    # 5sigma depth= mean of mags of galaxies with 4<SNR<6

    for data in data_iterator:
        ra = data['ra']
        dec = data['dec']
        snr = data['mcal_s2n_r']
        mags = data['mcal_mag'][:,1]
        pix_nums = hp.ang2pix(nside, ra, dec, lonlat=True)
        for p in np.unique(pix_nums):
            mask = (pix_nums==p) & (snr>snr_threshold-1) & (snr<snr_threshold+1)
            yield p, mags[mask]


def dr1_depth(data_iterator, nside, snr_threshold, sparse=False, comm=None):
    npix = hp.nside2npix(nside)
    stats = ParallelStatsCalculator(npix, sparse=sparse)
    values_iterator = dr1_values_iterator(data_iterator, nside, snr_threshold)
    count, mean, variance = stats.calculate(values_iterator, comm)
    return count, mean, variance