from ..utils import import_dask

import numpy as np


def make_dask_flag_maps(ra, dec, flag, max_exponent, pixel_scheme):
    """
    Flags are assumed to be bitmasks, and this makes maps showing
    the number of objects with each flag set in each pixel.

    Parameters
    ----------
    ra : dask array
        Right ascension in degrees

    dec : dask array
        Declination in degrees

    flag : dask array
        Flag values for each object

    max_exponent : int
        The maximum exponent for the flag.

    pixel_scheme : PixelScheme
        The pixelization scheme to use, typically Healpix with a given nside
    """
    _, da = import_dask()
    npix = pixel_scheme.npix
    pix = pixel_scheme.ang2pix(ra, dec)

    maps = []
    for i in range(max_exponent):
        f = 2**i
        flag_map = da.where(flag & f, 1, 0)
        flag_map = da.bincount(pix, weights=flag_map, minlength=npix).astype(int)
        maps.append(flag_map)
    pix = da.unique(pix)
    return pix, maps

def make_dask_shear_maps(ra, dec, g1, g2, weight, pixel_scheme,):
    _, da = import_dask()
    import healpy
    npix = pixel_scheme.npix
    # This seems to work directly, but we should check performance
    pix = pixel_scheme.ang2pix(ra, dec)

    # count map is just the number of galaxies per pixel
    count_map = da.bincount(pix, minlength=npix)

    # For the other map we use bincount with weights - these are the
    # various maps by pixel. bincount gives the number of objects in each
    # vaue of the first argument, weighted by the weights keyword, so effectively
    # it gives us
    # p_i = sum_{j} x[j] * delta_{pix[j], i}
    # which is out map
    weight_map = da.bincount(pix, weights=weight, minlength=npix)
    g1_map = da.bincount(pix, weights=weight * g1, minlength=npix)
    g2_map = da.bincount(pix, weights=weight * g2, minlength=npix)
    esq_map = da.bincount(pix, weights=weight**2 * 0.5 * (g1**2 + g2**2), minlength=npix)

    # normalize by weights to get the mean map value in each pixel
    g1_map /= weight_map
    g2_map /= weight_map

    # Generate a catalog-like vector of the means so we can
    # subtract from the full catalog.  Not sure if this ever actually gets
    # created, or if dask just keeps a conceptual reference to it.
    g1_mean = g1_map[pix]
    g2_mean = g2_map[pix]

    # Also generate variance maps
    var1_map = da.bincount(pix, weights=weight * (g1 - g1_mean)**2, minlength=npix)
    var2_map = da.bincount(pix, weights=weight * (g2 - g2_mean)**2, minlength=npix)

    # we want the variance on the mean, so we divide by both the weight
    # (to go from the sum to the variance) and then by the count (to get the
    # variance on the mean). Have verified that this is the same as using
    # var() on the original arrays.
    var1_map /= (weight_map * count_map)
    var2_map /= (weight_map * count_map)

    # replace nans with UNSEEN.  The NaNs can occur if there are no objects
    # in a pixel, so the value is undefined.
    g1_map[da.isnan(g1_map)] = healpy.UNSEEN
    g2_map[da.isnan(g2_map)] = healpy.UNSEEN
    var1_map[da.isnan(var1_map)] = healpy.UNSEEN
    var2_map[da.isnan(var2_map)] = healpy.UNSEEN
    esq_map[da.isnan(esq_map)] = healpy.UNSEEN

    return count_map, g1_map, g2_map, weight_map, esq_map, var1_map, var2_map


def make_dask_lens_maps(ra, dec, weight, tomo_bin, target_bin, pixel_scheme):
    # this will actually load numpy if a debug env var is set
    _, da = import_dask()

    # pixel scheme
    import healpy
    npix = pixel_scheme.npix
    pix = pixel_scheme.ang2pix(ra, dec)

    # one unweighted count and one weighted count
    if target_bin == "2D":
        hit = da.where(tomo_bin >= 0, 1, 0)
    else:
        hit = da.where(tomo_bin == target_bin, 1, 0)
    weighted_hit = da.where(tomo_bin == target_bin, weight, 0)
    
    # convert to maps and hit pixels
    count_map = da.bincount(pix, weights=hit, minlength=npix)
    weight_map = da.bincount(pix, weights=weighted_hit, minlength=npix)
    pix = da.unique(pix)

    # mask out nans
    count_map[da.isnan(weight_map)] = healpy.UNSEEN
    weight_map[da.isnan(weight_map)] = healpy.UNSEEN

    return pix, count_map, weight_map




