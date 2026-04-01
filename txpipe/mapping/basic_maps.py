from ..utils import import_dask

import numpy as np
import healpy


def make_dask_flag_maps(ra, dec, flag, max_exponent, pixel_scheme, cov_map):
    """
    Create maps of flag counts per pixel

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

    cov_map : HealSparseCoverage
        coverage map corresponding to these sources (or a superset of them)

    Returns
    -------
    maps : list of dask arrays
        List of length max_exponent, where maps[i] contains the count of
        objects with the 2^i flag bit set in each sparse pixel.
    """
    _, da = import_dask()
    pix = pixel_scheme.ang2pix(ra, dec)

    # get the sparse map index for each of these pixel (is dask aware)
    sparse_index, npix_sparse = pix2sparseindex(pix, cov_map)

    maps = []
    for i in range(max_exponent):
        f = 2**i
        flag_map = da.where(flag & f, 1, 0)
        flag_map = da.bincount(
            sparse_index, weights=flag_map, minlength=npix_sparse
        ).astype(int)
        maps.append(flag_map)

    return maps


def make_dask_shear_maps(
    ra, dec, g1, g2, weight, pixel_scheme, cov_map, sentinel=healpy.UNSEEN
):
    """
    Create weighted shear maps per pixel from a galaxy catalog.

    Computes count, mean shear (g1, g2), lensing weight, shape noise (esq),
    and variance maps over the sparse healpix pixelization defined by cov_map.

    Parameters
    ----------
    ra : dask array
        Right ascension in degrees.
    dec : dask array
        Declination in degrees.
    g1 : dask array
        First shear component for each object.
    g2 : dask array
        Second shear component for each object.
    weight : dask array
        Lensing weight for each object.
    pixel_scheme : PixelScheme
        The pixelization scheme to use, typically Healpix with a given nside.
    cov_map : HealSparseCoverage
        Coverage map corresponding to these sources (or a superset of them).
    sentinel : float, optional
        Value to fill empty pixels with. Default is healpy.UNSEEN.

    Returns
    -------
    dict with keys:
        count_map : dask array
            Number of objects per pixel.
        g1_map : dask array
            Weighted mean g1 per pixel.
        g2_map : dask array
            Weighted mean g2 per pixel.
        weight_map : dask array
            Sum of lensing weights per pixel.
        esq_map : dask array
            Weighted mean squared ellipticity (shape noise) per pixel,
            computed as sum(w^2 * 0.5 * (g1^2 + g2^2)).
        var1_map : dask array
            Weighted variance on the mean g1 per pixel.
        var2_map : dask array
            Weighted variance on the mean g2 per pixel.
    """
    _, da = import_dask()

    npix = pixel_scheme.npix
    # This seems to work directly, but we should check performance
    pix = pixel_scheme.ang2pix(ra, dec)

    # get the sparse map index for each of these pixel (is dask aware)
    sparse_index, npix_sparse = pix2sparseindex(pix, cov_map)

    # count map is just the number of galaxies per pixel
    count_map = da.bincount(sparse_index, minlength=npix_sparse)

    # For the other map we use bincount with weights - these are the
    # various maps by pixel. bincount gives the number of objects in each
    # vaue of the first argument, weighted by the weights keyword, so effectively
    # it gives us
    # p_i = sum_{j} x[j] * delta_{pix[j], i}
    # which is out map
    weight_map = da.bincount(sparse_index, weights=weight, minlength=npix_sparse)
    g1_map = da.bincount(sparse_index, weights=weight * g1, minlength=npix_sparse)
    g2_map = da.bincount(sparse_index, weights=weight * g2, minlength=npix_sparse)
    esq_map = da.bincount(
        sparse_index, weights=weight**2 * 0.5 * (g1**2 + g2**2), minlength=npix_sparse
    )

    # normalize by weights to get the mean map value in each pixel
    g1_map /= weight_map
    g2_map /= weight_map

    # Generate a catalog-like vector of the means so we can
    # subtract from the full catalog.  Not sure if this ever actually gets
    # created, or if dask just keeps a conceptual reference to it.
    g1_mean = g1_map[sparse_index]
    g2_mean = g2_map[sparse_index]

    # Also generate variance maps
    var1_map = da.bincount(
        sparse_index, weights=weight * (g1 - g1_mean) ** 2, minlength=npix_sparse
    )
    var2_map = da.bincount(
        sparse_index, weights=weight * (g2 - g2_mean) ** 2, minlength=npix_sparse
    )

    # we want the variance on the mean, so we divide by both the weight
    # (to go from the sum to the variance) and then by the count (to get the
    # variance on the mean). Have verified that this is the same as using
    # var() on the original arrays.
    var1_map /= weight_map * count_map
    var2_map /= weight_map * count_map

    # The NaNs can occur if there are no objects in a pixel
    # we set these to the sentinel (usually hp.UNSEEN)
    valid = (count_map != 0) & (weight_map > 0)
    weight_map = da.where(valid, weight_map, sentinel)
    g1_map = da.where(valid, g1_map, sentinel)
    g2_map = da.where(valid, g2_map, sentinel)
    var1_map = da.where(valid, var1_map, sentinel)
    var2_map = da.where(valid, var2_map, sentinel)
    esq_map = da.where(valid, esq_map, sentinel)

    return {
        "count_map": count_map,
        "g1_map": g1_map,
        "g2_map": g2_map,
        "weight_map": weight_map,
        "esq_map": esq_map,
        "var1_map": var1_map,
        "var2_map": var2_map,
    }


def make_dask_lens_maps(
    ra, dec, weight, tomo_bin, target_bin, pixel_scheme, cov_map, sentinel=healpy.UNSEEN
):
    """
    Create count and weight maps for a lens galaxy sample in a tomographic bin.

    Parameters
    ----------
    ra : dask array
        Right ascension in degrees.
    dec : dask array
        Declination in degrees.
    weight : dask array
        Weight for each object.
    tomo_bin : dask array
        Tomographic bin index for each object. Objects with tomo_bin < 0
        are considered unassigned.
    target_bin : int or "2D"
        The tomographic bin to select. If "2D", all objects with tomo_bin >= 0
        are counted.
    pixel_scheme : PixelScheme
        The pixelization scheme to use, typically Healpix with a given nside.
    cov_map : HealSparseCoverage
        Coverage map corresponding to these sources (or a superset of them).
    sentinel : float, optional
        Value to fill empty pixels with. Default is healpy.UNSEEN.

    Returns
    -------
    dict with keys:
        count_map : dask array
            Number of objects per pixel (unweighted). For "2D", counts all
            objects with tomo_bin >= 0; otherwise counts only target_bin objects.
        weight_map : dask array
            Sum of weights per pixel for target_bin objects. Empty pixels
            are set to sentinel.
    """
    # this will actually load numpy if a debug env var is set
    _, da = import_dask()

    pix = pixel_scheme.ang2pix(ra, dec)

    # one unweighted count and one weighted count
    if target_bin == "2D":
        hit = da.where(tomo_bin >= 0, 1, 0)
        weighted_hit = da.where(tomo_bin >= 0, weight, 0)
    else:
        hit = da.where(tomo_bin == target_bin, 1, 0)
        weighted_hit = da.where(tomo_bin == target_bin, weight, 0)

    # get the sparse map index for each of these pixel (is dask aware)
    sparse_index, npix_sparse = pix2sparseindex(pix, cov_map)

    # convert to maps and hit pixels
    count_map = da.bincount(sparse_index, weights=hit, minlength=npix_sparse)
    weight_map = da.bincount(sparse_index, weights=weighted_hit, minlength=npix_sparse)

    # mask out empty pixels
    weight_map = da.where(count_map != 0, weight_map, sentinel)

    return {
        "count_map": count_map,
        "weight_map": weight_map,
    }


def degrade_healsparse(hsp_map, degrade_nside, reduction, weight_map=None):
    """
    Degrade a HealSparseMap

    Implements reduction methods not included in healsparse by default.

    Parameters
    ----------
    hsp_map : HealSparseMap
        The map to degrade.
    degrade_nside : int
        The target nside resolution to degrade to. Must be coarser (smaller)
        than hsp_map.nside_sparse.
    reduction : str
        The reduction method to use. Must be one of:

        - "weightedmean" : Compute a weighted mean of the fine pixels within
          each coarse pixel, using weight_map as the weights. Pixels not
          present in weight_map are assumed to have weight 0 (which differs
          from healsparse's reuction="wmean"). Requires weight_map to be provided.

        - "mask" : Degrade a mask
          returns a map of the fractional coverage of each pixel.
          If the input mask is an binary (integer) map, we use healsparse's "fracdet_map" method
          If the mask is already a fractional (float) maps, we compute it ourselves as
          sum(frac_map) * (nside_new / nside_old)^2 in each coarse pixel.

    weight_map : HealSparseMap, optional
        Weight map to use for the "weightedmean" reduction. Must have the
        same nside_sparse as hsp_map. Required if reduction="weightedmean",
        must be None if reduction="mask".

    Returns
    -------
    map_out : HealSparseMap
        Degraded map at degrade_nside resolution.
    """
    import healsparse as hsp

    custom_reductions = ["weightedmean", "mask"]

    if reduction == "mask":
        assert weight_map is None, "weight_map not used for fractional mask degrade"

        # if the map is a binary mask, return the fracdet
        if np.issubdtype(hsp_map.dtype, np.integer):
            map_out = hsp_map.fracdet_map(degrade_nside)

        # if the map is already a fractional coverage map, sum(frac)/N_tot_subpix
        elif np.issubdtype(hsp_map.dtype, np.floating):
            map_degraded_sum = hsp_map.degrade(degrade_nside, reduction="sum")

            # only select unique pixels in the degraded map (I'm not sure why they sometimes duplicate)
            degraded_pixels = np.unique(map_degraded_sum.valid_pixels)

            map_out = hsp.HealSparseMap.make_empty_like(map_degraded_sum)
            map_out.update_values_pix(
                degraded_pixels,
                map_degraded_sum[degraded_pixels]
                * (degrade_nside / hsp_map.nside_sparse) ** 2.0,
            )
        else:
            raise RuntimeError(
                f"Map dtype is {hsp_map.dtype}, expected float-like or int-like for mask reduction"
            )

    elif reduction == "weightedmean":
        assert weight_map is not None
        assert hsp_map.nside_sparse == weight_map.nside_sparse

        # If the map is an integer map, first convert to floats
        if np.issubdtype(hsp_map.dtype, np.integer):
            valid_pix = hsp_map.valid_pixels
            m = hsp.HealSparseMap.make_empty(
                hsp_map.nside_coverage, hsp_map.nside_sparse, dtype=float
            )
            m.update_values_pix(valid_pix, hsp_map[valid_pix].astype(float))
        else:
            m = hsp_map

        xw = hsp.operations.product_intersection([weight_map, m])
        sumxw = xw.degrade(degrade_nside, reduction="sum")
        sumw = weight_map.degrade(degrade_nside, reduction="sum")
        map_out = hsp.operations.divide_intersection([sumxw, sumw])
    else:
        # reduction not in custom list, using healsparse degrade
        map_out = hsp_map.degrade(
            degrade_nside, reduction=reduction, weights=weight_map
        )

    return map_out


def make_coverage_map(ra, dec, pixel_scheme):
    """
    Create a map that describes which low-res "coverage pixels" will contain data

    Parameters
    ----------
    ra : dask.array
        Right ascension values of the objects.
    dec : dask.array
        Declination values of the objects.

    Returns
    -------
    Coverage map: HealSparseCoverage
    """
    _, da = import_dask()
    import healsparse as hsp

    cov_map = hsp.HealSparseCoverage.make_empty(
        pixel_scheme.nside_coverage, pixel_scheme.nside
    )

    pix = pixel_scheme.ang2pix(ra, dec)
    cov_pix = np.unique(cov_map.cov_pixels(pix))

    # we run compute here so we can return the HealSparseCoverage object
    (cov_pix,) = da.compute(cov_pix)
    cov_map.initialize_pixels(cov_pix)

    return cov_map


def pix2sparseindex(pix, cov_map, return_npix=True):
    """
    Convert healpix nest pixel IDs to indices into the HealSparseMap._sparse_map array.

    Parameters
    ----------
    pix : dask array of int
        Healpix pixel IDs in NEST scheme at nside_sparse resolution.
    cov_map : HealSparseCoverage
        Coverage map defining the sparse map layout.
    return_npix : bool, optional
        If True, also return the total length of the sparse map array.
        Default is True.

    Returns
    -------
    sparse_index : dask array of int
        Index into _sparse_map for each input pixel.
    npix_sparse : int
        Total number of elements in the sparse map array, including the
        sentinel block. Only returned if return_npix is True.
    """
    _, da = import_dask()

    if return_npix:
        # get the number of pixels in the coverage and sparse map
        ncov = np.sum(cov_map.coverage_mask)
        npix_sparse = (ncov + 1) * cov_map.nfine_per_cov

    # get coverage pixel for each healpix id
    bit_shift = cov_map._bit_shift
    cov_pix = da.right_shift(pix, bit_shift)

    # This needs to be a dask array
    cov_index_map = da.array(cov_map._cov_index_map)

    # convert healpix pixel ID to healsparse index
    sparse_index = cov_index_map[cov_pix] + pix

    if return_npix:
        return sparse_index, npix_sparse
    else:
        return sparse_index
