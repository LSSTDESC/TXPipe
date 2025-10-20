import numpy as np
from ..utils import import_dask

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
    return {
        "pix": pix, 
        "count_map": count_map, 
        "depth_map": depth_map, 
        "depth_var": depth_var,
    }

def make_dask_depth_map_det_prob(
    ra, dec, mag, det, det_prob_threshold, mag_delta, min_depth, max_depth, pixel_scheme, 
    smooth_det_frac=False, smooth_window=0.5,
):
    """
    Generate a depth map using Dask, by finding the mean magnitude of
    objects with a signal-to-noise ratio close to a given threshold.

    Parameters
    ----------
    ra : dask.array
        Right Ascension coordinates of injected SSI objects in degrees.
    dec : dask.array
        Declination coordinates of injected SSI objects in degrees.
    mag : dask.array
        True magnitudes of the injected objects, in band of user's choice
    det : dask.array
        dask array of boolean detection parameter
    det_prob_threshold : float
        Detection probability threshold for SSI depth 
        (i.e. 0.9 to get magnitude at which 90% of brighter objects are detected)
    mag_delta : float
        size of the magnitude increments at which to compute detection fraction
    min_depth : float
        minimum magnitude at which to compute detection fraction
    max_depth : float
        maximum magnitude at which to compute detection fraction
    pixel_scheme : PixelScheme
        An object that provides pixelization scheme with methods `npix` and `ang2pix`.
    smooth_det_frac: bool
        if True apply a savgol filtering to the individual detection frac vs magnitude cut signals
    smooth_window: float
        if smooth_det_frac==True, this is the window size of the filter (in magnitudes)

    Returns
    -------
    tuple
        A tuple containing:
        - pix (dask.array): Unique pixel indices.
        - det_count_map (dask.array): Count of detected objects per pixel.
        - inj_count_map (dask.array): Count of injected objects per pixel.
        - depth_map (dask.array): Mean depth per pixel.
        - det_frac_by_mag_thres (dask.array): stack of maps. fraction of detections brighter than mag_edges
        - mag_edges (dask.array): grid of magnitudes at which det_frac_by_mag_thres was evaluated
    """
    from scipy.signal import savgol_filter
    _, da = import_dask()
    npix = pixel_scheme.npix
    pix = pixel_scheme.ang2pix(ra, dec)

    det_count_map = da.bincount(pix, weights=det, minlength=npix)
    inj_count_map = da.bincount(pix, minlength=npix)

    # Make array of magnitude bins
    mag_edges = da.arange(min_depth, max_depth, mag_delta)
    n_depth_bins = len(mag_edges)

    # loop over mag bins
    # TODO: add option to compute fraction *at* each magnitude, rather than below
    frac_list = []
    for mag_thresh in mag_edges:
        above_thresh = mag < mag_thresh
        ntot = da.bincount(pix, weights=above_thresh, minlength=npix)
        ndet = da.bincount(pix, weights=above_thresh * det, minlength=npix)
        frac_det = da.where(ntot != 0, ndet / ntot, np.nan)
        frac_list.append(frac_det)
    det_frac_by_mag_thres = da.stack(frac_list)

    # Optional smoothing of the stacked detection fractions
    if smooth_det_frac:
        window_length = int(n_depth_bins*smooth_window/(max_depth-min_depth)) #converting to units of mag bin
        poly_order = 2 # TODO: could make config option

        # Here extend the chunks of the dask array when applying a local filter to avoid boundary issues
        det_frac_by_mag_thres = det_frac_by_mag_thres.map_overlap( 
            lambda a: savgol_filter(a, window_length, poly_order, axis=0), 
            depth=window_length // 2,   # Extend chunks by half window size
            boundary="reflect",         # Reflect at edges to avoid NaNs
            dtype=det_frac_by_mag_thres.dtype, 
            )

    # In order for pixel to give a valid depth estimate it must have 
    # (1) at least one mag_thresh with a computed frac_det above the threshold
    # (2) at least one mag_thresh with a computed frac_det below the threshold
    has_high = (det_frac_by_mag_thres > det_prob_threshold).any(axis=0)
    has_low = (det_frac_by_mag_thres < det_prob_threshold).any(axis=0)
    valid_pix_mask = has_high & has_low

    # We define det frac depth as the magnitude at which detection fraction drops below the given threshold
    # If detection fraction fluctuates around the threshold (e.g. due to noise) we choose the brightest magnitude with det_frac < threshold
    below_threshold = det_frac_by_mag_thres < det_prob_threshold
    masked = da.where(
        below_threshold, da.arange(det_frac_by_mag_thres.shape[0])[:, None], n_depth_bins - 1
    )  #if below_threshold is True -> magnitude index, if False -> set to index of maximum depth 
    thres_index = da.nanmin(masked, axis=0) #the index of the magnitude where det frac drops below threshold

    depth_map = mag_edges[thres_index]
    depth_map[~valid_pix_mask] = np.nan

    pix = da.unique(pix)
    return {
        "pix": pix,
        "det_count_map": det_count_map,
        "inj_count_map": inj_count_map,
        "depth_map": depth_map,
        "det_frac_by_mag_thres": det_frac_by_mag_thres,
        "mag_edges": mag_edges
    }