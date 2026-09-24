import random
import numpy as np


def radec_to_xyz(coords):
    ra_rad = np.radians(coords[:, 0])
    dec_rad = np.radians(coords[:, 1])
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return np.column_stack([x, y, z])


def jk_regions(X, njk, maxiter=100):
    from kmeans_radec import kmeans_sample, find_nearest

    # kmeans_radec's default first-pass sample size, max(2 sqrt(N), 10 njk), exceeds N
    # for the small per-bin cluster stacks, so cap it at N
    nsample = min(len(X), max(2 * np.sqrt(len(X)), 10 * njk))
    km = kmeans_sample(X, ncen=njk, nsample=nsample, maxiter=maxiter, verbose=0)
    labels = find_nearest(X, km.centers)
    return labels, km.centers


def compute_kmeans_jackknife_covariance(
    ensemble, njk, tan_component="tangential_comp", cross_component="cross_comp", seed=None, maxiter=100
):
    """
    Kmeans analogue of clmm.ClusterEnsemble.compute_jackknife_covariance:
    clusters are split into njk kmeans regions on the sky (instead of healpix
    pixels), each region is dropped in turn and the stack recomputed.
    Updates ensemble.cov with `tan_kmjk` and `cross_kmjk`.

    Returns the number of regions used and the region centers (cartesian).
    """
    from clmm.dataops import make_stacked_radial_profile

    if seed is not None:
        # kmeans_radec draws its initial samples with the stdlib random module
        random.seed(seed)
        np.random.seed(seed)

    X = np.vstack((ensemble["ra"], ensemble["dec"])).T
    njk = min(njk, len(X))
    labels, centers = jk_regions(X, njk=njk, maxiter=maxiter)
    unique_labels = np.unique(labels)
    gt_jack, gx_jack = [], []
    for drop_region in unique_labels:
        mask = labels != drop_region
        gt, gx = make_stacked_radial_profile(
            ensemble["radius"][mask],
            ensemble["W_l"][mask],
            [ensemble[tan_component][mask], ensemble[cross_component][mask]],
        )[1]
        gt_jack.append(gt)
        gx_jack.append(gx)
    n_jack = unique_labels.size
    coeff = (n_jack - 1) ** 2 / n_jack
    ensemble.cov["tan_kmjk"] = coeff * np.cov(np.transpose(gt_jack), bias=False, ddof=0)
    ensemble.cov["cross_kmjk"] = coeff * np.cov(np.transpose(gx_jack), bias=False, ddof=0)
    return n_jack, radec_to_xyz(centers)
