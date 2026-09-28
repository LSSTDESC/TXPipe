"""
Reading survey-property ("systematics") maps for use as NaMaster templates.

Shared by every stage that correlates or deprojects survey-property maps, so
that they all read maps the same way. The reader works out each file's format
from its content rather than its file name, and returns full-sky HEALPix
arrays in RING ordering at the resolution of the mask, which is what TXPipe's
masks and density maps use.
"""
import glob
import os
import warnings

import numpy as np


def find_systematics_map_files(location):
    """List the systematics map files at `location`.

    Args:
        location: a directory (every regular file in it, symlinks followed),
            a glob pattern (e.g. "/path/*_nside1024.hsp"), a path prefix
            (every file whose path starts with it, like the
            ``supreme_path_root`` options), or a list of file paths.

    Returns:
        sorted list of file paths
    """
    if isinstance(location, (list, tuple)):
        files = list(location)
    elif os.path.isdir(location):
        files = [os.path.join(location, f) for f in os.listdir(location)]
    elif glob.has_magic(location):
        files = glob.glob(location)
    else:
        files = glob.glob(glob.escape(location) + "*")
    return sorted(f for f in files if os.path.isfile(f))


def _is_healsparse(path):
    """True if `path` is a HealSparse map, whatever its file extension."""
    import healsparse

    try:
        healsparse.HealSparseCoverage.read(path)
    except Exception:
        return False
    return True


def read_map_ring(path, nside, reduction="mean"):
    """Read a HEALPix or HealSparse map as a full-sky RING array at `nside`.

    Args:
        path: map file. HealSparse maps (any extension) are detected from their
            content; anything else is read with ``healpy.read_map``, which
            converts NESTED files to RING using their ORDERING header.
        nside: output resolution.
        reduction: HealSparse reduction used when degrading a finer map
            ('mean', 'median', 'std', 'max', 'min', 'sum', 'prod').

    Returns:
        (values, valid): float64 array of map values and a boolean array that
        is True where the map has a finite value.
    """
    import healpy as hp

    npix = hp.nside2npix(nside)
    if _is_healsparse(path):
        import healsparse

        m = healsparse.HealSparseMap.read(path)
        if m.dtype.names is not None or m.is_wide_mask_map:
            raise ValueError(f"{path} is a multi-field or wide-mask HealSparse map, not a single-valued map")
        if m.nside_sparse > nside:
            m = m.degrade(nside, reduction=reduction)
        elif m.nside_sparse < nside:
            warnings.warn(
                f"{path} has nside={m.nside_sparse}, coarser than nside={nside}; "
                "upgrading by nearest-neighbour replication."
            )
            m = m.upgrade(nside)
        # valid_pixels are NESTED: convert explicitly rather than relying on
        # generate_healpix_map's ordering and sentinel conventions.
        vpix = m.valid_pixels
        ring = hp.nest2ring(nside, vpix)
        values = np.zeros(npix)
        valid = np.zeros(npix, dtype=bool)
        values[ring] = m[vpix]
        valid[ring] = True
    else:
        values = hp.read_map(path, nest=False, dtype=np.float64)
        if hp.get_nside(values) != nside:
            if hp.get_nside(values) < nside:
                warnings.warn(
                    f"{path} has nside={hp.get_nside(values)}, coarser than nside={nside}; "
                    "upgrading by nearest-neighbour replication."
                )
            values = hp.ud_grade(values, nside, order_in="RING", order_out="RING")
        valid = np.abs(values - hp.UNSEEN) > 1e-5 * np.abs(hp.UNSEEN)
    valid &= np.isfinite(values)
    values[~valid] = 0.0
    return values, valid


def read_systematics_maps(location, mask, reduction="mean"):
    """Read survey-property maps and prepare them as NaMaster templates.

    Each map is read at the resolution of `mask` in RING ordering, the mean
    (weighted by `mask`, over pixels where the map has a value) is
    subtracted, and pixels inside the mask where the map has no value are set
    to that mean, i.e. to zero after subtraction. Setting them to zero before
    subtracting would give them a value of -mean: a spurious template tracing
    the map's coverage. Pixels outside the mask are set to zero. Maps that
    can't be read, have no value inside the mask, or are constant inside it
    are skipped with a warning.

    Args:
        location: see :func:`find_systematics_map_files`.
        mask: full-sky RING mask (weights; zero outside the footprint).
        reduction: HealSparse reduction used when degrading finer maps.

    Returns:
        (maps, names): list of float64 full-sky RING arrays and the files they
        were read from.
    """
    import healpy as hp

    nside = hp.get_nside(mask)
    unmasked = mask > 0
    maps, names = [], []
    for path in find_systematics_map_files(location):
        try:
            values, valid = read_map_ring(path, nside, reduction=reduction)
        except Exception as exc:
            warnings.warn(f"Could not read systematics map {path}: {exc}. Skipping.")
            continue

        use = unmasked & valid
        if not use.any():
            warnings.warn(f"Systematics map {path} has no values within the mask; skipping.")
            continue
        n_missing = np.sum(unmasked & ~valid)
        if n_missing:
            warnings.warn(
                f"Systematics map {path} has no value in {n_missing} of {unmasked.sum()} "
                "mask pixels; filling them with the mask-weighted mean."
            )

        mean = np.sum(values[use] * mask[use]) / np.sum(mask[use])
        template = np.zeros_like(values)
        template[use] = values[use] - mean
        if np.all(template[use] == 0.0):
            warnings.warn(f"Systematics map {path} is constant within the mask; skipping.")
            continue

        print(f"Read systematics map {path} (mean on mask {mean:.6g} subtracted)")
        maps.append(template)
        names.append(path)

    print(f"Total systematics maps read: {len(maps)}")
    return maps, names
