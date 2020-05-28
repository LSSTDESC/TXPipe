import numpy as np

def dilated_healpix_map(m):
    """
    Dilate a healpix map - every pixel with a neighbour
    that is UNSEEN gets set to unseen as well

    Parameters
    ----------
    m: array
        Healpix float map

    Returns
    -------
    m2: array
        Matching-sized map with edge pixels UNSEEN
    """
    import healpy
    npix = m.size
    nside = healpy.npix2nside(npix)
    hit = np.where(m != healpy.UNSEEN)[0]
    neighbours = healpy.get_all_neighbours(nside, hit)

    bad = np.any(m[neighbours]==healpy.UNSEEN, axis=0)
    bad_index = hit[bad]

    m2 = m.copy()
    m2[bad_index] = healpy.UNSEEN
    return m2