import numpy as np
from ..mapping import Mapper
from ..utils import choose_pixelization
import healpy


def test_mapper():
    # 12 pixels
    nside = 1
    scheme = choose_pixelization(pixelization='healpix', nside=nside)
    mapper = Mapper(scheme, [0], [0])
    npix = healpy.nside2npix(nside)
    pix = np.arange(npix)
    ra, dec = healpy.pix2ang(nside, pix, lonlat=True)

    N = 5
    for i in range(N):
        data = {
            'ra': ra,
            'dec': dec,
            'source_bin': np.zeros(npix, dtype=float),
            'weight': np.ones(npix, dtype=float),
            'lens_bin': np.zeros(npix, dtype=float),
            'g1': pix.astype(float) * 2,
            'g2': np.ones(npix) * i,
        }
        # same data multiple times so we can
        # get variances
        mapper.add_data(data)

    pixel, ngal, g1, g2, var_g1, var_g2, source_weight = mapper.finalize()

    mu_2 = (N - 1) / 2
    # variance
    var_2 = (N - 1) * (2 * N - 1) / 6 - mu_2 ** 2
    # variance on mean
    var_2 /= N

    # should see all pixels
    assert np.allclose(pixel, pix)
    # all lens pixels hit once
    assert np.all(ngal[0] == 5)
    # averaged down
    assert np.allclose(g1[0], pix * 2)
    assert np.allclose(g2[0], mu_2)
    # no variance for g1, since constant
    # for
    assert np.allclose(var_g1[0], 0.0)
    assert np.allclose(var_g2[0], var_2)
    assert np.allclose(source_weight[0], 5)
