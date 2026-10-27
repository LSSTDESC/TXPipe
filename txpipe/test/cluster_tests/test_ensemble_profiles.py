from ...data_types import FiducialCosmology
from .cluster_mocks import (
    FIDUCIAL_COSMOLOGY,
    make_cluster_profiles,
    G_T,
    Z_SOURCE,
    ZEDGE,
    RICHEDGE,
    N_PER_BIN,
    NBINS_RADIUS,
    COV_KEY,
)
import numpy as np
import tempfile
import pickle
import pytest


def sigma_crit(z_cluster):
    import clmm.cosmology.ccl

    with FiducialCosmology(FIDUCIAL_COSMOLOGY, "r") as f:
        cosmo = clmm.cosmology.ccl.CCLCosmology()
        cosmo.set_be_cosmo(f.to_ccl())
    return cosmo.eval_sigma_crit(z_cluster, Z_SOURCE)


@pytest.mark.parametrize("bin_units", ["mpc", "arcmin"])
@pytest.mark.parametrize("cov_type", ["sample_covariance", "jackknife_covariance", "bootstrap_covariance"])
@pytest.mark.parametrize("delta_sigma", [False, True])
def test_ensemble_profiles(delta_sigma, cov_type, bin_units):
    with tempfile.TemporaryDirectory() as dirname:
        paths = make_cluster_profiles(dirname, delta_sigma, cov_type, bin_units)
        with open(paths["cluster_profiles"], "rb") as f:
            profiles = pickle.load(f)

    nz = len(ZEDGE) - 1
    nr = len(RICHEDGE) - 1
    assert len(profiles) == nz * nr
    for i in range(nz):
        for j in range(nr):
            p = profiles[f"bin_zbin_{i}_richbin_{j}"]
            assert p["n_cl"] == N_PER_BIN
            assert p["profile_type"] == ("delta_sigma" if delta_sigma else "reduced_shear")
            assert p["cluster_bin_edges"]["z_min"] == ZEDGE[i]
            assert p["cluster_bin_edges"]["z_max"] == ZEDGE[i + 1]
            assert p["cluster_bin_edges"]["rich_min"] == RICHEDGE[j]
            assert p["cluster_bin_edges"]["rich_max"] == RICHEDGE[j + 1]

            # The stacked profile recovers the input shear, which is the
            # same in all the radial bins
            ensemble = p["clmm_cluster_ensemble"]
            stacked = ensemble.stacked_data
            assert len(stacked) == NBINS_RADIUS
            expected = G_T
            if delta_sigma:
                expected *= sigma_crit(0.5 * (ZEDGE[i] + ZEDGE[i + 1]))
            assert np.allclose(stacked["tangential_comp"], expected, rtol=1e-3)
            assert np.allclose(stacked["cross_comp"], 0.0, atol=1e-3 * expected)

            cov = ensemble.cov[COV_KEY[cov_type]]
            assert cov.shape == (NBINS_RADIUS, NBINS_RADIUS)
            assert np.all(np.isfinite(cov))
