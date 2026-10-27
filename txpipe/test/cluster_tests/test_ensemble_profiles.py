from ...extensions.cluster_counts import (
    CLClusterBinningRedshiftRichness,
    CLClusterShearCatalogs,
    CLClusterEnsembleProfiles,
)
from ...data_types import FiducialCosmology
from .cluster_mocks import (
    FIDUCIAL_COSMOLOGY,
    write_cluster_catalog,
    write_source_catalogs,
    ring,
    tangential_shear,
    run_stage,
)
import numpy as np
import tempfile
import os
import pickle
import pytest


G_T = 0.05
Z_SOURCE = 1.5
ZEDGE = [0.2, 0.4, 0.6]
RICHEDGE = [5.0, 10.0, 20.0]
N_PER_BIN = 3
NBINS_RADIUS = 5
COV_KEY = {"sample_covariance": "tan_sc", "jackknife_covariance": "tan_jk", "bootstrap_covariance": "tan_bs"}
# radial range for each unit, inside the rings of sources for all the clusters
RADIAL_RANGE = {"mpc": (0.6, 2.5), "arcmin": (2.0, 15.0)}


def make_cluster_profiles(dirname, delta_sigma=False, cov_type="sample_covariance", bin_units="mpc",
                          n_per_bin=None):
    """
    Run the binning, shear catalog and ensemble profile stages on clusters
    with a pure tangential shear around them, and return the paths of all
    the files. n_per_bin gives the number of clusters in each
    (redshift, richness) bin.
    """
    if n_per_bin is None:
        n_per_bin = {(i, j): N_PER_BIN for i in range(len(ZEDGE) - 1) for j in range(len(RICHEDGE) - 1)}

    # clusters in the middle of each bin, a few degrees apart
    cluster_z = []
    cluster_richness = []
    for (i, j), n in n_per_bin.items():
        cluster_z += [0.5 * (ZEDGE[i] + ZEDGE[i + 1])] * n
        cluster_richness += [0.5 * (RICHEDGE[j] + RICHEDGE[j + 1])] * n
    ncluster = len(cluster_z)
    cluster_ra = 30.0 + 5.0 * np.arange(ncluster)
    cluster_dec = np.where(np.arange(ncluster) % 2 == 0, -30.0, -35.0)

    paths = {
        "cluster_catalog": os.path.join(dirname, "cluster_catalog.hdf5"),
        "cluster_catalog_tomography": os.path.join(dirname, "cluster_catalog_tomography.hdf5"),
        "cluster_shear_catalogs": os.path.join(dirname, "cluster_shear_catalogs.hdf5"),
        "cluster_profiles": os.path.join(dirname, "cluster_profiles.pkl"),
        "fiducial_cosmology": FIDUCIAL_COSMOLOGY,
    }
    write_cluster_catalog(paths["cluster_catalog"], cluster_ra, cluster_dec, cluster_z, cluster_richness)

    # rings of sources from 1.5 to 20 arcmin around every cluster
    ra = []
    dec = []
    g1 = []
    g2 = []
    for c in range(ncluster):
        for radius in np.geomspace(1.5, 20.0, 20):
            r, d, phi = ring(cluster_ra[c], cluster_dec[c], radius, 16)
            e1, e2 = tangential_shear(phi, G_T)
            ra.append(r)
            dec.append(d)
            g1.append(e1)
            g2.append(e2)
    ra = np.concatenate(ra)
    nsource = ra.size
    z = np.full(nsource, Z_SOURCE)
    paths.update(write_source_catalogs(
        dirname, ra, np.concatenate(dec), np.concatenate(g1), np.concatenate(g2),
        zmode=z, zmean=z, ztrue=z, zpdf=z, source_bin=np.zeros(nsource, dtype=int),
    ))

    run_stage(
        CLClusterBinningRedshiftRichness,
        cluster_catalog=paths["cluster_catalog"],
        cluster_catalog_tomography=paths["cluster_catalog_tomography"],
        zedge=ZEDGE,
        richedge=RICHEDGE,
    )
    run_stage(
        CLClusterShearCatalogs,
        cluster_catalog=paths["cluster_catalog"],
        fiducial_cosmology=paths["fiducial_cosmology"],
        shear_catalog=paths["shear_catalog"],
        shear_tomography_catalog=paths["shear_tomography_catalog"],
        source_photoz_pdfs=paths["source_photoz_pdfs"],
        cluster_shear_catalogs=paths["cluster_shear_catalogs"],
        use_radius=False,
        max_angle=25.0,
        delta_sigma=delta_sigma,
    )
    r_min, r_max = RADIAL_RANGE[bin_units]
    run_stage(
        CLClusterEnsembleProfiles,
        cluster_catalog_tomography=paths["cluster_catalog_tomography"],
        fiducial_cosmology=paths["fiducial_cosmology"],
        cluster_shear_catalogs=paths["cluster_shear_catalogs"],
        cluster_profiles=paths["cluster_profiles"],
        r_min=r_min,
        r_max=r_max,
        nbins=NBINS_RADIUS,
        bin_units=bin_units,
        cov_type=cov_type,
        bootstrap_nboot=10,
    )
    return paths


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
