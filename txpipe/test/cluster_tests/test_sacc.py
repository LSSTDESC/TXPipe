from ...extensions.cluster_counts import CLClusterSACC
from .cluster_mocks import run_stage
from .test_ensemble_profiles import make_cluster_profiles, NBINS_RADIUS, N_PER_BIN, ZEDGE, RICHEDGE, COV_KEY
import numpy as np
import tempfile
import os
import pytest


@pytest.mark.parametrize("cov_type", ["sample_covariance", "jackknife_covariance", "bootstrap_covariance"])
@pytest.mark.parametrize("delta_sigma", [False, True])
def test_cluster_sacc(delta_sigma, cov_type):
    import sacc

    with tempfile.TemporaryDirectory() as dirname:
        paths = make_cluster_profiles(dirname, delta_sigma, cov_type)
        sacc_file = os.path.join(dirname, "cluster_sacc_catalog.sacc")
        run_stage(
            CLClusterSACC,
            cluster_profiles=paths["cluster_profiles"],
            cluster_sacc_catalog=sacc_file,
            cov_component=COV_KEY[cov_type],
            survey_name="mock_survey",
            area=10.0,
        )
        s = sacc.Sacc.load_fits(sacc_file)

    nz = len(ZEDGE) - 1
    nr = len(RICHEDGE) - 1

    # one count per (redshift, richness) bin
    counts = s.get_data_points(sacc.standard_types.cluster_counts)
    assert len(counts) == nz * nr
    for d in counts:
        assert d.value == N_PER_BIN

    # one shear point per (redshift, richness, radius) bin, of the right type
    if delta_sigma:
        shear_type = sacc.standard_types.cluster_delta_sigma
        other_type = sacc.standard_types.cluster_shear
    else:
        shear_type = sacc.standard_types.cluster_shear
        other_type = sacc.standard_types.cluster_delta_sigma
    assert len(s.get_data_points(shear_type)) == nz * nr * NBINS_RADIUS
    assert len(s.get_data_points(other_type)) == 0

    # tracers for the bins and the survey
    assert s.tracers["mock_survey"].sky_area == 10.0
    for i in range(nz):
        t = s.tracers[f"bin_z_{i}"]
        assert (t.lower, t.upper) == (ZEDGE[i], ZEDGE[i + 1])
    for j in range(nr):
        t = s.tracers[f"bin_rich_{j}"]
        assert np.isclose(t.lower, np.log10(RICHEDGE[j]))
        assert np.isclose(t.upper, np.log10(RICHEDGE[j + 1]))

    # block diagonal covariance with the Poisson counts first
    n = nz * nr * (1 + NBINS_RADIUS)
    cov = s.covariance.dense
    assert cov.shape == (n, n)
    counts_index = s.indices(sacc.standard_types.cluster_counts)
    assert np.allclose(np.diag(cov)[counts_index], N_PER_BIN)


def test_cluster_sacc_bin_without_ensemble():
    # A bin with a single cluster has no ensemble profile, and the stage
    # should say which bin is the problem
    n_per_bin = {(0, 0): 3, (0, 1): 3, (1, 0): 3, (1, 1): 1}
    with tempfile.TemporaryDirectory() as dirname:
        paths = make_cluster_profiles(dirname, n_per_bin=n_per_bin)
        with pytest.raises(ValueError, match="bin_zbin_1_richbin_1"):
            run_stage(
                CLClusterSACC,
                cluster_profiles=paths["cluster_profiles"],
                cluster_sacc_catalog=os.path.join(dirname, "cluster_sacc_catalog.sacc"),
                cov_component="tan_sc",
            )
