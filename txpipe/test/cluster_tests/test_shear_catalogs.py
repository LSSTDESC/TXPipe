from ...extensions.cluster_counts import CLClusterShearCatalogs
from ...data_types import FiducialCosmology
from .cluster_mocks import (
    FIDUCIAL_COSMOLOGY,
    PDF_Z,
    PDF_SIGMA,
    write_cluster_catalog,
    write_source_catalogs,
    ring,
    tangential_shear,
    run_stage,
)
import numpy as np
import tempfile
import os
import pytest
import h5py
import mockmpi


# One cluster with sources around it, and one cluster with no sources at all
CLUSTER_RA = [60.0, 80.0]
CLUSTER_DEC = [-30.0, -30.0]
CLUSTER_Z = [0.3, 0.5]
CLUSTER_RICHNESS = [20.0, 20.0]
G_T = 0.05

# Groups of sources around the first cluster.
# Each one is (radius in arcmin, zmode, zmean, ztrue, centre of the PDF, tomographic bin).
# With delta_z = 0.1 a source is behind the cluster if its redshift is above 0.4.
# The "*_only" groups are behind the cluster for only one of the redshift estimates,
# so they show which estimate was used for the cut.
SOURCE_GROUPS = {
    "all_behind": (5.0, 1.0, 0.9, 0.8, 0.7, 0),
    "all_front": (5.0, 0.1, 0.1, 0.1, 0.1, 0),
    "zmode_only": (5.0, 1.0, 0.1, 0.1, 0.1, 0),
    "zmean_only": (5.0, 0.1, 1.0, 0.1, 0.1, 0),
    "ztrue_only": (5.0, 0.1, 0.1, 1.0, 0.1, 0),
    "pdf_only": (5.0, 0.1, 0.1, 0.1, 1.0, 0),
    "far": (20.0, 1.0, 0.9, 0.8, 0.7, 0),
    "not_selected": (5.0, 1.0, 0.9, 0.8, 0.7, -1),
}
N_PER_GROUP = 8
CUT_ONLY_GROUP = {"zmode": "zmode_only", "zmean": "zmean_only", "ztrue": "ztrue_only", "pdf": "pdf_only"}
# redshift of the "all_behind" group for each estimate
ALL_BEHIND_Z = {"zmode": 1.0, "zmean": 0.9, "ztrue": 0.8, "pdf": 0.7}


def make_inputs(dirname, catalog_type):
    """
    Write the fake inputs and return the stage arguments and
    the source indices of each group.
    """
    cluster_file = os.path.join(dirname, "cluster_catalog.hdf5")
    write_cluster_catalog(cluster_file, CLUSTER_RA, CLUSTER_DEC, CLUSTER_Z, CLUSTER_RICHNESS)

    cols = {k: [] for k in ["ra", "dec", "g1", "g2", "zmode", "zmean", "ztrue", "zpdf", "source_bin"]}
    group_index = {}
    start = 0
    for name, (radius, zmode, zmean, ztrue, zpdf, source_bin) in SOURCE_GROUPS.items():
        ra, dec, phi = ring(CLUSTER_RA[0], CLUSTER_DEC[0], radius, N_PER_GROUP)
        g1, g2 = tangential_shear(phi, G_T)
        for k, v in [("ra", ra), ("dec", dec), ("g1", g1), ("g2", g2)]:
            cols[k].append(v)
        for k, v in [("zmode", zmode), ("zmean", zmean), ("ztrue", ztrue), ("zpdf", zpdf), ("source_bin", source_bin)]:
            cols[k].append(np.full(N_PER_GROUP, v))
        group_index[name] = np.arange(start, start + N_PER_GROUP)
        start += N_PER_GROUP
    cols = {k: np.concatenate(v) for k, v in cols.items()}

    # a non-trivial response for metadetect, so we check the calibration is applied
    R = np.array([[0.8, 0.02], [0.01, 0.85]]) if catalog_type == "metadetect" else None
    paths = write_source_catalogs(dirname, catalog_type=catalog_type, R=R, **cols)

    args = dict(
        cluster_catalog=cluster_file,
        fiducial_cosmology=FIDUCIAL_COSMOLOGY,
        cluster_shear_catalogs=os.path.join(dirname, "cluster_shear_catalogs.hdf5"),
        max_radius=3.0,  # Mpc, about 12 arcmin at z = 0.3
        **paths,
    )
    return args, group_index


def read_cluster(filename, cluster_index):
    with h5py.File(filename, "r") as f:
        start = f["catalog/cluster_sample_start"][cluster_index]
        count = f["catalog/cluster_sample_count"][cluster_index]
        index = {k: f[f"index/{k}"][start:start + count] for k in f["index"].keys()}
        profile_type = f["index"].attrs["profile_type"]
    return index, profile_type


def expected_weight(z_source, criterion):
    """
    The Delta-Sigma weight 1/Sigma_crit^2 of the "all_behind" sources
    """
    import clmm
    import clmm.cosmology.ccl

    with FiducialCosmology(FIDUCIAL_COSMOLOGY, "r") as f:
        cosmo = clmm.cosmology.ccl.CCLCosmology()
        cosmo.set_be_cosmo(f.to_ccl())
    if criterion == "pdf":
        pdf = np.exp(-0.5 * ((PDF_Z - z_source) / PDF_SIGMA) ** 2)
        # CLMM fails with a single PDF, so we give it two copies
        sigma_c = clmm.theory.compute_critical_surface_density_eff(
            cosmo=cosmo, z_cluster=CLUSTER_Z[0], pzbins=PDF_Z, pzpdf=np.array([pdf, pdf])
        )[0]
    else:
        sigma_c = cosmo.eval_sigma_crit(CLUSTER_Z[0], z_source)
    return 1.0 / sigma_c**2


@pytest.mark.parametrize("catalog_type", ["simple", "metadetect"])
@pytest.mark.parametrize("delta_sigma", [False, True])
@pytest.mark.parametrize("weight_criterion", ["zmode", "zmean", "ztrue", "pdf"])
@pytest.mark.parametrize("cut_criterion", ["zmode", "zmean", "ztrue", "pdf"])
def test_source_selection(cut_criterion, weight_criterion, delta_sigma, catalog_type):
    with tempfile.TemporaryDirectory() as dirname:
        args, group_index = make_inputs(dirname, catalog_type)
        run_stage(
            CLClusterShearCatalogs,
            redshift_cut_criterion=cut_criterion,
            redshift_weight_criterion=weight_criterion,
            delta_sigma=delta_sigma,
            **args,
        )
        index, profile_type = read_cluster(args["cluster_shear_catalogs"], 0)
        empty_index, _ = read_cluster(args["cluster_shear_catalogs"], 1)

    assert profile_type == ("delta_sigma" if delta_sigma else "reduced_shear")

    # Only the sources that are close enough, selected for weak lensing,
    # and behind the cluster according to the cut criterion are kept.
    expected = np.concatenate([group_index["all_behind"], group_index[CUT_ONLY_GROUP[cut_criterion]]])
    assert np.array_equal(index["source_index"], np.sort(expected))
    assert np.all(index["cluster_id"] == 0)
    assert np.all(index["distance_arcmin"] < 5.01)

    # The other cluster has no sources
    assert empty_index["source_index"].size == 0

    # Check the values for the sources that are behind the cluster
    # for every estimate.
    behind = np.isin(index["source_index"], group_index["all_behind"])
    tangential = index["tangential_comp"][behind]
    cross = index["cross_comp"][behind]
    weight = index["weight"][behind]

    if delta_sigma:
        # the weight is 1 / Sigma_crit^2 computed with the weight criterion
        # and the tangential component is Sigma_crit * g_t
        w = expected_weight(ALL_BEHIND_Z[weight_criterion], weight_criterion)
        assert np.allclose(weight, w, rtol=1e-3)
        assert np.allclose(tangential * np.sqrt(weight), G_T, rtol=1e-3)
        assert np.allclose(cross * np.sqrt(weight), 0.0, atol=1e-4)
    else:
        assert np.allclose(weight, 1.0)
        assert np.allclose(tangential, G_T, rtol=1e-3)
        assert np.allclose(cross, 0.0, atol=1e-4)

    # the calibrated shears are saved too
    g = np.hypot(index["g1"][behind], index["g2"][behind])
    assert np.allclose(g, G_T, rtol=1e-3)


@pytest.mark.parametrize("use_radius,max_angle,expected_groups", [
    (True, 30.0, ["all_behind"]),  # max_radius = 3 Mpc is about 12 arcmin at z = 0.3
    (False, 30.0, ["all_behind", "far"]),
    (False, 10.0, ["all_behind"]),
])
def test_search_radius(use_radius, max_angle, expected_groups):
    with tempfile.TemporaryDirectory() as dirname:
        args, group_index = make_inputs(dirname, "simple")
        run_stage(
            CLClusterShearCatalogs,
            redshift_cut_criterion="zmode",
            redshift_weight_criterion="zmode",
            use_radius=use_radius,
            max_angle=max_angle,
            **args,
        )
        index, _ = read_cluster(args["cluster_shear_catalogs"], 0)
        with h5py.File(args["cluster_shear_catalogs"], "r") as f:
            theta_max = f["catalog/cluster_theta_max_arcmin"][0]

    # the zmode_only group is also behind the cluster for zmode
    expected = np.concatenate([group_index[g] for g in expected_groups + ["zmode_only"]])
    assert np.array_equal(index["source_index"], np.sort(expected))
    if use_radius:
        assert 5.0 < theta_max < 20.0
    else:
        assert np.isclose(theta_max, max_angle)


def core_collect_empty(comm):
    # A cluster with no sources on any process. The run method
    # unpacks seven arrays from collect, so it must return seven.
    stage = CLClusterShearCatalogs(
        dict(config={}, cluster_catalog="", fiducial_cosmology="", shear_catalog="",
             shear_tomography_catalog="", source_photoz_pdfs=""),
        comm=comm,
    )
    empty = np.zeros(0)
    out = stage.collect(np.zeros(0, dtype=int), empty, empty, empty, empty, empty, empty)
    assert len(out) == 7
    for x in out:
        assert x.size == 0


def test_collect_empty():
    # mockmpi does not implement the Gatherv used for non-empty clusters,
    # so we only test the case with no sources here
    mockmpi.mock_mpiexec(2, core_collect_empty)
