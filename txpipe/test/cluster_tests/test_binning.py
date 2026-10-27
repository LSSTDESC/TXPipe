from ...extensions.cluster_counts import CLClusterBinningRedshiftRichness
from .cluster_mocks import write_cluster_catalog, run_stage
import numpy as np
import tempfile
import os
import pytest
import h5py


# clusters inside, between and outside the bin edges
REDSHIFT = np.array([0.1, 0.25, 0.3, 0.45, 0.55, 0.7, 0.9, 1.2])
RICHNESS = np.array([30., 7., 12., 25., 50., 6., 15., 8.])


@pytest.mark.parametrize("zedge,richedge", [
    ([0.2, 0.4, 0.6, 0.8, 1.0], [5., 10., 20.]),
    ([0.2, 0.5, 1.0], [5., 10., 100.]),
])
def test_cluster_binning(zedge, richedge):
    n = REDSHIFT.size
    ra = np.linspace(50.0, 60.0, n)
    dec = np.full(n, -30.0)

    with tempfile.TemporaryDirectory() as dirname:
        cluster_file = os.path.join(dirname, "cluster_catalog.hdf5")
        output_file = os.path.join(dirname, "cluster_catalog_tomography.hdf5")
        write_cluster_catalog(cluster_file, ra, dec, REDSHIFT, RICHNESS)

        run_stage(
            CLClusterBinningRedshiftRichness,
            cluster_catalog=cluster_file,
            cluster_catalog_tomography=output_file,
            zedge=zedge,
            richedge=richedge,
        )

        nz = len(zedge) - 1
        nr = len(richedge) - 1
        with h5py.File(output_file, "r") as f:
            g = f["cluster_bin"]
            assert g.attrs["nz"] == nz
            assert g.attrs["nr"] == nr
            assert len(g.keys()) == nz * nr

            for i in range(nz):
                for j in range(nr):
                    b = g[f"bin_zbin_{i}_richbin_{j}"]
                    assert b.attrs["z_min"] == zedge[i]
                    assert b.attrs["z_max"] == zedge[i + 1]
                    assert b.attrs["rich_min"] == richedge[j]
                    assert b.attrs["rich_max"] == richedge[j + 1]

                    # the clusters that should be in this bin. Clusters outside
                    # all the edges should not end up in any bin.
                    expected = np.where(
                        (REDSHIFT >= zedge[i]) & (REDSHIFT < zedge[i + 1])
                        & (RICHNESS >= richedge[j]) & (RICHNESS < richedge[j + 1])
                    )[0]
                    assert np.array_equal(b["cluster_id"][:], expected)
                    assert np.allclose(b["redshift"][:], REDSHIFT[expected])
                    assert np.allclose(b["richness"][:], RICHNESS[expected])
