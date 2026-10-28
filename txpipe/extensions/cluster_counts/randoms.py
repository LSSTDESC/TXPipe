"""
Make a random catalog for the clusters.

Each random is a cluster row drawn (with replacement) from the cluster catalog,
given a new position drawn uniformly over the footprint. The randoms then get an
effective richness, weight = richness / scaleval, and only those above a cutoff
richness are kept, which acts like a loose detection mask
(a la https://arxiv.org/abs/1601.00621).

The output has the same "clusters" layout as cluster_catalog, so
CLClusterBinningRedshiftRichness can bin it with an alias. For DP2 the footprint
is the shear footprint, in the data registry as "rubin_dp2_shear_footprint".
"""

import numpy as np
from ...base_stage import PipelineStage
from ...data_types import DataFile, HDFFile
from ceci.config import StageParameter


class CLClusterRandoms(PipelineStage):
    """Make cluster randoms over a healsparse footprint, in the cluster_catalog format."""

    name = "CLClusterRandoms"
    parallel = False
    inputs = [
        ("cluster_catalog", HDFFile),
        ("cluster_footprint", DataFile),  # healsparse .hsp map
    ]
    outputs = [("cluster_random_catalog", HDFFile)]

    config_options = {
        "randoms_per_cluster": StageParameter(int, 100, msg="Number of randoms drawn per cluster, before the richness cut"),
        "richness_cutoff": StageParameter(float, 20.0, msg="Keep randoms with richness / scaleval above this"),
        "seed": StageParameter(int, 43, msg="Random seed"),
    }

    def run(self):
        import healsparse

        with self.open_input("cluster_catalog") as f:
            g = f["clusters"]
            clusters = {col: g[col][:] for col in g.keys()}
        n_clusters = clusters["ra"].size

        footprint = healsparse.HealSparseMap.read(self.get_input("cluster_footprint"))
        print(f"Read {n_clusters} clusters and a footprint with nside {footprint.nside_sparse}")

        # Resample whole cluster rows, so the randoms carry every column
        # CLClusterBinningRedshiftRichness reads, then move them to random positions.
        # A redshift mask (zmax per pixel) would go here as a rejection step; for now
        # every draw is kept.
        rng = np.random.default_rng(self.config["seed"])
        size = n_clusters * self.config["randoms_per_cluster"]
        idx = rng.integers(0, n_clusters, size)
        randoms = {col: data[idx] for col, data in clusters.items()}
        randoms["ra"], randoms["dec"] = healsparse.make_uniform_randoms(footprint, size, rng=rng)

        # Effective richness cut, only meaningful if scaleval != 1 so keeping for now
        randoms["weight"] = randoms["richness"] / randoms["scaleval"]
        keep = randoms["weight"] > self.config["richness_cutoff"]
        print(f"Kept {keep.sum()} of {size} randoms with richness / scaleval > {self.config['richness_cutoff']}")

        with self.open_output("cluster_random_catalog") as f:
            g = f.create_group("clusters")
            for col, data in randoms.items():
                g.create_dataset(col, data=data[keep])
