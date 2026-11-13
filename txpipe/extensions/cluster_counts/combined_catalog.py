import os
import pickle
import numpy as np
from ...data_types import HDFFile, ShearCatalog


class CombinedClusterCatalog:
    """
    The CCC class collects together:
    - a cluster catalog
    - the sources of each cluster from CLClusterEnsembleProfiles (cluster_source_quantities),
      with their calibrated shears, weights, tangential and cross components and, for
      delta sigma, Sigma_crit
    - optionally the profiles from CLClusterEnsembleProfiles (cluster_profiles)
    - optionally the shear catalog, for the true shears of mock catalogs

    It lets us pull together all the information associated with a cluster, for checks
    and plots. It only reads the files made by the stages and is not used by them.
    Cluster i is the cluster in row i of the catalog group of cluster_source_quantities.
    """
    def __init__(self, cluster_catalog, cluster_source_quantities, cluster_profiles=None, shear_catalog=None):
        """
        Initialise the CCC object by opening all the files.
        """
        self.cluster_catalog = HDFFile(cluster_catalog, "r").file
        self.source_quantities = HDFFile(cluster_source_quantities, "r").file
        self.cluster_profiles = cluster_profiles
        self.shear_cat = None if shear_catalog is None else ShearCatalog(shear_catalog, "r")
        self.cluster_cat_cols = list(self.cluster_catalog["clusters"].keys())
        self.cluster_row = {cluster_id: i for i, cluster_id in enumerate(self.cluster_catalog["clusters/cluster_id"][:])}
        self.coordinate_system = self.source_quantities["provenance"].attrs["config/coordinate_system"]
        self.cluster_ids = self.source_quantities["catalog/cluster_id"][:]
        self.sample_start = self.source_quantities["catalog/cluster_sample_start"][:]
        self.sample_count = self.source_quantities["catalog/cluster_sample_count"][:]
        self.ncluster = self.cluster_ids.size

    @classmethod
    def from_pipeline_file(cls, pipeline_file, run_dir='.'):
        """
        Factory method to build a CombinedClusterCatalog from a pipeline file.

        Get the names of all the files that we need to load by looking into the pipeline file.

        Parameters
        ----------
        pipeline_file : str
            Path to the pipeline file
        run_dir : str
            Path to the directory where the pipeline was run (option, default='.')

        Returns
        -------
        CombinedClusterCatalog
            An instance of the CombinedClusterCatalog class

        """
        import ceci

        pipe_config = ceci.Pipeline.build_config(
            pipeline_file,
            dry_run=True
        )

        pipeline = ceci.Pipeline.create(pipe_config)

        paths = pipeline.overall_inputs.copy()
        for stage in pipeline.stages:
            paths.update(stage.find_outputs(pipe_config["output_dir"]))

        files = {}
        for tag in ["cluster_catalog", "cluster_source_quantities", "cluster_profiles", "shear_catalog"]:
            if tag not in paths:
                raise ValueError(f"This pipeline did not generate or ingest {tag} needed for cluster WL")
            path = os.path.join(run_dir, paths[tag])
            if not os.path.exists(path):
                raise ValueError(f"File {path} does not exist - pipeline may not have run")
            files[tag] = path

        return cls(**files)

    def get_cluster_info(self, cluster_index):
        """
        All the columns of the cluster catalog for cluster cluster_index.
        """
        row = self.cluster_row[self.cluster_ids[cluster_index]]
        return {k: self.cluster_catalog[f'clusters/{k}'][row] for k in self.cluster_cat_cols}

    def get_background_shear_catalog(self, cluster_index):
        """
        The sources of cluster cluster_index, with the quantities computed for them by
        CLClusterEnsembleProfiles and, if the shear catalog has them, their true shears.
        """
        import clmm

        start = self.sample_start[cluster_index]
        end = start + self.sample_count[cluster_index]
        index_group = self.source_quantities["index"]
        cat = {col: index_group[col][start:end] for col in index_group.keys()}

        # The true shears are in the main catalog of the shear catalog (shear/ns for
        # metadetect). The rows of the sources are sorted, so we read them in contiguous
        # blocks, which is much faster than an HDF5 point selection.
        rows = cat["source_index"]
        group = None if self.shear_cat is None else self.shear_cat.file[self.shear_cat.get_primary_catalog_group()]
        if rows.size > 0 and group is not None and "true_g1" in group:
            blocks = np.split(rows, np.flatnonzero(np.diff(rows) > 100_000) + 1)
            for col in ["true_g1", "true_g2"]:
                cat[col] = np.concatenate([group[col][b[0]:b[-1] + 1][b - b[0]] for b in blocks])

        return clmm.GCData(data=cat, meta={"coordinate_system": self.coordinate_system})

    def profiles(self):
        """
        The output of CLClusterEnsembleProfiles: for each cluster bin, the CLMM ClusterEnsemble
        with the profile of each cluster, the stacked profile and its covariance.
        """
        with open(self.cluster_profiles, "rb") as f:
            return pickle.load(f)
