import os
import gc
import numpy as np
from ...base_stage import PipelineStage
from ...data_types import ShearCatalog, HDFFile, PhotozPDFFile, FiducialCosmology, TomographyCatalog, PickleFile
from ...utils.calibrators import Calibrator
from collections import defaultdict
import yaml
import ceci

class CLClusterEnsembleProfiles(PipelineStage):
    name = "CLClusterEnsembleProfiles"
    inputs = [
        ("cluster_catalog", HDFFile),
        ("shear_catalog", ShearCatalog),
        ("fiducial_cosmology", FiducialCosmology),
        ("shear_tomography_catalog", TomographyCatalog),
        ("source_photoz_pdfs", PhotozPDFFile),
        ("cluster_shear_catalogs", HDFFile)
    ]

    outputs = [
        ("cluster_ensemble", PickleFile),
    ]

    config_options = {
        "chunk_rows": 100_000,  # rows to read at once from source cat
        "max_radius": 10.0,  # Mpc
        "delta_z": 0.1,
        "redshift_criterion": "mean",  # might also need PDF
        "subtract_mean_shear": True,
    }

    def run(self):
        import sklearn.neighbors
        import astropy
        import h5py
        import clmm
        import clmm.cosmology.ccl


    def collect(self, indices, weights, distances):
        # total number of background objects for t

        counts = np.array(self.comm.allgather(indices.size))
        total = counts.sum()

        # Early exit if nothing here
        if total == 0:
            indices = np.zeros(0, dtype=int)
            weights = np.zeros(0)
            distances = np.zeros(0)
            return indices, weights, distances

        # This collects together all the results from different processes for this cluster
        if self.rank == 0:
            all_indices = np.empty(total, dtype=indices.dtype)
            all_weights = np.empty(total, dtype=weights.dtype)
            all_distances = np.empty(total, dtype=distances.dtype)
            self.comm.Gatherv(sendbuf=distances, recvbuf=(all_distances, counts))
            self.comm.Gatherv(sendbuf=weights, recvbuf=(all_weights, counts))
            self.comm.Gatherv(sendbuf=indices, recvbuf=(all_indices, counts))
            indices = all_indices
            weights = all_weights
            distances = all_distances
        else:
            self.comm.Gatherv(sendbuf=distances, recvbuf=(None, counts))
            self.comm.Gatherv(sendbuf=weights, recvbuf=(None, counts))
            self.comm.Gatherv(sendbuf=indices, recvbuf=(None, counts))

        return indices, weights, distances


    def compute_weights(self, clmm_cosmo, data, index, z_cluster):
        import clmm

        # Depending on whether we are using the PDF or not, choose
        # some keywords to give to compute_galaxy_weights
        if self.config["redshift_criterion"] == "pdf":
            # We need the z and PDF(z) arrays in this case
            pdf_z = data["pdf_z"]
            pdf_pz = data["pdf_pz"][index]
            redshift_keywords = {
                "pzpdf":pdf_pz,
                "pzbins":pdf_z,
                "use_pdz":True
            }
        else:
            # point-estimated redshift
            z_source = data["redshift"][index]
            redshift_keywords = {
                "z_source":z_source,
                "use_pdz":False
            }

        weight = clmm.dataops.compute_galaxy_weights(
            z_cluster,
            clmm_cosmo,
            is_deltasigma=True,
            use_shape_noise=True,
            use_shape_error=False,
            validate_input=True,
            shape_component1=data["g1"][index],
            shape_component2=data["g2"][index],
            **redshift_keywords
        )

        return weight

