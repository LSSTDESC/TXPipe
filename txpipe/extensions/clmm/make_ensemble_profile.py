import os
import gc
import numpy as np
from ...base_stage import PipelineStage
from .sources_select_compute import CLClusterShearCatalogs
from ...data_types import ShearCatalog, HDFFile, PhotozPDFFile, FiducialCosmology, TomographyCatalog, PickleFile
from ...utils.calibrators import Calibrator
from collections import defaultdict
import yaml
import ceci

class CLClusterEnsembleProfiles(CLClusterShearCatalogs):
    name = "CLClusterEnsembleProfiles"
    inputs = [
        ("cluster_catalog", HDFFile),
        ("shear_catalog", ShearCatalog),
        ("fiducial_cosmology", FiducialCosmology),
        ("shear_tomography_catalog", TomographyCatalog),
        ("source_photoz_pdfs", PhotozPDFFile),
        ("cluster_shear_catalogs", HDFFile),
    ]

    outputs = [
        ("cluster_profiles", HDFFile),
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

        # load cluster catalog as an astropy table
        clusters = self.load_cluster_catalog()
        ncluster = len(clusters)
        
        # load cluster shear catalog using similar astropy table set up as cluster catalog
        cluster_shears_cat = self.load_cluster_shear_catalog()
        
        
        # Store the profiles for each cluster
        per_cluster_data = [list() for i in range(ncluster)]

        with self.open_input("fiducial_cosmology", wrapper=True) as f:
            ccl_cosmo = f.to_ccl()
            clmm_cosmo = clmm.cosmology.ccl.CCLCosmology()
            clmm_cosmo.set_be_cosmo(ccl_cosmo)


        # Loop through clusters and calculate the profiles
        for cluster_index in range(ncluster) :

            # Select subset of background shear information for this particular cluster
            mask = (cluster_shears_cat["cluster_index"] == cluster_index)
            bg_cat = cluster_shears_cat[mask]
                            
            z_cluster = clusters[cluster_index]["redshift"]
            profile = self.make_clmm_profile(bg_cat, z_cluster, clmm_cosmo)


            per_cluster_data[cluster_index].append(profile)
            
    
        # The root process saves all the data. First it setps up the output
        # file here.
        if self.rank == 0:
            outfile = self.open_output("cluster_profiles")
            # Create space for the catalog
            catalog_group = outfile.create_group("catalog")
            catalog_group.create_dataset("cluster_sample_start", shape=(ncluster,), dtype=clmm.GCData)
#             catalog_group.create_dataset("cluster_sample_count", shape=(ncluster,), dtype=np.int32)
#            catalog_group.create_dataset("cluster_id", shape=(ncluster,), dtype=np.int64)
#             catalog_group.create_dataset("cluster_theta_max_arcmin", shape=(ncluster,), dtype=np.float64)
#             # and for the index into that catalog
#             index_group = outfile.create_group("index")
#             index_group.create_dataset("cluster_index", shape=(total_count,), dtype=np.int64)
#             index_group.create_dataset("source_index", shape=(total_count,), dtype=np.int64)
#             index_group.create_dataset("weight", shape=(total_count,), dtype=np.float64)
#             index_group.create_dataset("tangential_comp", shape=(total_count,), dtype=np.float64)
#             index_group.create_dataset("cross_comp", shape=(total_count,), dtype=np.float64)
#             index_group.create_dataset("distance_arcmin", shape=(total_count,), dtype=np.float64)

            
    def load_cluster_shear_catalog(self) :
        from astropy.table import Table

        with self.open_input("cluster_shear_catalogs") as f:
            g = f["index/"]
            cluster_index = g['cluster_index'][:]
            tangential_comp = g['tangential_comp'][:]
            cross_comp = g['cross_comp'][:]
            source_index = g['source_index'][:]
            weight = g['weight'][:]
            distance_arcmin = g['distance_arcmin'][:]

        print(len(cluster_index), len(tangential_comp), len(source_index))

        return Table({"cluster_index": cluster_index, "tangential_comp_clmm": tangential_comp,
                      "cross_comp_clmm": cross_comp, "source_index": source_index,
                      "weight_clmm": weight, "distance_arcmin": distance_arcmin})

        
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


    def make_clmm_profile(self, bg_cat, z_cluster, clmm_cosmo):
        import clmm

        tangential_comp = bg_cat["tangential_comp_clmm"]
        cross_comp = bg_cat["cross_comp_clmm"]
        weights = bg_cat["weight_clmm"]
        angsep = bg_cat["distance_arcmin"]

        profile = clmm.dataops.make_radial_profile(
            [tangential_comp, cross_comp],
            weights=weights,
            angsep=angsep,
            angsep_units="arcmin",
            bin_units="Mpc",
            bins=10,
            cosmo=clmm_cosmo,
            z_lens=z_cluster)


        return profile


