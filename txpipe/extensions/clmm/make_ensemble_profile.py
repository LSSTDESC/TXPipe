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
#        ("shear_catalog", ShearCatalog),
        ("fiducial_cosmology", FiducialCosmology),
#        ("shear_tomography_catalog", TomographyCatalog),
#        ("source_photoz_pdfs", PhotozPDFFile),
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

        num_profile_bins = 5
        
        # load cluster catalog as an astropy table
        clusters = self.load_cluster_catalog()
        ncluster = len(clusters)
        
        # load cluster shear catalog using similar astropy table set up as cluster catalog
        cluster_shears_cat = self.load_cluster_shear_catalog()  
        
        # Store the profiles for each cluster
        per_cluster_data = list()

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
     
            # richness = clusters[cluster_index]["richness"]
            # ra_cl = clusters[cluster_index]["ra"]
            # dec_cl = clusters[cluster_index]["dec"]
            # id_cluster = clusters[cluster_index]["id"]

            profile = self.make_clmm_profile(bg_cat, z_cluster, clmm_cosmo, num_profile_bins)

            # We want to append the columns as numpy arrays
            per_cluster_data.append(profile)

        print(len(per_cluster_data))
        print(type(per_cluster_data[0]))
        
        profile_len = num_profile_bins
        profile_colnames = list(profile.keys())
                
        # The root process saves all the data. First it sets up the output
        # file here.
        if self.rank == 0:
            outfile = self.open_output("cluster_profiles")
            # Create space for the catalog                                                                                                                               
            catalog_group = outfile.create_group("catalog")
            catalog_group.create_dataset("cluster_sample_start", shape=(ncluster,), dtype=np.int32)
            catalog_group.create_dataset("cluster_sample_count", shape=(ncluster,), dtype=np.int32)
            catalog_group.create_dataset("cluster_id", shape=(ncluster,), dtype=np.int64)

            
            # and for the profile columns into that catalog
            profile_group = outfile.create_group("profile")
            profile_group.create_dataset("cluster_index", shape=(ncluster, profile_len), dtype=np.int64)
           
            for colname in profile_colnames :
                profile_group.create_dataset(colname, shape=(ncluster,profile_len), dtype=np.float64)
                

        # Now we loop through each cluster and collect all the profile values we calculated
        # from all the different processes.
        start = 0
        
        for i, c in enumerate(clusters):

            if (self.rank == 0) and (i%100 == 0):
                print(f"Collecting data for cluster {i}")

            profiles_to_collect = per_cluster_data[i]
            # If we are running in parallel then collect together
            # the values from all the processes  
            print(self.comm)                                                                                                                                        
            if self.comm is not None:
                profiles_to_collect = self.collect(profiles_to_collect)

            # Only the root process does the writing, so the others just                                                                                
            # go to the next set of clusters.
            if self.rank != 0:
                continue


            # And finally write out all the data from the root process.                                                                                                  
            n = profile_len
            print(f"Created profile length {n} in catalog for cluster {c['id']}")

            catalog_group["cluster_sample_start"][i] = start
            catalog_group["cluster_sample_count"][i] = n
            catalog_group["cluster_id"][i] = c["id"]

            # ISSUE HERE - NOT QUITE THE SAME AS THE SOURCES_SELECT_COMPUTE step, as we have profiles
            print(start)
            profile_group["cluster_index"][start] = i
            for k in profile_colnames : 
                profile_group[k][start] = profiles_to_collect[k]

            start += 1

        if self.rank == 0:
            outfile.close()

                    

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

        
    def collect(self, profiles_to_collect):
        # total number of background objects for t??????  Number of profiles per cluster?

#        collected_profiles = **profiles_to_collect

        profile_colnames = list(collected_profiles.keys())
        num_profile_bins = len(collected_profiles[profile_colnames[0]])
        
        counts = np.array(self.comm.allgather(indices.size))
        total = counts.sum()

        # This collects together all the results from different processes for this cluster
        if self.rank == 0:
            all_profiles = {}

            for colname in profile_colnames :
                all_profiles[colname] = np.empty(num_profile_bins,
                                                 dtype=collected_profiles[colname].dtype)

                self.comm.Gatherv(sendbuf=collected_profile[colname],
                                  recvbuf=(all_profiles[colname], 1))

            for colname in profile_colnames :
                collected_profiles[colname] = all_profiles[colname]

        else:
            for colname in profile_colnames :
                
                self.comm.Gatherv(sendbuf=collected_profiles[colname], recvbuf=(None, 1))

        return collected_profiles


    def make_clmm_profile(self, bg_cat, z_cluster, clmm_cosmo, num_bins):
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
            bins=num_bins,
            cosmo=clmm_cosmo,
            z_lens=z_cluster)


        return {k: np.array(profile[k]) for k in profile.colnames}


