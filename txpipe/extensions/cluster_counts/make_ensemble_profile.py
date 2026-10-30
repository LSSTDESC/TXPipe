import os
import gc
import numpy as np
from ...base_stage import PipelineStage
from .sources_select_compute import CLClusterShearCatalogs, CombinedClusterCatalog
from ...data_types import ShearCatalog, HDFFile, PhotozPDFFile, FiducialCosmology, TomographyCatalog, PickleFile
from ...shear_calibration import Calibrator
from .utils import iterate_source_catalog
from collections import defaultdict
import yaml
import ceci
import pickle

class CLClusterEnsembleProfiles(CLClusterShearCatalogs):
    name = "CLClusterEnsembleProfiles"
    inputs = [
        #("cluster_catalog", HDFFile),
        ("shear_catalog", ShearCatalog),
        ("cluster_catalog_tomography", HDFFile), # TO TEST
        ("fiducial_cosmology", FiducialCosmology),
        ("cluster_shear_selection", HDFFile),
        ("source_photoz_pdfs", PhotozPDFFile),
    ]

    outputs = [
        ("cluster_profiles",  PickleFile),
    ]

    config_options = {
        #radial binning intervals
        "r_min" : 0.2, #in Mpc or arcmin
        "r_max" : 3.0, #in Mpc or arcmin
        "nbins" : 5, # number of bins
        #type of profile
        "magnification_profile" : False,
        "bin_units": "mpc", # options are mpc or arcmin
        "cov_type" : "jackknife_covariance", #sample_covariance, jackknife_covariance, bootstrap_covariance
        "jackknife_nside": 32,
        "bootstrap_nboot": 100,
        #coordinate_system for shear
        #"coordinate_system" : 'euclidean' #Must be either 'celestial' or 'euclidean'
        "coordinate_system": "celestial",
        "use_shape_noise": False,
        "use_true_shear": False,
        "delta_sigma": True, # If True then we compute delta-sigma, otherwise we compute gamma
        "redshift_weight_criterion": "zmode",  # pdf / mean / true / None
        "chunk_rows": 100_000,  # rows to read at once from source cat
    }

    def run(self):
        import sklearn.neighbors
        import astropy
        import h5py
        import clmm
        import clmm.cosmology.ccl 
        
        bin_min = self.config["r_min"]
        bin_max = self.config["r_max"]
        self.distance_bins = clmm.dataops.make_bins(bin_min, bin_max, nbins=self.config["nbins"], method="evenlog10width")

        print (self.distance_bins) 
        self.coordinate_system = self.config["coordinate_system"]
        print (self.coordinate_system) 

        
        with self.open_input("fiducial_cosmology", wrapper=True) as f:
            ccl_cosmo = f.to_ccl()
            self.clmm_cosmo = clmm.cosmology.ccl.CCLCosmology()
            self.clmm_cosmo.set_be_cosmo(ccl_cosmo)
                 
        clusters_in_bins = self.load_cluster_catalog_tomography_group() 
        self.add_source_data(clusters_in_bins)
        cluster_stack_dict = self.make_ensenble(clusters_in_bins) 
        pickle.dump(cluster_stack_dict, open(self.get_output("cluster_profiles"), 'wb'))


        return tab, meta_coord_sys, profile_type

    def add_source_data(self, clusters_in_bins):

        import sklearn.neighbors
        import astropy
        import h5py
        import clmm
        import clmm.cosmology.ccl

        print("Adding source data")

        with self.open_input("fiducial_cosmology", wrapper=True) as f:
            ccl_cosmo = f.to_ccl()
            clmm_cosmo = clmm.cosmology.ccl.CCLCosmology()
            clmm_cosmo.set_be_cosmo(ccl_cosmo)

        print("    Get cluster-galaxy pairs")
        # get cluster-galaxy pairs
        with self.open_input("cluster_shear_selection") as f:
            #catalog_group = outfile.create_group("catalog")
            #catalog_group.create_dataset("cluster_id", shape=(ncluster,), dtype=np.int64)
            #catalog_group.create_dataset("cluster_theta_max_arcmin", shape=(ncluster,), dtype=np.float64)
        
            # sort by cl_id to optimize the assignment
            order = np.argsort(f["index/cluster_id"][:])
            cl_ids = f["index/cluster_id"][:][order]
            src_inds = f["index/source_index"][:][order]
            dist_amin = f["index/distance_arcmin"][:][order]

            unique_cl_ids, num_src_in_cl = np.unique(cl_ids, return_counts=True)
            split_pos = num_src_in_cl.cumsum()

            cluster_source_index = dict(list(zip(unique_cl_ids, np.split(src_inds, split_pos)[:-1])))
            cluster_source_dist = dict(list(zip(unique_cl_ids, np.split(dist_amin, split_pos)[:-1])))

        # sort gal_index just to be sure:
        _sort_order = {key: np.argsort(value) for key, value in cluster_source_index.items()}
        cluster_source_index = {key: value[_sort_order[key]] for key, value in cluster_source_index.items()}
        cluster_source_dist = {key: value[_sort_order[key]] for key, value in cluster_source_dist.items()}

        # add list to fill with sources
        clusters_list_of_source_lists = {
            bin_key: [[] for cluster in cluster_list]
            for bin_key, cluster_list in clusters_in_bins.items()
        }

        print("    Loop through the source catalog")
        # Loop through the source catalog.
        for s, e, data in iterate_source_catalog(self, self.config["redshift_weight_criterion"]):
            print(f"         Process {self.rank} processing chunk {s:,} - {e:,}")

            gal_indexes = np.arange(s, e)
            for bin_key, cluster_list in clusters_in_bins.items():
                for i, cluster in enumerate(cluster_list):

                    sources_in_cluster = cluster_source_index[cluster["id"]]
                    if sources_in_cluster[0]>e or sources_in_cluster[-1]<s:
                        continue

                    _in_cluster = np.isin(gal_indexes, sources_in_cluster)
                    if not _in_cluster.any():
                        continue

                    _in_gals = np.isin(sources_in_cluster, gal_indexes)
                    distance = cluster_source_dist[cluster["id"]][_in_gals]
                    
                    print(f"             Compute source quantities, cluster id: {cluster['id']}")
                    print(f"                 dist: {distance}")
                    # Compute source quantities
                    weights, tangential_comp, cross_comp, g1, g2 = self.compute_sources_quantities(
                        clmm_cosmo, data, _in_cluster, cluster['redshift'], cluster['ra'], cluster['dec']
                    )

                    clusters_list_of_source_lists[bin_key][i].append((distance, weights, tangential_comp, cross_comp, g1, g2))

            gc.collect()

        print(f"Process {self.rank} done reading")

        # Now we loop through each cluster and collect all the galaxies
        # behind it from all the different processes. Each different
        # process received a different chunk of the shear catalog,
        # so we need to pull together all the shears relevant to each
        # cluster
        j = 0
        for cluster_list in clusters_list_of_source_lists.values():
            for src_list_in_cluster in cluster_list:

                if (self.rank == 0) and (j%100 == 0):
                    print(f"Collecting data for cluster {j}")

                if len(clusters_list_of_source_lists) == 0:
                    # If there is no background data for this cluster
                    # for this particular processor then we just send
                    # an empty array
                    weights = np.zeros(0)
                    tangential_comps = np.zeros(0)
                    cross_comps = np.zeros(0)
                    g1 = np.zeros(0)
                    g2 = np.zeros(0)
                    distances = np.zeros(0)
                else:
                    # Each process flattens the list of all the galaxies for this cluster
                    distances = np.concatenate([d[1] for d in src_list_in_cluster])
                    if self.config['delta_sigma']:
                        weights = np.concatenate([d[2] for d in src_list_in_cluster])
                    else:
                        #If we are using gamma, the weights do not deppend on each galaxy, so the function returns a float
                        weights = np.full(len(distances), src_list_in_cluster[0][1])
                    tangential_comps = np.concatenate([d[3] for d in src_list_in_cluster])
                    cross_comps = np.concatenate([d[4] for d in src_list_in_cluster])
                    g1 = np.concatenate([d[5] for d in src_list_in_cluster])
                    print("g2")
                    g2 = np.concatenate([d[6] for d in src_list_in_cluster])
                    print("done concatenating!")


                # If we are running in parallel then collect together the values from
                # all the processes
                if self.comm is not None:
                    print("Final collect!!!")
                    cluster.source_data = self.collect(weights, tangential_comps, cross_comps, g1, g2, distances)
                    #del cluster.list_of_source_lists

                j += 1

        print("Adding source data - DONE!")


    def collect(self, weights, tangential_comps, cross_comps, g1, g2, distances):
        # total number of background objects for t

        counts = np.array(self.comm.allgather(weights.size))
        total = counts.sum()

        # Early exit if nothing here
        if total == 0:
            weights = np.zeros(0)
            tangential_comps = np.zeros(0)
            cross_comps = np.zeros(0)
            distances = np.zeros(0)
            g1 = np.zeros(0)
            g2 = np.zeros(0)
        # This collects together all the results from different processes for this cluster
        elif self.rank == 0:
            all_weights = np.empty(total, dtype=weights.dtype)
            all_tangential_comps = np.empty(total, dtype=tangential_comps.dtype)
            all_cross_comps = np.empty(total, dtype=cross_comps.dtype)
            all_g1 = np.empty(total, dtype=tangential_comps.dtype)
            all_g2 = np.empty(total, dtype=cross_comps.dtype)
            all_distances = np.empty(total, dtype=distances.dtype)
            self.comm.Gatherv(sendbuf=distances, recvbuf=(all_distances, counts))
            self.comm.Gatherv(sendbuf=cross_comps, recvbuf=(all_cross_comps, counts))
            self.comm.Gatherv(sendbuf=tangential_comps, recvbuf=(all_tangential_comps, counts))
            self.comm.Gatherv(sendbuf=g1, recvbuf=(all_g1, counts))
            self.comm.Gatherv(sendbuf=g2, recvbuf=(all_g2, counts))
            self.comm.Gatherv(sendbuf=weights, recvbuf=(all_weights, counts))
            weights = all_weights
            tangential_comps = all_tangential_comps
            cross_comps = all_cross_comps
            distances = all_distances
            g1 = all_g1
            g2 = all_g2
        else:
            self.comm.Gatherv(sendbuf=distances, recvbuf=(None, counts))
            self.comm.Gatherv(sendbuf=cross_comps, recvbuf=(None, counts))
            self.comm.Gatherv(sendbuf=tangential_comps, recvbuf=(None, counts))
            self.comm.Gatherv(sendbuf=g1, recvbuf=(None, counts))
            self.comm.Gatherv(sendbuf=g2, recvbuf=(None, counts))
            self.comm.Gatherv(sendbuf=weights, recvbuf=(None, counts))

        return weights, tangential_comps, cross_comps, g1, g2, distances

    def compute_sources_quantities(self, clmm_cosmo, data, index, z_cluster, ra_cluster, dec_cluster):
        import clmm
        z_cluster = float(z_cluster)
        is_deltasigma = self.config["delta_sigma"]
        criterion = self.config["redshift_weight_criterion"]
        use_shape_noise = self.config["use_shape_noise"]
        sigma_c = None
        # Depending on whether we are using the PDF or not, choose
        # some keywords to give to compute_galaxy_weights
        if criterion == "pdf":
            # We need the z and PDF(z) arrays in this case
            pdf_z = data["pdf_z"]
            pdf_pz = data["pdf_pz"][index]
            # suppress user warnings containing string "nSome source redshifts are lower than the cluster redshift"
            if is_deltasigma:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    sigma_c = clmm.theory.compute_critical_surface_density_eff(
                        cosmo=clmm_cosmo,
                        z_cluster=z_cluster,
                        pzbins=pdf_z,
                        pzpdf=pdf_pz,
                        )
        elif criterion in ["zmode", "ztrue", "zmean"]:
            if is_deltasigma:
                z_source = data["redshift"][index]
                sigma_c = clmm_cosmo.eval_sigma_crit(z_cluster, z_source)
        elif criterion is None:
            pass
        else:
            raise NotImplementedError(f"Not implemented {criterion} weighting")
            
        g1 = data["g1"][index]
        g2 = data["g2"][index]
        weight = clmm.dataops.compute_galaxy_weights(
            sigma_c = sigma_c,
            is_deltasigma=is_deltasigma,
            use_shape_noise=use_shape_noise,
            shape_component1=np.array(g1),
            shape_component2=np.array(g2),
        )
        _, tangential_comp, cross_comp = clmm.compute_tangential_and_cross_components(
            ra_cluster,
            dec_cluster,
            data['ra'][index],
            data['dec'][index],
            g1,
            g2,
            coordinate_system=self.coordinate_system,
            geometry="curve",
            is_deltasigma=is_deltasigma,
            sigma_c=sigma_c,
            validate_input=True,
        )
        
        return weight, tangential_comp, cross_comp, g1, g2




    def create_cluster_ensemble(self, cluster_list, cluster_ensemble_id=0):
        import clmm
        bin_units = self.config["bin_units"].lower()
        # load cluster shear catalog using similar astropy table set up as cluster catalog
        
        # Create empty cluster ensemble 
        cluster_ensemble = clmm.ClusterEnsemble(cluster_ensemble_id) 
        
        # Loop through clusters and calculate the profiles
        ncluster = len(cluster_list)
        print('Ncluster', ncluster)

        # Extract cluster list from cluster_shears_cat
        #cluster_shear_list = self.cluster_shears_cat["cluster_id"].tolist()

        #########################################################
        #profile_group_name = "reduced_shear"
        #if self.config['delta_sigma'] == True:
        #    profile_group_name = "delta_sigma"
        #index_group.attrs["profile_type"] = profile_group_name
        #########################################################

        
        for cluster_index in range(ncluster) :

            # Select subset of background shear information for this particular cluster
            
            #print("-> cluster id ",id_cl,"  ",cluster_cat_index," ",cluster_cat_indices,"  ",cluster_cat_indices[0]+cluster_cat_indices[1])
            #print(self.profile_type) 
            #print('For cluster', id_cl, 'at z=',z_cl,'with n_source = ',len(bg_cat["source_index"]) , 'theta_max is', np.max(bg_cat["distance_arcmin"]), ' arcmin =', clmm.utils.convert_units(np.max(bg_cat["distance_arcmin"]), 'arcmin', 'Mpc', z_cl, self.clmm_cosmo), 'Mpc')
            
            # To use CLMM, need to have galaxy table in clmm.GCData type
            weights, tangential_comps, cross_comps, g1, g2, distances = cluster.source_data
            galcat = clmm.GCData(
                Table(
                    {
                        "tangential_comp_clmm": tangential_comp,
                        "cross_comp_clmm": cross_comp,
                        "weight_clmm": weight,
                        "distance_arcmin": np.degrees(distances) * 60,
                    }
                ),
                meta={"coordinate_system": self.coordinate_system}
            )
            print(galcat.meta)
            galcat['theta'] = galcat['distance_arcmin']*np.pi/(60*180) # CLMM galcat requires a column called "theta" in radians
            galcat['z'] = np.zeros(len(galcat)) # clmm needs a column named 'z' but all computation have been done 
                                                # in source_select_compute --> don't need it here, filling dummy array
            
            # Instantiating a CLMM galaxy cluster object

            gc_object = clmm.GalaxyCluster(
                int(cluster_list[cluster_index]["id"]),
                cluster_list[cluster_index]["ra"],
                cluster_list[cluster_index]["dec"],
                cluster_list[cluster_index]["redshift"],
                galcat,
            )
            gc_object.richness = cluster_list[cluster_index]["richness"]
            cat_max_distance = (
                clmm.utils.convert_units(
                    np.max(bg_cat["distance_arcmin"]), "arcmin", "Mpc", z_cl, self.clmm_cosmo
                )
                if bin_units == "mpc" else np.max(bg_cat["distance_arcmin"])
            )
            if cat_max_distance < self.distance_bins[-1]:
                print ("!!! maximum distance of source smaller than distance_bins")

            # Compute radial profile for the current cluster
            gc_object.make_radial_profile(
                    bin_units, 
                    bins=self.distance_bins,
                    cosmo=self.clmm_cosmo, 
                    tan_component_in = "tangential_comp_clmm", # name given in the CLClusterShearCatalogs stage
                    cross_component_in = "cross_comp_clmm", # name given in the CLClusterShearCatalogs stage
                    tan_component_out = "tangential_comp",
                    cross_component_out = "cross_comp",
                    weights_in = "weight_clmm", # name given in the CLClusterShearCatalogs stage
                    weights_out = "W_l",
                    include_empty_bins = True ,
                    use_weights=True
                    )
                # Quick check - Print out the profile information for the first 2 cluster of the list
            # if cluster_index == 0 or cluster_index == 1:
            #    print(gc_object.profile)

                # Add the profile to the ensemble
            cluster_ensemble.add_individual_radial_profile(
                    galaxycluster=gc_object,
                    profile_table=gc_object.profile, 
                    tan_component="tangential_comp",
                    cross_component="cross_comp",
                    weights="W_l")

        # Individual profile for all cluster of the ensemble have been computed in the loop above
        # Now, compute the stacked profile of the ensemble
        cluster_ensemble.make_stacked_radial_profile(tan_component="tangential_comp", cross_component="cross_comp", weights="W_l")  
        print("cluster ensemble computed")
            
        #compute sample covariance
        if self.config["cov_type"] == "sample_covariance":
            cluster_ensemble.compute_sample_covariance(tan_component="tangential_comp", cross_component="cross_comp")
        elif self.config["cov_type"] == "jackknife_covariance":
            cluster_ensemble.compute_jackknife_covariance(tan_component="tangential_comp", cross_component="cross_comp", n_side = self.config["jackknife_nside"])
        elif self.config["cov_type"] == "bootstrap_covariance":
            cluster_ensemble.compute_bootstrap_covariance(tan_component="tangential_comp", cross_component="cross_comp", n_bootstrap = self.config["bootstrap_nboot"])
        print("covariance computed")
    
        return cluster_ensemble
   
    
    def load_cluster_catalog_tomography_group(self):
        print("Loading binned cluster catalog")
        from astropy.table import Table
        
        with self.open_input("cluster_catalog_tomography") as f:
            clusters_in_bins = {
                key: self.load_cluster_list(group=group)
                for key, group in f["cluster_bin"].items()
            }

            
        return clusters_in_bins

    def make_ensemble(self, clusters_in_bins):
        binned_cluster_stack = {}
        for key, clusters in clusters_in_bins.items():
            print(clusters)
            if  len(clusters)>1:
                cluster_stack = self.create_cluster_ensemble(clusters, cluster_ensemble_id=key)
            else :
                cluster_stack = None
            print('cl_ensemble_created')
            
            #dict(dset_out[i].attrs), dset_out[i]['redshift'][:].size) 
            
            binned_cluster_stack[key]={'cluster_bin_edges':dict(group.attrs), 'n_cl':len(clusters), 'clmm_cluster_ensemble':cluster_stack, 'profile_type': self.profile_type}
