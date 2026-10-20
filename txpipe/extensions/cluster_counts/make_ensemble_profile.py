import os
import gc
import numpy as np
from ...base_stage import PipelineStage
from .sources_select_compute import CLClusterShearCatalogs, CombinedClusterCatalog
from ...data_types import ShearCatalog, HDFFile, PhotozPDFFile, FiducialCosmology, TomographyCatalog, PickleFile
from ...shear_calibration import Calibrator
from collections import defaultdict
import yaml
import ceci
import pickle

class CLClusterEnsembleProfiles(CLClusterShearCatalogs):
    name = "CLClusterEnsembleProfiles"
    inputs = [
        ("cluster_catalog", HDFFile), # positions for the k-means jackknife regions
        ("cluster_catalog_tomography", HDFFile), # TO TEST
        ("fiducial_cosmology", FiducialCosmology),
        ("cluster_shear_catalogs", HDFFile),
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
        # regions of the jackknife covariance: "healpix" (pixels of jackknife_nside) or "kmeans"
        # (jackknife_kmeans_nregions regions shared by all bins, defined from the positions
        # of every cluster of cluster_catalog with background sources)
        "jackknife_regions": "healpix",
        "jackknife_nside": 32,
        "jackknife_kmeans_nregions": 30,
        "jackknife_kmeans_seed": 11,
        "bootstrap_nboot": 100,
        #coordinate_system for shear
        #"coordinate_system" : 'euclidean' #Must be either 'celestial' or 'euclidean'
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
        
        self.cluster_shears_cat, self.coordinate_system, self.profile_type= self.load_cluster_shear_catalog()

        print (self.coordinate_system)

        # row of each cluster in cluster_shears_cat; with repeated ids, clusters would
        # silently get the sources of another cluster
        cluster_ids = self.cluster_shears_cat["cluster_id"]
        if np.unique(cluster_ids).size != len(cluster_ids):
            raise ValueError("cluster_id is not unique in cluster_shear_catalogs, "
                             "so clusters cannot be matched to their sources")
        self.cluster_shear_row = {cluster_id: i for i, cluster_id in enumerate(cluster_ids)}

        if self.config["jackknife_regions"] not in ("healpix", "kmeans"):
            raise ValueError(f"jackknife_regions must be 'healpix' or 'kmeans', "
                             f"not '{self.config['jackknife_regions']}'")
        self.jackknife_centers = None
        if self.config["cov_type"] == "jackknife_covariance" and self.config["jackknife_regions"] == "kmeans":
            self.jackknife_centers = self.compute_jackknife_centers()

        
        with self.open_input("fiducial_cosmology", wrapper=True) as f:
            ccl_cosmo = f.to_ccl()
            self.clmm_cosmo = clmm.cosmology.ccl.CCLCosmology()
            self.clmm_cosmo.set_be_cosmo(ccl_cosmo)
                 
        # load cluster catalog as an astropy table
        #clusters = self.load_cluster_catalog()
        

        cluster_stack_dict = self.load_cluster_catalog_tomography_group() 
        pickle.dump(cluster_stack_dict, open(self.get_output("cluster_profiles"), 'wb'))

                    

    def compute_jackknife_centers(self):
        """Centers of the k-means jackknife regions shared by all bins, from the positions of
        every cluster of cluster_catalog with background sources (no redshift or richness cut,
        so the regions do not depend on the binning)."""
        import clmm

        with self.open_input("cluster_catalog") as f:
            ra = f["clusters/ra"][:]
            dec = f["clusters/dec"][:]
            cluster_id = f["clusters/cluster_id"][:]
        # cluster_shear_catalogs keeps the row order of cluster_catalog
        if not np.array_equal(cluster_id, self.cluster_shears_cat["cluster_id"]):
            raise ValueError("cluster_catalog and cluster_shear_catalogs rows do not match")
        has_sources = np.asarray(self.cluster_shears_cat["cluster_sample_count"]) > 0
        centers = clmm.utils.compute_kmeans_centers(
            ra[has_sources], dec[has_sources], self.config["jackknife_kmeans_nregions"],
            seed=self.config["jackknife_kmeans_seed"])
        print(f"{len(centers)} k-means jackknife regions from {has_sources.sum()} clusters with sources")
        return centers

    def load_cluster_shear_catalog(self, indices = None) :
        from astropy.table import Table

        # Read catalog group data only
        if indices == None:
            with self.open_input("cluster_shear_catalogs") as f:
                meta_coord_sys = f['provenance'].attrs['config/coordinate_system']
                g_idx = f["index/"]
                profile_type = g_idx.attrs.get("profile_type")
                g = f["catalog/"]
                cluster_id = g['cluster_id'][:]
                cluster_sample_start = g['cluster_sample_start'][:]
                cluster_sample_count = g['cluster_sample_count'][:]
                cluster_theta_max_arcmin = g['cluster_theta_max_arcmin'][:]
            tab = Table({"cluster_id": cluster_id,
                         "cluster_sample_count": cluster_sample_count, "cluster_sample_start": cluster_sample_start,
                         "cluster_theta_max_arcmin": cluster_theta_max_arcmin})

            return tab, meta_coord_sys, profile_type

        # Read index group data corresponding to a specific cluster_id
        #     indices are defined base on parameter stored in catalog/cluster_sample_start and _count for the given cluster id
        with self.open_input("cluster_shear_catalogs") as f:
            ifirst,icount = indices
            ilast = ifirst+icount
            meta_coord_sys = f['provenance'].attrs['config/coordinate_system']
            g = f["index/"]
            profile_type = g.attrs["profile_type"]
            cluster_index = g['cluster_index'][ifirst:ilast]
            cluster_id = g['cluster_id'][ifirst:ilast]
            tangential_comp = g['tangential_comp'][ifirst:ilast]
            cross_comp = g['cross_comp'][ifirst:ilast]
            source_index = g['source_index'][ifirst:ilast]
            weight = g['weight'][ifirst:ilast]
            distance_arcmin = g['distance_arcmin'][ifirst:ilast]


        tab = Table({"cluster_index": cluster_index, "cluster_id" : cluster_id, "tangential_comp_clmm": tangential_comp,
                      "cross_comp_clmm": cross_comp, "source_index": source_index,
                      "weight_clmm": weight, "distance_arcmin": distance_arcmin})
                  
        return tab, meta_coord_sys, profile_type




    def create_cluster_ensemble(self, cluster_list, cluster_ensemble_id=0):
        import clmm
        bin_units = self.config["bin_units"].lower()
        # load cluster shear catalog using similar astropy table set up as cluster catalog
        #cluster_shears_cat, coordinate_system = self.load_cluster_shear_catalog() 
        
        # Create empty cluster ensemble 
        cluster_ensemble = clmm.ClusterEnsemble(cluster_ensemble_id) 
        
        # Loop through clusters and calculate the profiles
        ncluster = len(cluster_list)
        print('Ncluster', ncluster)

        self.jackknife_info = {"n_jk": None}
        n_skipped = 0

        for cluster_index in range(ncluster) :

            # Select subset of background shear information for this particular cluster
            
            z_cl = cluster_list[cluster_index]["redshift"]
            rich_cl = cluster_list[cluster_index]["richness"]
            ra_cl = cluster_list[cluster_index]["ra"]
            dec_cl = cluster_list[cluster_index]["dec"]
            id_cl = cluster_list[cluster_index]["id"]

            # Read cluster shear catalog data corresponding to the current cluster_id (i.e. id_cl)
            cluster_cat_index = self.cluster_shear_row[id_cl]
            cluster_cat_indices = ( int(self.cluster_shears_cat["cluster_sample_start"][cluster_cat_index]),
                                    int(self.cluster_shears_cat["cluster_sample_count"][cluster_cat_index]))
            #print("-> cluster id ",id_cl,"  ",cluster_cat_index," ",cluster_cat_indices,"  ",cluster_cat_indices[0]+cluster_cat_indices[1])
            
            bg_cat, _, self.profile_type = self.load_cluster_shear_catalog(cluster_cat_indices)

            # Clusters outside the shear footprint (or in masked regions) have no
            # background sources -- skip them rather than failing on empty arrays
            if len(bg_cat["source_index"]) == 0:
                n_skipped += 1
                continue

            print(self.profile_type)
            print('For cluster', id_cl, 'at z=',z_cl,'with n_source = ',len(bg_cat["source_index"]) , 'theta_max is', np.max(bg_cat["distance_arcmin"]), ' arcmin =', clmm.utils.convert_units(np.max(bg_cat["distance_arcmin"]), 'arcmin', 'Mpc', z_cl, self.clmm_cosmo), 'Mpc')
            
            # To use CLMM, need to have galaxy table in clmm.GCData type
            galcat = clmm.GCData(bg_cat, meta={"coordinate_system": self.coordinate_system})
            print(galcat.meta)
            galcat['theta'] = galcat['distance_arcmin']*np.pi/(60*180) # CLMM galcat requires a column called "theta" in radians
            galcat['z'] = np.zeros(len(galcat)) # clmm needs a column named 'z' but all computation have been done 
                                                # in source_select_compute --> don't need it here, filling dummy array
            
            # Instantiating a CLMM galaxy cluster object
            gc_object = clmm.GalaxyCluster(int(id_cl), ra_cl, dec_cl, z_cl, galcat)
            gc_object.richness = rich_cl
            cat_max_distance = (clmm.utils.convert_units(np.max(bg_cat["distance_arcmin"]), "arcmin", "Mpc", z_cl, self.clmm_cosmo) if bin_units == "mpc" else np.max(bg_cat["distance_arcmin"]))
            if cat_max_distance < self.distance_bins[-1]:
                print ("!!! maximum distance of source smaller than distance_bins")

            del bg_cat

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

        n_used = ncluster - n_skipped
        print(f"Skipped {n_skipped} of {ncluster} clusters with no background sources in {cluster_ensemble_id}")
        if n_used < 2:
            print(f"Only {n_used} cluster(s) with sources in {cluster_ensemble_id}; not stacking")
            return None

        # Individual profile for all cluster of the ensemble have been computed in the loop above
        # Now, compute the stacked profile of the ensemble
        cluster_ensemble.make_stacked_radial_profile(tan_component="tangential_comp", cross_component="cross_comp", weights="W_l")  
        print("cluster ensemble computed")
            
        #compute sample covariance
        if self.config["cov_type"] == "sample_covariance":
            cluster_ensemble.compute_sample_covariance(tan_component="tangential_comp", cross_component="cross_comp")
        elif self.config["cov_type"] == "jackknife_covariance":
            # healpix pixels, or the shared k-means regions when jackknife_centers is set
            cluster_ensemble.compute_jackknife_covariance(tan_component="tangential_comp", cross_component="cross_comp",
                                                          n_side = self.config["jackknife_nside"], centers=self.jackknife_centers)
            self.jackknife_info = self.get_jackknife_info(cluster_ensemble)
            print(f"jackknife covariance computed with {self.jackknife_info['n_jk']} "
                  f"{self.config['jackknife_regions']} regions")
        elif self.config["cov_type"] == "bootstrap_covariance":
            cluster_ensemble.compute_bootstrap_covariance(tan_component="tangential_comp", cross_component="cross_comp", n_bootstrap = self.config["bootstrap_nboot"])
        print("covariance computed")

        return cluster_ensemble

    def get_jackknife_info(self, cluster_ensemble):
        """Jackknife regions of the clusters of an ensemble, for the output: number of occupied
        regions, and for k-means the shared centers and the region of each cluster."""
        import clmm
        import healpy

        ra, dec = cluster_ensemble.data["ra"], cluster_ensemble.data["dec"]
        info = {"jackknife_regions": self.config["jackknife_regions"]}
        if self.jackknife_centers is None:
            labels = healpy.ang2pix(self.config["jackknife_nside"], ra, dec, nest=True, lonlat=True)
        else:
            labels = clmm.utils.assign_kmeans_regions(ra, dec, self.jackknife_centers)
            info["jackknife_centers"] = self.jackknife_centers
            info["jackknife_labels"] = labels
        info["n_jk"] = np.unique(labels).size
        return info
   
    
    def load_cluster_catalog_tomography_group(self): # NEED TO CHNAGE FUNCTION NAME
        from astropy.table import Table
        binned_cluster_stack = {}
        
        with self.open_input("cluster_catalog_tomography") as f:
            k = f["cluster_bin"].keys()

            for key in k :
                group = f["cluster_bin"][key]
                clusters = self.load_cluster_list(group=group) #elf.get_cluster_indice( DOES THIS FUNCTION COMES FROM ?
                print(clusters)
                if  len(clusters)>1:
                    cluster_stack = self.create_cluster_ensemble(clusters, cluster_ensemble_id=key)
                else :
                    cluster_stack = None
                    self.jackknife_info = {"n_jk": None}
                print('cl_ensemble_created')
                
                #dict(dset_out[i].attrs), dset_out[i]['redshift'][:].size) 
                
                binned_cluster_stack[key]={'cluster_bin_edges':dict(group.attrs), 'n_cl':len(clusters), 'clmm_cluster_ensemble':cluster_stack, 'profile_type': self.profile_type,
                                         **self.jackknife_info}
                
            
        return binned_cluster_stack
