import os
import gc
import numpy as np
from ...base_stage import PipelineStage
from .sources_select_compute import CLClusterShearCatalogs, CombinedClusterCatalog
from ...data_types import ShearCatalog, HDFFile, PhotozPDFFile, FiducialCosmology, TomographyCatalog, PickleFile
from ...utils.calibrators import Calibrator
from collections import defaultdict
import yaml
import ceci
import pickle

class CLClusterEnsembleProfiles(CLClusterShearCatalogs):
    name = "CLClusterEnsembleProfiles"
    inputs = [
        #("cluster_catalog", HDFFile),
        ("cluster_catalog_tomography", HDFFile), # TO TEST
        ("fiducial_cosmology", FiducialCosmology),
        ("cluster_shear_catalogs", HDFFile),
    ]

    outputs = [
        ("cluster_profiles",  PickleFile),
    ]

    config_options = {
        #radial bin definition
        "r_min" : 0.2, #in Mpc
        "r_max" : 3.0, #in Mpc
        "nbins" : 5, # number of bins
        #Angular bin definition
        "angle_arcmin_min" : 25.0, #in arcmin
        "angle_arcmin_max" : 45.0, #in arcmin
        "nbins" : 5, # number of bins
        #type of profile
        "delta_sigma_profile" : True,
        "shear_profile" : False,
        "magnification_profile" : False,
        "units": "mpc" # options are mpc or arcmin
        #coordinate_system for shear
        #"coordinate_system" : 'euclidean' #Must be either 'celestial' or 'euclidean'
    }

    def run(self):
        import sklearn.neighbors
        import astropy
        import h5py
        import clmm
        import clmm.cosmology.ccl
        if self.config["units"].lower()== "mpc":
            bin_min = self.config["angle_arcmin_min"]
            bin_max = self.config["angle_arcmin_max"]
        else:
            bin_min = self.config["r_min"]
            bin_max = self.config["r_max"]
        self.distance_bins = clmm.dataops.make_bins(bin_min, bin_max, nbins=self.config["nbins"], method="evenlog10width")
        print (self.distance_bins) 

        self.cluster_shears_cat, self.coordinate_system = self.load_cluster_shear_catalog() 

        print (self.coordinate_system) 
        
        with self.open_input("fiducial_cosmology", wrapper=True) as f:
            ccl_cosmo = f.to_ccl()
            self.clmm_cosmo = clmm.cosmology.ccl.CCLCosmology()
            self.clmm_cosmo.set_be_cosmo(ccl_cosmo)
            
        
                 
        # load cluster catalog as an astropy table
        #clusters = self.load_cluster_catalog()
        

        cluster_stack_dict = self.load_cluster_catalog_tomography_group() 
        pickle.dump(cluster_stack_dict, open(self.get_output("cluster_profiles"), 'wb'))
        print(cluster_stack_dict)
                    

    def load_cluster_shear_catalog(self) :
        from astropy.table import Table

        with self.open_input("cluster_shear_catalogs") as f:
            meta_coord_sys = f['provenance'].attrs['config/coordinate_system'] 
            g = f["index/"]
            cluster_index = g['cluster_index'][:]
            cluster_id = g['cluster_id'][:]
            tangential_comp = g['tangential_comp'][:]
            cross_comp = g['cross_comp'][:]
            source_index = g['source_index'][:]
            weight = g['weight'][:]
            distance_arcmin = g['distance_arcmin'][:]
            profile_type = g.attrs["profile_type"]

        print(len(cluster_index), len(tangential_comp), len(source_index))
        
        tab = Table({"cluster_index": cluster_index, "cluster_id" : cluster_id, "tangential_comp_clmm": tangential_comp,
                      "cross_comp_clmm": cross_comp, "source_index": source_index,
                      "weight_clmm": weight, "distance_arcmin": distance_arcmin, "profile_type": profile_type})
    
                  
        return tab, meta_coord_sys



    def create_cluster_ensemble(self, cluster_list, cluster_ensemtble_id=0):
        import clmm

        # load cluster shear catalog using similar astropy table set up as cluster catalog
        #cluster_shears_cat, coordinate_system = self.load_cluster_shear_catalog() 
        
        # Create empty cluster ensemble 
        cluster_ensemble = clmm.ClusterEnsemble(cluster_ensemble_id) 
        
        # Loop through clusters and calculate the profiles
        ncluster = len(cluster_list)
        print('Ncluster', ncluster)
        
        for cluster_index in range(ncluster) :

            # Select subset of background shear information for this particular cluster
            
            z_cl = cluster_list[cluster_index]["redshift"]
            rich_cl = cluster_list[cluster_index]["richness"]
            ra_cl = cluster_list[cluster_index]["ra"]
            dec_cl = cluster_list[cluster_index]["dec"]
            id_cl = cluster_list[cluster_index]["id"]

            
            mask = (self.cluster_shears_cat['cluster_id'] == id_cl)
            bg_cat = self.cluster_shears_cat[mask]
               
            print('For cluster', id_cl, 'at z=',z_cl,'with n_source = ',len(bg_cat["source_index"]) , 'theta_max is', np.max(bg_cat["distance_arcmin"]), ' arcmin =', clmm.utils.convert_units(np.max(bg_cat["distance_arcmin"]), 'arcmin', 'Mpc', z_cl, self.clmm_cosmo), 'Mpc')
            
            
            # To use CLMM, need to have galaxy table in clmm.GCData type
            galcat = clmm.GCData(bg_cat, meta={"coordinate_system": self.coordinate_system})
            print(galcat.meta)
            galcat['theta'] = galcat['distance_arcmin']*np.pi/(60*180) # CLMM galcat requires a column called "theta" in radians
            galcat['z'] = np.zeros(len(galcat)) # clmm needs a column named 'z' but all computation have been done 
                                                # in source_select_compute --> don't need it here, filling dummy array
            
            # Instantiating a CLMM galaxy cluster object
            gc_object = clmm.GalaxyCluster(np.int(id_cl), ra_cl, dec_cl, z_cl, galcat)
            gc_object.richness = rich_cl
            cat_max_distance = (clmm.utils.convert_units(np.max(bg_cat["distance_arcmin"]), "arcmin", "Mpc", z_cl, self.clmm_cosmo) if self.units == "mpc" else np.max(bg_cat["distance_arcmin"]))
            if cat_max_distance < self.distance_bins[-1]:
                print ("!!! maximum distance of source smaller than distance_bins")

            # Compute radial profile for the current cluster
            gc_object.make_radial_profile(
                    self.units, 
                    bins=self.distance_bins,
                    cosmo=self.clmm_cosmo, 
                    tan_component_in = "tangential_comp_clmm", # name given in the CLClusterShearCatalogs stage
                    cross_component_in = "cross_comp_clmm", # name given in the CLClusterShearCatalogs stage
                    tan_component_out = "tangential_comp",
                    cross_component_out = "cross_comp",
                    weights_in = "weight_clmm", # name given in the CLClusterShearCatalogs stage
                    weights_out = "W_l",
                    include_empty_bins = True 
                    )
            if np.isnan(gc_object.galcat['W_l']).all():
                print(gc_object.galcat['tangential_comp_clmm'], gc_object.z, gc_object.unique_id)


                # Quick check - Print out the profile information for the first 2 cluster of the list
            #if cluster_index == 0 or cluster_index == 1:
            #if cluster_index <2:  
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
        cluster_ensemble.compute_sample_covariance(tan_component="tangential_comp", cross_component="cross_comp")
        print("covariance computed")
    
        return cluster_ensemble
   
    
    def load_cluster_catalog_tomography_group(self): # NEED TO CHNAGE FUNCTION NAME
        from astropy.table import Table
        binned_cluster_stack = {}
        
        with self.open_input("cluster_catalog_tomography") as f:
            k = f["cluster_bin"].keys()

            for key in k :
                group = f["cluster_bin"][key]
                clusters = self.load_cluster_list(group=group) #elf.get_cluster_indice( DOES THIS FUNCTION COMES FROM ?
                print(key, group, dict(group.attrs), len(clusters), clusters)
                
                if  len(clusters)>1:
                    cluster_stack = self.create_cluster_ensemble(clusters, cluster_ensemble_id=key)
                else :
                    cluster_stack = None
                print('cl_ensemble_created')
                
                #dict(dset_out[i].attrs), dset_out[i]['redshift'][:].size) 
                
                binned_cluster_stack[key]={'cluster_bin_edges':dict(group.attrs), 'n_cl':len(clusters), 'clmm_cluster_ensemble':cluster_stack}
                
            
        return binned_cluster_stack

 
