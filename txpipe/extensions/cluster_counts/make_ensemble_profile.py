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
        #type of profile
        "delta_sigma_profile" : True,
        "shear_profile" : False,
        "magnification_profile" : False,
        #coordinate_system for shear
        "coordinate_system" : 'euclidean' #Must be either 'celestial' or 'euclidean'
    }

    def run(self):
        import sklearn.neighbors
        import astropy
        import h5py
        import clmm
        import clmm.cosmology.ccl

        radial_bins = clmm.dataops.make_bins(self.config["r_min"], self.config["r_max"], nbins=self.config["nbins"], method="evenlog10width")
        print (radial_bins)
        
        # load cluster shear catalog using similar astropy table set up as cluster catalog
        cluster_shears_cat = self.load_cluster_shear_catalog()  
        
        # Store the profiles for each cluster
        # per_cluster_data = list()

        with self.open_input("fiducial_cosmology", wrapper=True) as f:
            ccl_cosmo = f.to_ccl()
            clmm_cosmo = clmm.cosmology.ccl.CCLCosmology()
            clmm_cosmo.set_be_cosmo(ccl_cosmo)
            
                
        
        
                 
        # load cluster catalog as an astropy table
        #clusters = self.load_cluster_catalog()
        
        if self.config["delta_sigma_profile"]==True:
            
            cluster_stack_dict = self.load_cluster_catalog_tomography_group(radial_bins, clmm_cosmo, cluster_shears_cat) ### TEST BUT NOT WORKING
        #clusters = self.load_cluster_list(group=bins[0])
        
        #print(bins[0].keys())

            pickle.dump(cluster_stack_dict, open(self.get_output("cluster_profiles"), 'wb'))
        
        #cluster_stack.save(self.get_output("cluster_profiles"))
        else :
            print("Config option not supported, only delta_sigma_profiles supported")

                    

    def load_cluster_shear_catalog(self) :
        from astropy.table import Table

        with self.open_input("cluster_shear_catalogs") as f:
            g = f["index/"]
            cluster_index = g['cluster_index'][:]
            cluster_id = g['cluster_id'][:]
            tangential_comp = g['tangential_comp'][:]
            cross_comp = g['cross_comp'][:]
            source_index = g['source_index'][:]
            weight = g['weight'][:]
            distance_arcmin = g['distance_arcmin'][:]

        print(len(cluster_index), len(tangential_comp), len(source_index))
        
        tab = Table({"cluster_index": cluster_index, "cluster_id" : cluster_id, "tangential_comp_clmm": tangential_comp,
                      "cross_comp_clmm": cross_comp, "source_index": source_index,
                      "weight_clmm": weight, "distance_arcmin": distance_arcmin})
    
        print(tab[0:4])
                  
        return tab 



    def create_cluster_ensemble(self, radial_bins, clmm_cosmo, cluster_list, cluster_shears_cat, cluster_ensemble_id=0):
        import clmm
            
        # Create empty cluster ensemble 
        cluster_ensemble = clmm.ClusterEnsemble(cluster_ensemble_id) 
        
        # Loop through clusters and calculate the profiles
        ncluster = len(cluster_list)
        print('Ncluster', ncluster)
        
        for cluster_index in range(ncluster) :

            # Select subset of background shear information for this particular cluster
            
            print('cluster_index', cluster_index)
            
            #mask = (cluster_shears_cat["cluster_id"] == id_cl) #THERE IS A PROBLEM HERE !!!
       
            z_cl = cluster_list[cluster_index]["redshift"]
            rich_cl = cluster_list[cluster_index]["richness"]
            ra_cl = cluster_list[cluster_index]["ra"]
            dec_cl = cluster_list[cluster_index]["dec"]
            id_cl = cluster_list[cluster_index]["id"]

            
            mask = (cluster_shears_cat['cluster_id'] == id_cl)
            print(mask)
            bg_cat = cluster_shears_cat[mask]
               
            print('For cluster', id_cl, 'at z=',z_cl, 'theta_max is', np.max(bg_cat["distance_arcmin"]), ' arcmin =', clmm.utils.convert_units(np.max(bg_cat["distance_arcmin"]), 'arcmin', 'Mpc', z_cl, clmm_cosmo), 'Mpc')
            print(len(bg_cat), bg_cat[0:3])
            
            # To use CLMM, need to have galaxy table in clmm.GCData type
            galcat = clmm.GCData(bg_cat)
            galcat['theta'] = galcat['distance_arcmin']*np.pi/(60*180) # CLMM galcat requires a column called "theta" in radians
            galcat['z'] = np.zeros(len(galcat)) # clmm needs a column named 'z' but all computation have been done 
                                                # in source_select_compute --> don't need it here, filling dummy array
            
            # Instantiating a CLMM galaxy cluster object
            gc_object = clmm.GalaxyCluster(np.int(id_cl), ra_cl, dec_cl, z_cl, galcat, coordinate_system = self.config["coordinate_system"] )
            gc_object.richness = rich_cl


            if (clmm.utils.convert_units(np.max(bg_cat["distance_arcmin"]), 'arcmin', 'Mpc', z_cl, clmm_cosmo)< radial_bins[-1]):
                print ("!!! maximum radial distance of source smaller than radial_bins")

            # Compute radial profile for the current cluster
            gc_object.make_radial_profile(
                    "Mpc", 
                    bins=radial_bins,
                    cosmo=clmm_cosmo, 
                    tan_component_in = "tangential_comp_clmm", # name given in the CLClusterShearCatalogs stage
                    cross_component_in = "cross_comp_clmm", # name given in the CLClusterShearCatalogs stage
                    tan_component_out = "tangential_comp",
                    cross_component_out = "cross_comp",
                    weights_in = "weight_clmm", # name given in the CLClusterShearCatalogs stage
                    weights_out = "W_l",
                    include_empty_bins = True 
                    )


                # Quick check - Print out the profile information for the first 2 cluster of the list
            #if cluster_index == 0 or cluster_index == 1:
            if cluster_index <2:  
#                print(galcat['weight_clmm'])
                print(gc_object.profile)

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
        print(cluster_ensemble.stacked_data)
            
        #compute sample covariance
        cluster_ensemble.compute_sample_covariance(tan_component="tangential_comp", cross_component="cross_comp")
    
    
        return cluster_ensemble
   
    
    def load_cluster_catalog_tomography_group(self, radial_bins, clmm_cosmo, cluster_shears_cat): # NEED TO CHNAGE FUNCTION NAME
        from astropy.table import Table
        binned_cluster_stack = {}
        
        with self.open_input("cluster_catalog_tomography") as f:
            k = f["cluster_bin"].keys()

            for key in k :
                group = f["cluster_bin"][key]
                clusters = self.load_cluster_list(group=group) #elf.get_cluster_indice( DOES THIS FUNCTION COMES FROM ?
                print(key, group, dict(group.attrs), len(clusters), clusters)
                
                if  len(clusters)>1:
                    cluster_stack = self.create_cluster_ensemble(radial_bins, clmm_cosmo, clusters, cluster_shears_cat, cluster_ensemble_id=key)
                else :
                    cluster_stack = None
                print('cl_ensemble_created')
                
                #dict(dset_out[i].attrs), dset_out[i]['redshift'][:].size) 
                
                binned_cluster_stack[key]={'cluster_bin_edges':dict(group.attrs), 'n_cl':len(clusters), 'clmm_cluster_ensemble':cluster_stack}
                
            
        return binned_cluster_stack

 