import os
import gc
import numpy as np
from ...base_stage import PipelineStage
from .sources_select_compute import CLClusterShearCatalogs, CombinedClusterCatalog
from ...data_types import ShearCatalog, HDFFile, PhotozPDFFile, FiducialCosmology, TomographyCatalog, PickleFile
from .utils import iterate_source_catalog
from ...calibration.calibration_utils import Calibrator
from collections import defaultdict
import yaml
import ceci
import pickle
import warnings

class CLClusterEnsembleProfiles(CLClusterShearCatalogs):
    """Compute stacked radial shear profiles of ensembles of clusters.

    For each redshift/richness bin of clusters, computes the tangential and cross shear profile
    of each cluster, using CLMM, and stacks them. Clusters with no background sources are
    skipped, and ensembles with fewer than two usable clusters are not stacked.

    The sources of each cluster are its pairs from CLClusterShearCatalogs. Their shears are
    calibrated with the calibration of the whole source sample in shear_tomography_catalog
    (shear_calibration "global", or "none"), and their quantities are saved in
    cluster_source_quantities. source_photoz_pdfs can be "none" if redshift_weight_criterion
    does not use photo-z.
    """
    name = "CLClusterEnsembleProfiles"
    # The stacking is done in one process
    parallel = False
    inputs = [
        #("cluster_catalog", HDFFile),
        ("shear_catalog", ShearCatalog),
        ("shear_tomography_catalog", TomographyCatalog),
        ("cluster_catalog_tomography", HDFFile), # TO TEST
        ("fiducial_cosmology", FiducialCosmology),
        ("cluster_shear_selection", HDFFile),
        ("source_photoz_pdfs", PhotozPDFFile),
    ]

    outputs = [
        ("cluster_profiles",  PickleFile),
        ("cluster_source_quantities", HDFFile),
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
        "use_shape_noise": False, # If True, the weights include the shape noise of the whole source sample (a constant, so the profiles do not change)
        "use_true_shear": False,
        "delta_sigma": True, # If True then we compute delta-sigma, otherwise we compute gamma
        "redshift_weight_criterion": "zmode",  # pdf / mean / true / None
        "chunk_rows": 100_000,  # rows to read at once from source cat
        "shear_calibration": "global", # global / none
        "subtract_mean_shear": False,
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
        self.profile_type = "delta_sigma" if self.config["delta_sigma"] else "reduced_shear"

        
        with self.open_input("fiducial_cosmology", wrapper=True) as f:
            ccl_cosmo = f.to_ccl()
            self.clmm_cosmo = clmm.cosmology.ccl.CCLCosmology()
            self.clmm_cosmo.set_be_cosmo(ccl_cosmo)
                 
        clusters_in_bins, bin_attrs = self.load_cluster_catalog_tomography_group() 
        clusters_list_of_source_lists = self.get_source_data(clusters_in_bins)
        self.write_source_quantities(clusters_in_bins, clusters_list_of_source_lists)
        cluster_stack_dict = self.make_ensemble(clusters_in_bins, bin_attrs, clusters_list_of_source_lists) 
        pickle.dump(cluster_stack_dict, open(self.get_output("cluster_profiles"), 'wb'))

    def get_source_data(self, clusters_in_bins):

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
        # get cluster-galaxy pairs, only for the clusters in the bins and within the
        # outer radial bin, the only ones that enter the profiles
        n_small_search = 0
        cluster_source_index = {}
        cluster_source_dist = {}
        with self.open_input("cluster_shear_selection") as f:
            catalog_row = {cl_id: i for i, cl_id in enumerate(f["catalog/cluster_id"][:])}
            sample_start = f["catalog/cluster_sample_start"][:]
            sample_count = f["catalog/cluster_sample_count"][:]
            search_angle = f["catalog/cluster_theta_max_arcmin"][:]
            for cluster_list in clusters_in_bins.values():
                for cluster in cluster_list:
                    i = catalog_row[cluster["id"]]
                    # outer radial bin in arcmin. The 1e-6 margins are for rounding.
                    max_angle = clmm.utils.convert_units(
                        self.distance_bins[-1], self.config["bin_units"], "arcmin", float(cluster["redshift"]), clmm_cosmo
                    )
                    n_small_search += search_angle[i] < max_angle * (1 - 1e-6)
                    rows = slice(sample_start[i], sample_start[i] + sample_count[i])
                    distance = f["index/distance_arcmin"][rows]
                    near = distance <= max_angle * (1 + 1e-6)
                    cluster_source_index[cluster["id"]] = f["index/source_index"][rows][near]
                    cluster_source_dist[cluster["id"]] = distance[near]
        if n_small_search:
            print(f"!!! {n_small_search} clusters have a search radius smaller than the outer radial bin")

        # sort gal_index just to be sure:
        _sort_order = {key: np.argsort(value) for key, value in cluster_source_index.items()}
        cluster_source_index = {key: value[_sort_order[key]] for key, value in cluster_source_index.items()}
        cluster_source_dist = {key: value[_sort_order[key]] for key, value in cluster_source_dist.items()}

        # add list to fill with sources
        clusters_list_of_source_lists = {
            bin_key: [[] for cluster in cluster_list]
            for bin_key, cluster_list in clusters_in_bins.items()
        }

        # The shears are calibrated with the calibration of the
        # whole source sample (the 2D bin) in shear_tomography_catalog
        if self.config["shear_calibration"] == "global":
            _, calibrator = Calibrator.load(self.get_input("shear_tomography_catalog"), null=self.config["use_true_shear"])
        elif self.config["shear_calibration"] == "none":
            calibrator = None
        else:
            raise ValueError(f"shear_calibration must be global or none, not {self.config['shear_calibration']}")

        # The shape noise of the whole source sample, the same for all the sources
        self.shape_noise = None
        if self.config["use_shape_noise"]:
            with self.open_input("shear_tomography_catalog") as f:
                self.shape_noise = float(f["counts/sigma_e_2d"][0])

        print("    Loop through the source catalog")
        # Loop through the source catalog.
        for s, e, data in iterate_source_catalog(self, self.config["redshift_weight_criterion"]):
            print(f"         Process {self.rank} processing chunk {s:,} - {e:,}")

            if calibrator is not None:
                data["g1"], data["g2"] = calibrator.apply(data["g1"], data["g2"], subtract_mean=self.config["subtract_mean_shear"])

            for bin_key, cluster_list in clusters_in_bins.items():
                for i, cluster in enumerate(cluster_list):

                    # the sources of this cluster in this chunk (the indices are sorted)
                    sources_in_cluster = cluster_source_index[cluster["id"]]
                    first, last = np.searchsorted(sources_in_cluster, [s, e])
                    if first == last:
                        continue

                    # Compute source quantities
                    sources = self.compute_sources_quantities(
                        clmm_cosmo, data, sources_in_cluster[first:last] - s, cluster['redshift'], cluster['ra'], cluster['dec']
                    )
                    sources["source_index"] = sources_in_cluster[first:last]
                    sources["distance_arcmin"] = cluster_source_dist[cluster["id"]][first:last]
                    print("weights:", sources["weight"].min(), sources["weight"].max())

                    clusters_list_of_source_lists[bin_key][i].append(sources)

            #gc.collect()


        print(f"Process {self.rank} done reading")

        # Now we loop through each cluster and join the pieces of its
        # sources from the different chunks of the shear catalog.
        # Clusters without sources keep an empty list.
        j = 0
        for cluster_list in clusters_list_of_source_lists.values():
            for k, src_list_in_cluster in enumerate(cluster_list):

                if (self.rank == 0) and (j%100 == 0):
                    print(f"Collecting data for cluster {j}")

                if len(src_list_in_cluster) > 0:
                    cluster_list[k] = {col: np.concatenate([d[col] for d in src_list_in_cluster]) for col in src_list_in_cluster[0]}

                j += 1

        print("Adding source data - DONE!")

        return clusters_list_of_source_lists

    def compute_sources_quantities(self, clmm_cosmo, data, index, z_cluster, ra_cluster, dec_cluster):
        import clmm
        z_cluster = float(z_cluster)
        is_deltasigma = self.config["delta_sigma"]
        criterion = self.config["redshift_weight_criterion"]
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
                    # CLMM fails for a single p(z), so in that case we give it twice
                    sigma_c = clmm.theory.compute_critical_surface_density_eff(
                        cosmo=clmm_cosmo,
                        z_cluster=z_cluster,
                        pzbins=pdf_z,
                        pzpdf=np.repeat(pdf_pz, 2, axis=0) if len(pdf_pz) == 1 else pdf_pz,
                        )[:len(pdf_pz)]
        elif criterion in ["zmode", "ztrue", "zmean", "zmedian"]:
            if is_deltasigma:
                z_source = data["redshift"][index]
                sigma_c = clmm_cosmo.eval_sigma_crit(z_cluster, z_source)
        elif criterion is None:
            pass
        else:
            raise NotImplementedError(f"Not implemented {criterion} weighting")
            
        g1 = data["g1"][index]
        g2 = data["g2"][index]
        weights = clmm.dataops.compute_galaxy_weights(
            sigma_c = sigma_c,
            is_deltasigma=is_deltasigma,
            use_shape_error=self.shape_noise is not None,
            shape_component1_err=self.shape_noise,
            shape_component2_err=self.shape_noise,
        )
        # CLMM gives a single number when all the sources have the same weight
        weights = np.broadcast_to(weights, g1.shape).astype(float)
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
        
        sources = {
            "ra": data["ra"][index],
            "dec": data["dec"][index],
            "g1": g1,
            "g2": g2,
            "weight": weights,
            "tangential_comp": tangential_comp,
            "cross_comp": cross_comp,
        }
        if sigma_c is not None:
            sources["sigma_c"] = sigma_c
        return sources

    def write_source_quantities(self, clusters_in_bins, clusters_list_of_source_lists):
        """
        Save the sources of the clusters in the bins with their quantities. In catalog/,
        each cluster has the start and the number of its sources in index/.
        """
        cluster_ids = np.array([cluster["id"] for cluster_list in clusters_in_bins.values() for cluster in cluster_list], dtype=np.int64)
        # clusters without sources have an empty list
        sources = [s for source_lists in clusters_list_of_source_lists.values() for s in source_lists]
        counts = np.array([len(s["source_index"]) if len(s) else 0 for s in sources], dtype=np.int64)
        columns = ["source_index", "distance_arcmin", "ra", "dec", "g1", "g2", "weight", "tangential_comp", "cross_comp"]
        if self.config["delta_sigma"]:
            columns.append("sigma_c")

        with self.open_output("cluster_source_quantities") as f:
            catalog_group = f.create_group("catalog")
            catalog_group.create_dataset("cluster_id", data=cluster_ids)
            catalog_group.create_dataset("cluster_sample_start", data=np.cumsum(counts) - counts)
            catalog_group.create_dataset("cluster_sample_count", data=counts)
            index_group = f.create_group("index")
            index_group.create_dataset("cluster_id", data=np.repeat(cluster_ids, counts))
            for col in columns:
                values = [s[col] for s in sources if len(s)]
                index_group.create_dataset(col, data=np.concatenate(values) if values else np.zeros(0))


    def create_cluster_ensemble(self, cluster_list, sources_lists, cluster_ensemble_id=0):
        from astropy.table import Table
        import clmm
        bin_units = self.config["bin_units"].lower()
        # load cluster shear catalog using similar astropy table set up as cluster catalog
        
        # Create empty cluster ensemble 
        cluster_ensemble = clmm.ClusterEnsemble(cluster_ensemble_id) 
        
        # Loop through clusters and calculate the profiles
        ncluster = len(cluster_list)
        print('Ncluster', ncluster)

        # Number of clusters skipped because they have no background sources
        n_skipped = 0

        for cluster_index in range(ncluster) :

            # Select subset of background shear information for this particular cluster

            # Clusters outside the shear footprint (or in masked regions) have no
            # background sources -- skip them rather than failing on empty arrays
            if len(sources_lists[cluster_index]) == 0:
                n_skipped += 1
                continue

            # To use CLMM, need to have galaxy table in clmm.GCData type
            #print(sources_lists[cluster_index][0])
            sources = sources_lists[cluster_index]
            distance, weights, tangential_comp, cross_comp = sources["distance_arcmin"], sources["weight"], sources["tangential_comp"], sources["cross_comp"]
            print("dist:", distance.min(), distance.max())
            print("weights:", weights.min(), weights.max())
            print("tangential_comp:", tangential_comp.min(), tangential_comp.max())
            galcat = clmm.GCData(
                Table(
                    {
                        "tangential_comp_clmm": tangential_comp,
                        "cross_comp_clmm": cross_comp,
                        "weight_clmm": weights,
                        "distance_arcmin": distance,
                    }
                ),
                meta={"coordinate_system": self.coordinate_system}
            )
            #print(galcat.meta)
            galcat['theta'] = galcat['distance_arcmin']*np.pi/(60*180) # CLMM galcat requires a column called "theta" in radians
            galcat['z'] = np.zeros(len(galcat)) # clmm needs a column named 'z' but all computation have been done 
                                                # in source_select_compute --> don't need it here, filling dummy array
            
            # Instantiating a CLMM galaxy cluster object

            gc_object = clmm.GalaxyCluster(
                int(cluster_list["id"][cluster_index]),
                cluster_list["ra"][cluster_index],
                cluster_list["dec"][cluster_index],
                cluster_list["redshift"][cluster_index],
                galcat,
            )
            gc_object.richness = cluster_list["richness"][cluster_index]

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
        print("cluster ensemble created")

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
            # the redshift and richness edges of each bin
            bin_attrs = {key: dict(group.attrs) for key, group in f["cluster_bin"].items()}

            
        return clusters_in_bins, bin_attrs

    def make_ensemble(self, clusters_in_bins, bin_attrs, clusters_list_of_source_lists):
        print("Making ensemble")
        binned_cluster_stack = {}
        for key, clusters in clusters_in_bins.items():
            #print(clusters)
            if  len(clusters)>1:
                cluster_stack = self.create_cluster_ensemble(clusters, clusters_list_of_source_lists[key], cluster_ensemble_id=key)
            else :
                cluster_stack = None
            print('cl_ensemble_created')
            
            #dict(dset_out[i].attrs), dset_out[i]['redshift'][:].size) 
            
            binned_cluster_stack[key]={'cluster_bin_edges':bin_attrs[key], 'n_cl':len(clusters), 'clmm_cluster_ensemble':cluster_stack, 'profile_type': self.profile_type}

        return binned_cluster_stack
