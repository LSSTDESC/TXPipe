import numpy as np
from ...base_stage import PipelineStage
from ...data_types import PickleFile, SACCFile
import pickle as pkl

class CLClusterSACC(PipelineStage):
    name = "CLClusterSACC"
    inputs = [
        ("cluster_profiles",  PickleFile),
    ]

    outputs = [
        ("cluster_sacc_catalog",  SACCFile),
    ]

    config_options = {
        #radial bin definition
        "r_min" : 0.5, #in Mpc
        "r_max" : 5.0, #in Mpc
    }

    def run(self):
        import sacc
        print(self.get_input("cluster_profiles"))
        data = pkl.load(open(self.get_input("cluster_profiles"), "rb"))            
        print(data)
        my_configs = self.config
        survey_name = my_configs['survey_name']
        area = my_configs['area']
        output_filename = self.get_output("cluster_sacc_catalog", final_name=True)

        sacc_obj = sacc.Sacc()
        self.add_tracers(sacc_obj, data, survey_name, area)
        self.add_counts_data(sacc_obj, data, survey_name)
        self.add_deltasigma_data(sacc_obj, data, survey_name)
        self.add_covariance_data(sacc_obj, data)

        sacc_obj.to_canonical_order()
        sacc_obj.save_fits(output_filename, overwrite=True)

    def transform_bin_string(self, bin_string: str) -> tuple:
        """
        Transforms a string like 'bin_zbin_X_richbin_Y' into ('bin_z_X', 'bin_rich_Y').
        """
        import re
        zbin_match = re.search(r'zbin_(\d+)', bin_string)
        richbin_match = re.search(r'richbin_(\d+)', bin_string)
        
        if zbin_match and richbin_match:
            return f'bin_z_{zbin_match.group(1)}', f'bin_rich_{richbin_match.group(1)}'
        
        raise ValueError("Input string is not in the expected format.")

    def get_bins(self, data: dict):
        """
        Retrieves and organizes the bin edges and centers.
        Returns:
            tuple: (z_bins, rich_bins, radius_bins)
        """
        bin_z_dict, bin_rich_dict, bin_radius_dict = {}, {}, {}

        for bin_comb, bin_data in data.items():
            bin_z, bin_rich = self.transform_bin_string(bin_comb)
            z_edges = (bin_data['cluster_bin_edges']['z_min'], bin_data['cluster_bin_edges']['z_max'])
            rich_edges = (bin_data['cluster_bin_edges']['rich_min'], bin_data['cluster_bin_edges']['rich_max'])
            bin_z_dict[bin_z] = z_edges
            bin_rich_dict[bin_rich] = rich_edges

        radius_centers = np.array(data['bin_zbin_0_richbin_0']['clmm_cluster_ensemble'].stacked_data['radius'])
        rmin = self.config_options['r_min']
        rmax = self.config_options['r_max'] 
        radius_edges = np.logspace(np.log10(rmin), np.log10(rmax), len(radius_centers) + 1)
        for i in range(len(radius_edges) - 1):
            bin_radius_dict[f'radius_{i}'] = (radius_edges[i], radius_edges[i+1], radius_centers[i])

        return bin_z_dict, bin_rich_dict, bin_radius_dict

    def add_tracers(self, sacc_obj, data: dict, survey_name: str, area: float):
        """
        Adds tracer data to the SACC object.
        """
        import sacc
        z_bins, rich_bins, radius_bins = self.get_bins(data)

        for bin_comb in data:
            bin_z, bin_rich = self.transform_bin_string(bin_comb)
            sacc_obj.add_tracer("bin_z", bin_z, *z_bins[bin_z])
            sacc_obj.add_tracer("bin_richness", bin_rich, np.log10(rich_bins[bin_rich][0]), np.log10(rich_bins[bin_rich][1]))

        for radius_bin, radius_edges in radius_bins.items():
            sacc_obj.add_tracer("bin_radius", radius_bin, *radius_edges)

        sacc_obj.add_tracer("survey", survey_name, area)

    def add_counts_data(self, sacc_obj, data: dict, survey_name: str):
        """
        Adds cluster count data to the SACC object.
        """
        import sacc
        cluster_count = sacc.standard_types.cluster_counts

        for bin_comb, bin_data in data.items():
            bin_z, bin_rich = self.transform_bin_string(bin_comb)
            sacc_obj.add_data_point(cluster_count, (survey_name, bin_rich, bin_z), int(bin_data['n_cl']))

    def add_deltasigma_data(self, sacc_obj, data: dict, survey_name: str):
        """
        Adds cluster shear (delta sigma) data to the SACC object.
        """
        import sacc
        cluster_shear = sacc.standard_types.cluster_shear
        _, _, radius_bins = self.get_bins(data)

        for bin_comb, bin_data in data.items():
            bin_z, bin_rich = self.transform_bin_string(bin_comb)
            for i, bin_radius in enumerate(radius_bins):
                tangential_comp = bin_data['clmm_cluster_ensemble'].stacked_data[i]['tangential_comp']
                sacc_obj.add_data_point(cluster_shear, (survey_name, bin_rich, bin_z, bin_radius), tangential_comp)

    def add_covariance_data(self, sacc_obj, data: dict):
        """
        Adds covariance data to the SACC object.
        """
        import sacc
        cluster_count = sacc.standard_types.cluster_counts
        counts_points = np.array(sacc_obj.get_data_points(cluster_count))
        counts_cov = np.array([point.value for point in counts_points])

        deltasigma_cov = [
            bin_data['clmm_cluster_ensemble'].cov['tan_sc'].diagonal()
            for bin_data in data.values()
        ]

        diag_cov_vector = np.concatenate([counts_cov.flatten(), np.array(deltasigma_cov).flatten()])
        sacc_obj.add_covariance(np.diag(diag_cov_vector))
