import sys
import os
import numpy as np
import treecorr
import h5py
import sacc
from txpipe.base_stage import PipelineStage
from txpipe.data_types import HDFFile, SACCFile


class TXTwoPointCluster(PipelineStage):
    """
    TXPipe task for measuring two-point correlation functions
    of galaxy clusters in an indefinite number of richness bins using TreeCorr.
    """

    name = 'TXTwoPointCluster'

    inputs = [
        ("cluster_data_catalog", HDFFile),
        ("cluster_random_catalog", HDFFile)
    ]

    outputs = [
        ("cluster_twopoint_real", SACCFile)
    ]

    config_options = {
        'redshift_bin_edges': [0.4,0.8,1.2],   # Edges of redshift bins
        'richness_bin_edges': [20, 30, 200],   # Edges of richness bins
        'nbins': 20,                           # Number of angular bins
        'min_sep': 0.1,                        # Minimum separation [arcmins]
        'max_sep': 250.0,                      # Maximum separation [arcmins]
        'units': 'arcmin',                     # Units for the separations
        'binning_scale': 'Log'                 # 'Log', 'Linear', 'TwoD'
    }

    def load_data(self, file_path):
        """
        Load data from catalog (hdf5 file).
        Read ra and dec in each richness and redshift bin.
        """
        # dictionaries for all the tomographic bins
        ra  = {}
        dec = {}

        # read data and extract coordinates in each bin
        data = h5py.File(file_path, 'r+')
        for zbin_richbin in data['cluster_bin'].keys():
            ra[zbin_richbin]  = data['cluster_bin'][zbin_richbin]['ra'][:]
            dec[zbin_richbin] = data['cluster_bin'][zbin_richbin]['dec'][:]

        return ra, dec


    def create_catalog(self, ra, dec):
        """
        Create a TreeCorr catalog for the given data.
        """
        catalog = treecorr.Catalog(ra=ra, dec=dec,
                                   ra_units='deg', dec_units='deg')
        return catalog


    def measure_correlation(self, data_cat1, data_cat2,
                                  rand_cat1, rand_cat2, bin_config):
        """
        Measure the cross-correlation function between two richness bins.
        """
        dd = treecorr.NNCorrelation(**bin_config)
        rr = treecorr.NNCorrelation(**bin_config)
        dr = treecorr.NNCorrelation(**bin_config)
        rd = treecorr.NNCorrelation(**bin_config)

        dd.process(data_cat1, data_cat2)
        rr.process(rand_cat1, rand_cat2)
        dr.process(data_cat1, rand_cat2)
        rd.process(data_cat2, rand_cat1)

        xi, varxi = dd.calculateXi(rr=rr, dr=dr, rd=rd)

        return xi, varxi, dd.rnom

    def save_to_sacc(self, results, output_file):
        """
        Save the correlation function results to a SACC file.
        """
        s = sacc.Sacc()

        for i,result in enumerate(results):
            meanr, xi, varxi, tracer1, tracer2, log10rich1, log10rich2 = result
            print(i,tracer1,tracer2)

            # Add tracers
            s.add_tracer('bin_richness', tracer1, log10rich1[0], log10rich1[1])
            s.add_tracer('bin_richness', tracer2, log10rich2[0], log10rich2[1])
            
    
            # Add data points
            for i in range(len(meanr)):
                s.add_data_point(
                    data_type=sacc.standard_types.cluster_density_xi,
                    tracers=(tracer1, tracer2),
                    value=xi[i],
                    error=varxi[i]**0.5,
                    theta=meanr[i]
                )

        s.save_fits(output_file, overwrite=True)

    def run(self):
        """
        Run the analysis for this stage.
        """

        
        # Load data and random catalogs 
        data_ra_bins, data_dec_bins = self.load_data(
            self.config['cluster_data_catalog'])

        rand_ra_bins, rand_dec_bins = self.load_data(
            self.config['cluster_random_catalog'])


        # Create TreeCorr data and random catalogs for each redshift and richness bin combination
        data_cats = {}
        rand_cats = {}
        for zbin_richbin in data_ra_bins.keys():

            data_cats[zbin_richbin] = [self.create_catalog(data_ra_bins[zbin_richbin], 
                                                           data_dec_bins[zbin_richbin])]
    
            rand_cats[zbin_richbin] = [self.create_catalog(rand_ra_bins[zbin_richbin], 
                                                           rand_dec_bins[zbin_richbin])]

        
        # Configuration for angular bins
        bin_config = {
            'nbins': self.config['nbins'],
            'min_sep': self.config['min_sep'],
            'max_sep': self.config['max_sep'],
            'sep_units': self.config['units'],
            'bin_slop': 0,  # Set bin_slop to 0 for accurate binning
            'bin_type': self.config['binning_scale']}

        num_zbins    = len(self.config['redshift_bin_edges'])-1
        num_richbins = len(self.config['richness_bin_edges'])-1
        bins_log10rich = np.log10(self.config['richness_bin_edges'])   
        
        # Store results for saving
        results = []

        # Measure cross-correlations
        num_bins = len(data_cats)
        for zbin in range(num_zbins):
            for richbin_i in range(num_richbins):
                for richbin_j in range(richbin_i, num_richbins):

                    # save bin edges to write in the tracer object
                    log10rich1 = [bins_log10rich[richbin_i],bins_log10rich[richbin_i+1]]
                    log10rich2 = [bins_log10rich[richbin_j],bins_log10rich[richbin_j+1]]

                    # tracer names
                    tracer_1 = f'zbin_{zbin}_richbin_{richbin_i}'
                    tracer_2 = f'zbin_{zbin}_richbin_{richbin_j}'

                    # measure correlation
                    xi, varxi, meanr = self.measure_correlation(
                        data_cats['bin_'+tracer_1], data_cats['bin_'+tracer_2],
                        rand_cats['bin_'+tracer_1], rand_cats['bin_'+tracer_2], bin_config)
    
                    results.append((meanr, xi, varxi, 
                                    tracer_1, tracer_2, 
                                    log10rich1,log10rich2))

        # Save results to a SACC file
        self.save_to_sacc(results, self.get_output("cluster_twopoint_real"))


if __name__ == "__main__":
    PipelineStage.main()
