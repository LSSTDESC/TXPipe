import numpy as np
import treecorr
import h5py
import sacc
from .base_stage import PipelineStage
from .data_types import HDFFile, SACCFile


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
        'richness_bin_edges': [20, 50, 200],   # Edges of richness bins
        'nbins': 20,                           # Number of angular bins
        'min_sep': 0.1,                        # Minimum separation [arcmins]
        'max_sep': 250.0,                      # Maximum separation [arcmins]
        'units': 'arcmin',                     # Units for the separations
        'binning_scale': 'Log'                 # 'Log', 'Linear', 'TwoD'
    }

    def load_data(self, file_path):
        """
        Load data from catalog (hdf5 file).
        Read ra, dec and the richness proxy.
        """
        try:
            data = h5py.File(file_path, 'r+')
            ra = data['ra'][:]
            dec = data['dec'][:]
            richness = data['richness'][:]
        except ValueError:
            raise ValueError('Catalog format not recognized. Use HDF5 format.')
        return ra, dec, richness

    def split_data_by_richness(self, ra, dec, richness):
        """
        Split coordinates into richness bins based on the config.
        """
        richness_bin_edges = self.config['richness_bin_edges']
        num_bins = len(richness_bin_edges) - 1

        ra_bins = []
        dec_bins = []

        # Iterate over richness bin edges to create bins
        for i in range(num_bins):
            indices = ((richness >= richness_bin_edges[i])
                       & (richness < richness_bin_edges[i+1]))
            ra_bins.append(ra[indices])
            dec_bins.append(dec[indices])

        return ra_bins, dec_bins

    def create_catalog(self, ra, dec):
        """
        Create a TreeCorr catalog for the given data.
        """
        catalog = treecorr.Catalog(ra=ra, dec=dec,
                                   ra_units='deg', dec_units='deg')
        return catalog

    def measure_auto_correlation(self, data_cat, rand_cat, bin_config):
        """
        Measure the auto-correlation function for one richness bin.
        """
        dd = treecorr.NNCorrelation(**bin_config)
        rr = treecorr.NNCorrelation(**bin_config)
        dr = treecorr.NNCorrelation(**bin_config)

        dd.process(data_cat)
        rr.process(rand_cat)
        dr.process(data_cat, rand_cat)

        xi, varxi = dd.calculateXi(rr=rr, dr=dr)
        return xi, varxi, dd.rnom

    def measure_cross_correlation(self, data_cat1, data_cat2,
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

        for result in results:
            meanr, xi, varxi, tracer1, tracer2 = result

            # Add tracers (placeholders z/Nz values)
            s.add_tracer('NZ', tracer1, np.array([1]), np.array([1]))
            s.add_tracer('NZ', tracer2, np.array([1]), np.array([1]))

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
        data_ra, data_dec, data_richness = self.load_data(
            self.config['cluster_data_catalog'])

        rand_ra, rand_dec, rand_richness = self.load_data(
            self.config['cluster_random_catalog'])

        # Split data and random catalogs by richness bins
        data_ra_bins, data_dec_bins = self.split_data_by_richness(
            data_ra, data_dec, data_richness)

        rand_ra_bins, rand_dec_bins = self.split_data_by_richness(
            rand_ra, rand_dec, rand_richness)

        # Create TreeCorr data and random catalogs for all bins
        data_cats = [self.create_catalog(ra, dec)
                     for ra, dec in zip(data_ra_bins, data_dec_bins)]

        rand_cats = [self.create_catalog(ra, dec)
                     for ra, dec in zip(rand_ra_bins, rand_dec_bins)]

        # Configuration for angular bins
        bin_config = {
            'nbins': self.config['nbins'],
            'min_sep': self.config['min_sep'],
            'max_sep': self.config['max_sep'],
            'sep_units': self.config['units'],
            'bin_slop': 0,  # Set bin_slop to 0 for accurate binning
            'bin_type': self.config['binning_scale']}

        # Store results for saving
        results = []

        # Measure auto-correlations
        for i, (data_cat, rand_cat) in enumerate(zip(data_cats, rand_cats)):
            tracer_name = f'richness_bin_{i+1}'
            xi, varxi, meanr = self.measure_auto_correlation(
                data_cat, rand_cat, bin_config)

            results.append((meanr, xi, varxi, tracer_name, tracer_name))

        # Measure cross-correlations
        num_bins = len(data_cats)
        for i in range(num_bins):
            for j in range(i+1, num_bins):
                tracer_name1 = f'richness_bin_{i+1}'
                tracer_name2 = f'richness_bin_{j+1}'
                xi, varxi, meanr = self.measure_cross_correlation(
                    data_cats[i], data_cats[j],
                    rand_cats[i], rand_cats[j], bin_config)

                results.append((meanr, xi, varxi, tracer_name1, tracer_name2))

        # Save results to a SACC file
        self.save_to_sacc(results, self.get_output("cluster_twopoint_real"))


if __name__ == "__main__":
    PipelineStage.main()
