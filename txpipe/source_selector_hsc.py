from .base_stage import PipelineStage
from .source_selector import TXSourceSelector
from .data_types import ShearCatalog, YamlFile, PhotozPDFFile, TomographyCatalog, HDFFile, TextFile
from .utils import SourceNumberDensityStats
from .utils.calibration_tools import read_shear_catalog_type
from .utils.calibration_tools import metacal_variants, band_variants, ParallelCalibratorMetacal, ParallelCalibratorNonMetacal
import numpy as np
import warnings

class TXHSCSourceSelector(TXSourceSelector):

    """
    This pipeline stage selects objects to be used
    as the source sample for the shear-shear and
    shear-position calibrations.  It applies some
    general cuts based on the flags that metacal
    gives for the objects, and size and S/N cuts
    based on the configuration file.

    It also splits those objects into tomographic
    bins according to the choice the user makes
    in the input file, from the information in the
    photo-z PDF file.

    Once these selections are made it constructs
    the quantities needed to calibrate each bin -
    this consists of two shear response quantities.

    TODO: add option to use lensfit catalogs, which 
    would be much much simpler.
    """

    name='TXHSCSourceSelector'

    inputs = [
        ('shear_catalog', ShearCatalog),
        ('calibration_table', TextFile),
        ('photometry_catalog', HDFFile),  # this is to get the photo-z, does not necessarily need it
    ]

    outputs = [
        ('shear_tomography_catalog', TomographyCatalog)
    ]

    config_options = {
        'input_pz': False,
        'bands': 'riz', # bands from metacal to use
        'verbose': False,
        'delta_gamma': float,
        'chunk_rows':10000,
        'source_zbin_edges':[float],
        'random_seed': 42,
        'shear_prefix': 'mcal_',
    }

    def run(self):
        """
        Run the analysis for this stage.
        
         - Collect the list of columns to read
         - Create iterators to read chunks of those columns
         - Loop through chunks:
            - select objects for each bin
            - write them out
            - accumulate selection bias values
         - Average the selection biases
         - Write out biases and close the output
        """
        import astropy.table
        import sklearn.ensemble

        # Suppress some warnings from numpy that are not relevant
        original_warning_settings = np.seterr(all='ignore')  

        # Are we using a metacal or lensfit catalog?
        shear_catalog_type = read_shear_catalog_type(self)

        # The output file we will put the tomographic
        # information into
        output_file = self.setup_output()

        # various config options
        bands = self.config['bands']
        chunk_rows = self.config['chunk_rows']
        delta_gamma = self.config['delta_gamma']
        
        shear_prefix = self.config['shear_prefix']


        # Columns we need from the shear catalog, will need to modify for lensfit catalogs
        shear_cols = [f'{shear_prefix}flags', f'{shear_prefix}psf_T_mean']
        shear_cols += band_variants(bands,
                                    f'{shear_prefix}mag',
                                    f'{shear_prefix}mag_err',
                                    shear_catalog_type=shear_catalog_type)

        if shear_catalog_type == 'metacal':
            shear_cols += metacal_variants('mcal_T', 'mcal_s2n', 'mcal_g1', 'mcal_g2')
        else:
            shear_cols += ['T', 's2n', 'g1', 'g2','weight','m','c1','c2','sigma_e']

        if self.config['input_pz'] and self.config['shear_catalog_type']=='metacal':
            shear_cols += ['mean_z']
            shear_cols += ['mean_z_1p']
            shear_cols += ['mean_z_1m']
            shear_cols += ['mean_z_2p']
            shear_cols += ['mean_z_2m']
        elif self.config['input_pz'] and self.config['shear_catalog_type']!='metacal':
            shear_cols += ['mean_z']
        elif self.config['true_z']:
            shear_cols += ['redshift_true']
        else:
            # Build a classifier used to put objects into tomographic bins
            classifier, features = self.build_tomographic_classifier()

            # this bit is for metacal if we want to use it later

        # Input data.  These are iterators - they lazily load chunks
        # of the data one by one later when we do the for loop.
        # This code can be run in parallel, and different processes will
        # each get different chunks of the data
        iter_shear = self.iterate_hdf('shear_catalog', 'shear', shear_cols, chunk_rows)

        # We will collect the selection biases for each bin
        # as a matrix.  We will collect together the different
        # matrices for each chunk and do a weighted average at the end.
        nbin_source = len(self.config['source_zbin_edges'])-1

        selection_biases = []
        number_density_stats = SourceNumberDensityStats(nbin_source, comm=self.comm,shear_type=self.config['shear_catalog_type'])

        if shear_catalog_type == 'metacal':
            calibrators = [ParallelCalibratorMetacal(self.select, delta_gamma) for i in range(nbin_source)]
        else:
            calibrators = [ParallelCalibratorNonMetacal(self.select) for i in range(nbin_source)]

        # Loop through the input data, processing it chunk by chunk
        for (start, end, shear_data) in iter_shear:
            print(f"Process {self.rank} running selection for rows {start:,}-{end:,}")

            if self.config['true_z'] or self.config['input_pz']:
                pz_data = self.apply_simple_redshift_cut(shear_data)

            else:
                # Select most likely tomographic source bin
                pz_data = self.apply_classifier(classifier, features, shear_data)

            # Combine this selection with size and snr cuts to produce a source selection
            # and calculate the shear bias it would generate
            tomo_bin, R, counts = self.calculate_tomography(pz_data, shear_data, calibrators)

            # Save the tomography for this chunk
            self.write_tomography(output_file, start, end, tomo_bin, R)

            # Accumulate information on the number counts and the selection biases.
            # These will be brought together at the end.
            number_density_stats.add_data(shear_data, tomo_bin)  # check this

        # Do the selection bias averaging and output that too.
        self.write_global_values(output_file, calibrators, number_density_stats)

        # Save and complete
        output_file.close()

        # Restore the original warning settings in case we are being called from a library
        np.seterr(**original_warning_settings)

    def setup_output(self):
        """
        Set up the output data file.

        Creates the data sets and groups to put module output
        in the shear_tomography_catalog output file.
        """
        n = self.open_input('shear_catalog')['shear/ra'].size
        zbins = self.config['source_zbin_edges']
        nbin_source = len(zbins)-1

        outfile = self.open_output('shear_tomography_catalog', parallel=True)
        group = outfile.create_group('tomography')
        group.create_dataset('source_bin', (n,), dtype='i')
        group.create_dataset('source_counts', (nbin_source,), dtype='i')
        group.create_dataset('sigma_e', (nbin_source,), dtype='f')
        group.create_dataset('N_eff', (nbin_source,), dtype='f')

        group.attrs['nbin_source'] = nbin_source
        for i in range(nbin_source):
            group.attrs[f'source_zmin_{i}'] = zbins[i]
            group.attrs[f'source_zmax_{i}'] = zbins[i+1]

        #group = outfile.create_group('multiplicative_bias')  # why is this called "multiplicative_bias"?
        if self.config['shear_catalog_type']=='metacal':
            group = outfile.create_group('metacal_response') 
            group.create_dataset('R_gamma', (n,2,2), dtype='f')
            group.create_dataset('R_S', (nbin_source,2,2), dtype='f')
            group.create_dataset('R_gamma_mean', (nbin_source,2,2), dtype='f')
            group.create_dataset('R_total', (nbin_source,2,2), dtype='f')
        else:
            group = outfile.create_group('response') 
            group.create_dataset('R', (n,), dtype='f')
            group.create_dataset('K', (nbin_source,), dtype='f')
            group.create_dataset('C', (nbin_source,1,2), dtype='f')
            group.create_dataset('R_mean', (nbin_source,), dtype='f')

        return outfile

    def write_tomography(self, outfile, start, end, source_bin, R):
        """
        Write out a chunk of tomography and response.

        Parameters
        ----------


        outfile: h5py.File

        start: int
            The index into the output this chunk starts at

        end: int
            The index into the output this chunk ends at

        tomo_bin: array of shape (nrow,)
            The bin index for each output object

        R: array of shape (nrow,2,2)
            Multiplicative bias calibration factor for each object


        """
        group = outfile['tomography']
        group['source_bin'][start:end] = source_bin
        if self.config['shear_catalog_type']=='metacal':
            group = outfile['metacal_response']
            group['R_gamma'][start:end,:,:] = R
        else:
            group = outfile['response']
            group['R'][start:end] = R

    def write_global_values(self, outfile, calibrators, number_density_stats):
        """
        Write out overall selection biases

        Parameters
        ----------

        outfile: h5py.File

        S: array of shape (nbin,2,2)
            Selection bias matrices
        """
        nbin_source = len(calibrators)

        R = np.zeros((nbin_source, 2, 2))
        S = np.zeros((nbin_source, 2, 2))
        K = np.zeros(nbin_source)
        C = np.zeros((nbin_source,1,2))
        N = np.zeros(nbin_source)
        R_scalar = np.zeros(nbin_source)

        # this needs fixing
        sigma_e = number_density_stats.collect()

        for i, cal in enumerate(calibrators):
            if self.config['shear_catalog_type']=='metacal':
                R[i], S[i], N[i] = cal.collect(self.comm)
                sigma_e[i] /= 0.5*(R[i,0,0] + R[i,1,1])
            else:
                R_scalar[i], K[i], C[i], N[i] = cal.collect(self.comm)
                sigma_e[i] /= 0.5*(R_scalar[i] + R_scalar[i])
        

        if self.rank==0:
            if self.config['shear_catalog_type']=='metacal':
                group = outfile['metacal_response']
                group['R_S'][:,:,:] = S
                group['R_gamma_mean'][:,:,:] = R
                group['R_total'][:,:,:] = R + S
                group = outfile['tomography']
                group['sigma_e'][:] = sigma_e
                # These are the same in metacal
                group['source_counts'][:] = N
                group['N_eff'][:] = N
            else:
                group = outfile['response']
                group['R_mean'][:] = R_scalar
                group['C'][:] = C
                group['K'][:] = K
                group = outfile['tomography']
                group['sigma_e'][:] = sigma_e
                # These are the same in metacal
                group['source_counts'][:] = N
                group['N_eff'][:] = N 

    
    def select(self, data, bin_index):
        """
        Apply magnitude cut to the HSC data as described in Tab. 4 of Mandelbaum et al., 2018.
        Parameters
        ----------
        data
        bin_index

        Returns
        -------

        """

        verbose = self.config['verbose']
        # Magnitude cut
        mag_i_cut = self.config['i_hi_cut']

        zbin = data['zbin']

        mag_i = data['mag_i']
        #a_i = data['a_i']
        a_i = np.zeros(len(mag_i))
        n0 = len(mag_i)
        sel = ((mag_i - a_i) <= mag_i_cut)
        f1 = sel.sum() / n0
        sel &= zbin==bin_index
        f2 = sel.sum() / n0
        
        variant = data.suffix
        if verbose:
            print(f"Bin {bin_index} ({variant}) {f1:.2%} magnitude cut, {f2:.2%} redshift")

        return sel


def flatten_list(lst):
    return [item for sublist in lst for item in sublist]


if __name__ == '__main__':
    PipelineStage.main()

