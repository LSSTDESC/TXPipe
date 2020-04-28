from .base_stage import PipelineStage
from .data_types import YamlFile, TomographyCatalog, HDFFile, TextFile
from .utils import LensNumberDensityStats
import numpy as np
import warnings

class TXLensSelector(PipelineStage):
    """
    This pipeline stage selects objects to be used
    as the lens sample for the galaxy clustering and
    shear-position calibrations.
    """

    name='TXLensSelector'

    inputs = [
        ('calibration_table', TextFile),
        ('photometry_catalog', HDFFile),
    ]

    outputs = [
        ('lens_tomography_catalog', TomographyCatalog)
    ]

    config_options = {
        'verbose': False,
        'chunk_rows':10000,
        'lens_zbin_edges':[float],
        # Mag cuts
        # Default photometry cuts based on the BOSS Galaxy Target Selection:                                                     
        # http://www.sdss3.org/dr9/algorithms/boss_galaxy_ts.php                                           
        'cperp_cut':0.2,
        'r_cpar_cut':13.5,
        'r_lo_cut':16.0,
        'r_hi_cut':19.6,
        'i_lo_cut':17.5,
        'i_hi_cut':19.9,
        'r_i_cut':2.0,
        'random_seed': 42,
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

        # The output file we will put the tomographic
        # information into
        output_file = self.setup_output()

        # various config options
        chunk_rows = self.config['chunk_rows']

        phot_cols = ['i_mag','r_mag','g_mag', 'redshift_true']

        # Input data.  These are iterators - they lazily load chunks
        # of the data one by one later when we do the for loop.
        # This code can be run in parallel, and different processes will
        # each get different chunks of the data 
        iter_phot = self.iterate_hdf('photometry_catalog', 'photometry', phot_cols, chunk_rows)

        # We will collect the selection biases for each bin
        # as a matrix.  We will collect together the different
        # matrices for each chunk and do a weighted average at the end.
        nbin_lens = len(self.config['lens_zbin_edges'])-1

        number_density_stats = LensNumberDensityStats(nbin_lens, self.comm)

        # Loop through the input data, processing it chunk by chunk
        for (start, end, phot_data) in iter_phot:
            print(f"Process {self.rank} running selection for rows {start:,}-{end:,}")

            pz_data = self.apply_simple_redshift_cut(phot_data)

            # Select lens bin objects
            lens_gals = self.select_lens(phot_data)

            # Combine this selection with size and snr cuts to produce a source selection
            # and calculate the shear bias it would generate
            tomo_bin, counts = self.calculate_tomography(pz_data, phot_data, lens_gals)

            # Save the tomography for this chunk
            self.write_tomography(output_file, start, end, tomo_bin)

            # Accumulate information on the number counts and the selection biases.
            # These will be brought together at the end.
            number_density_stats.add_data(tomo_bin)

        # Do the selection bias averaging and output that too.
        self.write_global_values(output_file, number_density_stats)

        # Save and complete
        output_file.close()

        # Restore the original warning settings in case we are being called from a library
        np.seterr(**original_warning_settings)


    def apply_simple_redshift_cut(self, phot_data):

        pz_data = {}

        zz = phot_data[f'redshift_true']

        pz_data_v = np.zeros(len(zz), dtype=int) -1
        for zi in range(len(self.config['lens_zbin_edges'])-1):
            mask_zbin = (zz>=self.config['lens_zbin_edges'][zi]) & (zz<self.config['lens_zbin_edges'][zi+1])
            pz_data_v[mask_zbin] = zi
            
        pz_data[f'zbin'] = pz_data_v

        return pz_data


    def setup_output(self):
        """
        Set up the output data file.

        Creates the data sets and groups to put module output
        in the tomography_catalog output file.
        """
        n = self.open_input('photometry_catalog')['photometry/ra'].size
        nbin_lens = len(self.config['lens_zbin_edges'])-1

        outfile = self.open_output('lens_tomography_catalog', parallel=True)
        group = outfile.create_group('tomography')
        group.create_dataset('lens_bin', (n,), dtype='i')
        group.create_dataset('lens_counts', (nbin_lens,), dtype='i')

        group.attrs['nbin_lens'] = nbin_lens
        group.attrs[f'lens_zbin_edges'] = self.config['lens_zbin_edges']

        return outfile

    def write_tomography(self, outfile, start, end, lens_bin):
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
        group['lens_bin'][start:end] = lens_bin

    def write_global_values(self, outfile, number_density_stats):
        """
        Write out overall selection biases

        Parameters
        ----------

        outfile: h5py.File

        """
        lens_counts = number_density_stats.collect()


        if self.rank==0:
            group = outfile['tomography']
            group['lens_counts'][:] = lens_counts


    def select_lens(self, phot_data):
        """Photometry cuts based on the BOSS Galaxy Target Selection:
        http://www.sdss3.org/dr9/algorithms/boss_galaxy_ts.php
        """
        mag_i = phot_data['i_mag']
        mag_r = phot_data['r_mag']
        mag_g = phot_data['g_mag']

        # Mag cuts 
        cperp_cut_val = self.config['cperp_cut']
        r_cpar_cut_val = self.config['r_cpar_cut']
        r_lo_cut_val = self.config['r_lo_cut']
        r_hi_cut_val = self.config['r_hi_cut']
        i_lo_cut_val = self.config['i_lo_cut']
        i_hi_cut_val = self.config['i_hi_cut']
        r_i_cut_val = self.config['r_i_cut']

        n = len(mag_i)
        # HDF does not support bools, so we will prepare a binary array
        # where 0 is a lens and 1 is not
        lens_gals = np.repeat(0,n)

        cpar = 0.7 * (mag_g - mag_r) + 1.2 * ((mag_r - mag_i) - 0.18)
        cperp = (mag_r - mag_i) - ((mag_g - mag_r) / 4.0) - 0.18
        dperp = (mag_r - mag_i) - ((mag_g - mag_r) / 8.0)

        # LOWZ
        cperp_cut = np.abs(cperp) < cperp_cut_val #0.2
        r_cpar_cut = mag_r < r_cpar_cut_val + cpar / 0.3
        r_lo_cut = mag_r > r_lo_cut_val #16.0
        r_hi_cut = mag_r < r_hi_cut_val #19.6

        lowz_cut = (cperp_cut) & (r_cpar_cut) & (r_lo_cut) & (r_hi_cut)

        # CMASS
        i_lo_cut = mag_i > i_lo_cut_val #17.5
        i_hi_cut = mag_i < i_hi_cut_val #19.9
        r_i_cut = (mag_r - mag_i) < r_i_cut_val #2.0
        #dperp_cut = dperp > 0.55 # this cut did not return any sources...

        cmass_cut = (i_lo_cut) & (i_hi_cut) & (r_i_cut)

        # If a galaxy is a lens under either LOWZ or CMASS give it a zero
        lens_mask =  lowz_cut | cmass_cut
        lens_gals[lens_mask] = 1

        return lens_gals

    def calculate_tomography(self, pz_data, phot_data, lens_gals):
    
        nbin = len(self.config['lens_zbin_edges'])-1
        n = len(phot_data['i_mag'])

        # The main output data - the tomographic
        # bin index for each object, or -1 for no bin.
        tomo_bin = np.repeat(-1, n)

        # We also keep count of total count of objects in each bin
        counts = np.zeros(nbin, dtype=int)

        print(pz_data)
        print(lens_gals)
        for i in range(nbin):
            sel_00 = (pz_data['zbin']==i)*(lens_gals==1)
            tomo_bin[sel_00] = i
            counts[i] = sel_00.sum()

        return tomo_bin, counts


def flatten_list(lst):
    return [item for sublist in lst for item in sublist]


if __name__ == '__main__':
    PipelineStage.main()


