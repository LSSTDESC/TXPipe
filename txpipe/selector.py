from .base_stage import PipelineStage
from .data_types import MetacalCatalog, YamlFile, PhotozPDFFile, TomographyCatalog, HDFFile, TextFile
from .utils import NumberDensityStats
from .utils.metacal import metacal_variants, metacal_band_variants, ParallelCalibrator
import numpy as np
import warnings

class TXSelector(PipelineStage):
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
    """

    name='TXSelector'

    inputs = [
        ('shear_catalog', MetacalCatalog),
        ('calibration_table', TextFile),
        ('photometry_catalog', HDFFile),
    ]

    outputs = [
        ('tomography_catalog', TomographyCatalog)
    ]

    config_options = {
        'input_pz': False,
        'bands': 'riz', # bands from metacal to use
        'verbose': False,
        'T_cut':float,
        's2n_cut':float,
        'delta_gamma': float,
        'chunk_rows':10000,
        'zbin_edges':[float],
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
        bands = self.config['bands']
        chunk_rows = self.config['chunk_rows']
        delta_gamma = self.config['delta_gamma']

        if not self.config['input_pz']:
            # Build a classifier used to put objects into tomographic bins
            classifier, features = self.build_tomographic_classifier()        

        # Columns we need from the photometry data.
        # We use the photometry data to select the lenses.
        # Although this will be one by redmagic soon.
        phot_cols = ['mag_g', 'mag_r', 'mag_i']

        # Columns we need from the shear catalog
        shear_cols = ['mcal_flags', 'mcal_psf_T_mean']
        shear_cols += metacal_band_variants(bands, 'mcal_mag', 'mcal_mag_err')
        shear_cols += metacal_variants('mcal_T', 'mcal_s2n', 'mcal_g1', 'mcal_g2')
        if self.config['input_pz']:
            shear_cols += ['mean_z']
            shear_cols += ['mean_z_1p']
            shear_cols += ['mean_z_1m']
            shear_cols += ['mean_z_2p']
            shear_cols += ['mean_z_2m']


        # Input data.  These are iterators - they lazily load chunks
        # of the data one by one later when we do the for loop.
        # This code can be run in parallel, and different processes will
        # each get different chunks of the data 
        iter_shear = self.iterate_hdf('shear_catalog', 'metacal', shear_cols, chunk_rows)
        iter_phot = self.iterate_hdf('photometry_catalog', 'photometry', phot_cols, chunk_rows)

        # We will collect the selection biases for each bin
        # as a matrix.  We will collect together the different
        # matrices for each chunk and do a weighted average at the end.
        nbin_source = len(self.config['zbins'])
        nbin_lens = 1

        selection_biases = []
        number_density_stats = NumberDensityStats(nbin_source, nbin_lens, self.comm)
        calibrators = [ParallelCalibrator(self.select, delta_gamma) for i in range(nbin_source)]


        # Loop through the input data, processing it chunk by chunk
        for (start, end, shear_data), (_, _, phot_data) in zip(iter_shear, iter_phot):
            print(f"Process {self.rank} running selection for rows {start:,}-{end:,}")

            if self.config['input_pz']:
                pz_data = self.apply_simple_redshift_cut(shear_data)

            else:
                # Select most likely tomographic source bin
                pz_data = self.apply_classifier(classifier, features, shear_data)

            # Combine this selection with size and snr cuts to produce a source selection
            # and calculate the shear bias it would generate
            tomo_bin, R, counts = self.calculate_tomography(pz_data, shear_data, calibrators)

            # Select lens bin objects
            lens_gals = self.select_lens(phot_data)

            # Save the tomography for this chunk
            self.write_tomography(output_file, start, end, tomo_bin, lens_gals, R, lens_gals)

            # Accumulate information on the number counts and the selection biases.
            # These will be brought together at the end.
            number_density_stats.add_data(shear_data, tomo_bin, lens_gals)

        # Do the selection bias averaging and output that too.
        self.write_global_values(output_file, calibrators, number_density_stats)

        # Save and complete
        output_file.close()

        # Restore the original warning settings in case we are being called from a library
        np.seterr(**original_warning_settings)

    


    def build_tomographic_classifier(self):
        # Load the training data
        # Build the SOM from the training data
        from astropy.table import Table
        from sklearn.ensemble import RandomForestClassifier

        if self.rank > 0:
            classifier = self.comm.bcast(None)
            features = self.comm.bcast(None)
            return classifier, features

        # Load the training data
        training_file = self.get_input("calibration_table")
        training_data_table = Table.read(training_file, format='ascii')

        # Pull out the appropriate columns and combinations of the data
        bands = self.config['bands']
        print(f"Using these bands to train the tomography selector: {bands}")

        # Generate the training data that we will use
        # We record both the name of the column and the data iself
        features = []
        training_data = []
        for b1 in bands[:]:
            # First we use the magnitudes themselves
            features.append(b1)
            training_data.append(training_data_table[b1])
            # We also use the colours as training data, even the redundant ones
            for b2 in bands[:]:
                if b1<b2:
                    features.append(f'{b1}-{b2}')
                    training_data.append(training_data_table[b1] - training_data_table[b2])
        training_data = np.array(training_data).T

        print("Training data for bin classifier has shape ", training_data.shape)

        # Now put the training data into redshift bins
        # We use -1 to indicate that we are outside the desired ranges
        z = training_data_table['sz']
        training_bin = np.repeat(-1, len(z))
        print("Using these bin edges:", self.config['zbin_edges'])
        for i,(zmin, zmax) in enumerate(self.config['zbins']):
            training_bin[(z>zmin) & (z<zmax)] = i
            ntrain_bin = ((z>zmin) & (z<zmax)).sum()
            print(f"Training set: {ntrain_bin} objects in tomographic bin {i}")

        # Can be replaced with any classifier
        classifier = RandomForestClassifier(max_depth=10, max_features=None, n_estimators=20,
            random_state=self.config['random_seed'])
        classifier.fit(training_data, training_bin)

        # Sklearn fitters can be pickled, which means they can also be sent through
        # mpi4py
        if self.is_mpi():
            self.comm.bcast(classifier)
            self.comm.bcast(features)

        return classifier, features


    def apply_classifier(self, classifier, features, shear_data):
        """Apply the classifier to the measured magnitudes
        """
        bands = self.config['bands']
        variants = ['', '_1p', '_2p', '_1m', '_2m']

        pz_data = {}
        
        for v in variants:
            # Pull out the columns that we have trained this bin selection
            # model on.
            data = []
            for f in features:
                # may be a single band
                if len(f) == 1:
                    col = shear_data[f'mcal_mag_{f}{v}']
                # or a colour
                else:
                    b1,b2 = f.split('-')
                    col = shear_data[f'mcal_mag_{b1}{v}'] - shear_data[f'mcal_mag_{b2}{v}']
                if np.all(~np.isfinite(col)):
                    # entire column is NaN.  Hopefully this will get deselected elsewhere
                    col[:] = 30.0
                else:
                    ok = np.isfinite(col)
                    col[~ok] = col[ok].max()
                data.append(col)
            data = np.array(data).T

            # Run the random forest on this data chunk
            pz_data[f'zbin{v}'] = classifier.predict(data)
        return pz_data

    def apply_simple_redshift_cut(self, shear_data):
        variants = ['', '_1p', '_2p', '_1m', '_2m']

        pz_data = {}

        for v in variants:
            zz = shear_data[f'mean_z{v}']

            pz_data_v = np.zeros(len(zz), dtype=int) -1
            for zi in range(len(self.config['zbin_edges'])-1):
                mask_zbin = (zz>=self.config['zbin_edges'][zi]) & (zz<self.config['zbin_edges'][zi+1])
                pz_data_v[mask_zbin] = zi
            
            pz_data[f'zbin{v}'] = pz_data_v

        return pz_data

    def calculate_tomography(self, pz_data, shear_data, calibrators):
        """
        Select objects to go in each tomographic bin and their calibration.

        Parameters
        ----------

        pz_data: table or dict of arrays
            A chunk of input photo-z data containing mean values for each object
        shear_data: table or dict of arrays
            A chunk of input shear data with metacalibration variants.
        """
        delta_gamma = self.config['delta_gamma']
        nbin = len(self.config['zbins'])
        n = len(shear_data['mcal_g1'])

        # The main output data - the tomographic
        # bin index for each object, or -1 for no bin.
        tomo_bin = np.repeat(-1, n)
        R = np.zeros((n, 2, 2))

        # We also keep count of total count of objects in each bin
        counts = np.zeros(nbin, dtype=int)

        data = {**pz_data, **shear_data}

        R[:,0,0] = (data['mcal_g1_1p'] - data['mcal_g1_1m']) / delta_gamma
        R[:,0,1] = (data['mcal_g1_2p'] - data['mcal_g1_2m']) / delta_gamma
        R[:,1,0] = (data['mcal_g2_1p'] - data['mcal_g2_1m']) / delta_gamma
        R[:,1,1] = (data['mcal_g2_2p'] - data['mcal_g2_2m']) / delta_gamma


        for i in range(nbin):
            sel_00 = calibrators[i].add_data(data, i)
            tomo_bin[sel_00] = i
            counts[i] = sel_00.sum()

        return tomo_bin, R, counts



    def setup_output(self):
        """
        Set up the output data file.

        Creates the data sets and groups to put module output
        in the tomography_catalog output file.
        """
        n = self.open_input('shear_catalog')['metacal/ra'].size
        zbins = self.config['zbins']
        nbin_source = len(zbins)
        nbin_lens = 1

        outfile = self.open_output('tomography_catalog', parallel=True)
        group = outfile.create_group('tomography')
        group.create_dataset('source_bin', (n,), dtype='i')
        group.create_dataset('lens_bin', (n,), dtype='i')
        group.create_dataset('source_counts', (nbin_source,), dtype='i')
        group.create_dataset('lens_counts', (nbin_lens,), dtype='i')
        group.create_dataset('sigma_e', (nbin_source,), dtype='f')
        group.create_dataset('N_eff', (nbin_source,), dtype='f')

        group.attrs['nbin_source'] = nbin_source
        group.attrs['nbin_lens'] = nbin_lens
        for i in range(nbin_source):
            group.attrs[f'source_zmin_{i}'] = zbins[i][0]
            group.attrs[f'source_zmax_{i}'] = zbins[i][1]

        group = outfile.create_group('multiplicative_bias')
        group.create_dataset('R_gamma', (n,2,2), dtype='f')
        group.create_dataset('R_S', (nbin_source,2,2), dtype='f')
        group.create_dataset('R_gamma_mean', (nbin_source,2,2), dtype='f')
        group.create_dataset('R_total', (nbin_source,2,2), dtype='f')

        return outfile

    def write_tomography(self, outfile, start, end, source_bin, lens_bin, R, lens_gals):
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
        group['lens_bin'][start:end] = lens_bin
        group = outfile['multiplicative_bias']
        group['R_gamma'][start:end,:,:] = R

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
        N = np.zeros(nbin_source)
        R_scalar = np.zeros(nbin_source)

        sigma_e, lens_counts = number_density_stats.collect()

        for i, cal in enumerate(calibrators):
            R[i], S[i], N[i] = cal.collect(self.comm)
            sigma_e[i] /= 0.5*(R[i,0,0] + R[i,1,1])
        

        if self.rank==0:
            group = outfile['multiplicative_bias']
            group['R_S'][:,:,:] = S
            group['R_gamma_mean'][:,:,:] = R
            group['R_total'][:,:,:] = R + S
            group = outfile['tomography']
            group['lens_counts'][:] = lens_counts
            group['sigma_e'][:] = sigma_e
            # These are the same in metacal
            group['source_counts'][:] = N
            group['N_eff'][:] = N


    def read_config(self, args):
        """
        Extend the parent config reader to get z bin pairs

        Turns the list of redshift bin edges into a list
        of pairs.
        """
        config = super().read_config(args)
        zbin_edges = config['zbin_edges']
        zbins = list(zip(zbin_edges[:-1], zbin_edges[1:]))
        config['zbins'] = zbins
        return config

    

    def select_lens(self, phot_data):
        """Photometry cuts based on the BOSS Galaxy Target Selection:
        http://www.sdss3.org/dr9/algorithms/boss_galaxy_ts.php
        """
        mag_i = phot_data['mag_i']
        mag_r = phot_data['mag_r']
        mag_g = phot_data['mag_g']

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
        lens_gals = np.repeat(-1,n)

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
        lens_gals[lens_mask] = 0
        n_lens = lens_mask.sum()

        return lens_gals

    def select(self, data, bin_index):
        s2n_cut = self.config['s2n_cut']
        T_cut = self.config['T_cut']
        verbose = self.config['verbose']

        s2n = data['mcal_s2n']
        T = data['mcal_T']
        zbin = data['zbin']

        Tpsf = data['mcal_psf_T_mean']
        flag = data['mcal_flags']

        n0 = len(flag)
        sel  = flag==0
        f1 = sel.sum() / n0
        sel &= (T/Tpsf)>T_cut
        f2 = sel.sum() / n0
        sel &= s2n>s2n_cut
        f3 = sel.sum() / n0
        sel &= zbin==bin_index
        f4 = sel.sum() / n0
        
        variant = data.suffix
        if verbose:
            print(f"Bin {bin_index} ({variant}) {f1:.2%} flag, {f2:.2%} size, {f3:.2%} SNR, {f4:.2%} z")

        return sel


def flatten_list(lst):
    return [item for sublist in lst for item in sublist]


if __name__ == '__main__':
    PipelineStage.main()
