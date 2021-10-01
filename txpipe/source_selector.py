from .base_stage import PipelineStage
from .data_types import ShearCatalog, YamlFile, PhotozPDFFile, TomographyCatalog, HDFFile, TextFile
from .utils import SourceNumberDensityStats, rename_iterated
from .utils.calibration_tools import read_shear_catalog_type, apply_metacal_response
from .utils.calibration_tools import metacal_variants, metadetect_variants, band_variants, MetacalCalculator, LensfitCalculator, HSCCalculator, MetaDetectCalculator
from .utils.calibrators import MetaCalibrator, MetaDetectCalibrator, LensfitCalibrator, HSCCalibrator
import numpy as np
import warnings


class BinStats:
    def __init__(self, source_count, N_eff, mean_e, sigma_e, calibrator):
        super(BinStats, self).__init__()
        self.source_count = source_count
        self.N_eff = N_eff
        self.mean_e = mean_e
        self.sigma_e = sigma_e
        self.calibrator = calibrator

    def write_to(self, outfile, i):
        group = outfile['tomography']
        if i == '2d':
            group['source_counts_2d'][:] = self.source_count
            group['N_eff_2d'][:] = self.N_eff
            # This might get saved by the calibrator also
            # but in case not we do it here.
            group['mean_e1_2d'][:] = self.mean_e[0]
            group['mean_e2_2d'][:] = self.mean_e[1]
            group['sigma_e_2d'][:] = self.sigma_e
        else:
            group['source_counts'][i] = self.source_count
            group['N_eff'][i] = self.N_eff
            group['mean_e1'][i] = self.mean_e[0]
            group['mean_e2'][i] = self.mean_e[1]
            group['sigma_e'][i] = self.sigma_e

        self.calibrator.save(outfile, i)


class TXSourceSelectorBase(PipelineStage):
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
    name = "TXSourceSelector"


    inputs = [
        ('shear_catalog', ShearCatalog),
        ('calibration_table', TextFile),
    ]

    outputs = [
        ('shear_tomography_catalog', TomographyCatalog)
    ]

    config_options = {
        'input_pz': False,
        'true_z': False,
        'bands': 'riz', # bands from metacal to use
        'verbose': False,
        'T_cut':float,
        's2n_cut':float,
        'delta_gamma': float,
        'chunk_rows':10000,
        'source_zbin_edges':[float],
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

        if self.name == "TXSourceSelector":
            raise ValueError("Do not use the class TXSourceSelector any more. "
                             "Use one of the subclasses like TXSourceSelectorMetacal")

        # Suppress some warnings from numpy that are not relevant
        original_warning_settings = np.seterr(all='ignore')

        # Are we using a metacal or lensfit catalog?
        shear_catalog_type = read_shear_catalog_type(self)

        # The output file we will put the tomographic
        # information into
        output_file = self.setup_output()

        # The iterator that will loop through the data.
        # Set it up here so that we can find out if there are any
        # problems with it before we get run the slow classifier.
        it = self.data_iterator()

        # Build a classifier used to put objects into tomographic bins
        if not (self.config['input_pz'] or self.config['true_z']):
            classifier, features = self.build_tomographic_classifier()

        # We will collect the selection biases for each bin
        # as a matrix.  We will collect together the different
        # matrices for each chunk and do a weighted average at the end.
        nbin_source = len(self.config['source_zbin_edges'])-1

        number_density_stats = SourceNumberDensityStats(nbin_source, comm=self.comm,shear_type=self.config['shear_catalog_type'])

        calculators = self.setup_response_calculators(nbin_source)

        # Loop through the input data, processing it chunk by chunk
        for (start, end, shear_data) in it:
            print(f"Process {self.rank} running selection for rows {start:,}-{end:,}")

            if self.config['true_z'] or self.config['input_pz']:
                pz_data = self.apply_simple_redshift_cut(shear_data)

            else:
                # Select most likely tomographic source bin
                pz_data = self.apply_classifier(classifier, features, shear_data)

            # Combine this selection with size and snr cuts to produce a source selection
            # and calculate the shear bias it would generate
            tomo_bin, R, counts = self.calculate_tomography(pz_data, shear_data, calculators)

            # Save the tomography for this chunk
            self.write_tomography(output_file, start, end, tomo_bin, R)

            # Accumulate information on the number counts and the selection biases.
            # These will be brought together at the end.
            number_density_stats.add_data(shear_data, tomo_bin)  # check this

        # Do the selection bias averaging and output that too.
        self.write_global_values(output_file, calculators, number_density_stats)

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
        print("Using these bin edges:", self.config['source_zbin_edges'])
        for i, zmin in enumerate(self.config['source_zbin_edges'][:-1]):
            zmax = self.config['source_zbin_edges'][i+1]
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

        if self.config['shear_catalog_type'] == 'metacal':
            prefixes = ['', '', '', '' '', '']
            suffixes = ['', '_1p', '_2p', '_1m', '_2m']
        elif self.config['shear_catalog_type'] == 'metadetect':
            prefixes = ['00/', '1p/', '2p/', '1m/', '2m/']
            suffixes = ['', '', '', '' '', '']
        else:
            prefixes = ['', '', '', '' '', '']
            suffixes = ['', '', '', '' '', '']

        pz_data = {}

        for prefix, suffix in zip(prefixes, suffixes):
            # Pull out the columns that we have trained this bin selection
            # model on.
            data = []
            for f in features:
                # may be a single band
                if len(f) == 1:
                    col = shear_data[f'{prefix}mag_{f}{suffix}']
                # or a colour
                else:
                    b1,b2 = f.split('-')
                    col = shear_data[f'{prefix}mag_{b1}{suffix}'] - shear_data[f'{prefix}mag_{b2}{suffix}']
                if np.all(~np.isfinite(col)):
                    # entire column is NaN.  Hopefully this will get deselected elsewhere
                    col[:] = 30.0
                else:
                    ok = np.isfinite(col)
                    col[~ok] = col[ok].max()
                data.append(col)
            data = np.array(data).T

            # Run the random forest on this data chunk
            pz_data[f'{prefix}zbin{suffix}'] = classifier.predict(data)
        return pz_data

    def apply_simple_redshift_cut(self, shear_data):

        pz_data = {}
        if self.config['input_pz']:
            zz = shear_data['mean_z']
        else:
            zz = shear_data['redshift_true']

        pz_data_bin = np.zeros(len(zz), dtype=int) -1
        for zi in range(len(self.config['source_zbin_edges'])-1):
            mask_zbin = (zz>=self.config['source_zbin_edges'][zi]) & (zz<self.config['source_zbin_edges'][zi+1])
            pz_data_bin[mask_zbin] = zi

        return {'zbin': pz_data_bin}


    def calculate_tomography(self, pz_data, shear_data, calculators):
        """
        Select objects to go in each tomographic bin and their calibration.

        Parameters
        ----------

        pz_data: table or dict of arrays
            A chunk of input photo-z data containing mean values for each object
        shear_data: table or dict of arrays
            A chunk of input shear data with metacalibration variants.
        """
        nbin = len(self.config['source_zbin_edges'])-1
        n = len(list(shear_data.values())[0])

        # The main output data - the tomographic
        # bin index for each object, or -1 for no bin.
        tomo_bin = np.repeat(-1, n)

        # We also keep count of total count of objects in each bin
        counts = np.zeros(nbin + 1, dtype=int)

        data = {**pz_data, **shear_data}

        R = self.compute_per_object_response(data)

        for i in range(nbin):
            sel_00 = calculators[i].add_data(data, i)
            tomo_bin[sel_00] = i
            nsum = sel_00.sum()
            counts[i] = nsum
            # also count up the 2D sample
            counts[-1] += nsum

        # and calibrate the 2D sample.
        # This calibrator refers to self.select_2d
        calculators[-1].add_data(data)

        return tomo_bin, R, counts

    def compute_per_object_response(self, data):
        # The default implementation has no per-object weight
        return None

    def setup_output(self):
        """
        Set up the output data file.

        Creates the data sets and groups to put module output
        in the shear_tomography_catalog output file.
        """
        cat_type = read_shear_catalog_type(self)
        with self.open_input('shear_catalog', wrapper=True) as f:
            n = f.get_size()

        zbins = self.config['source_zbin_edges']
        nbin_source = len(zbins) - 1

        outfile = self.open_output('shear_tomography_catalog', parallel=True)
        group = outfile.create_group('tomography')
        group.create_dataset('source_bin', (n,), dtype='i')
        group.create_dataset('source_counts', (nbin_source,), dtype='i')
        group.create_dataset('source_counts_2d', (1,), dtype='i')
        group.create_dataset('sigma_e', (nbin_source,), dtype='f')
        group.create_dataset('sigma_e_2d', (1,), dtype='f')
        group.create_dataset('mean_e1', (nbin_source,), dtype='f')
        group.create_dataset('mean_e2', (nbin_source,), dtype='f')
        group.create_dataset('mean_e1_2d', (1,), dtype='f')
        group.create_dataset('mean_e2_2d', (1,), dtype='f')
        group.create_dataset('N_eff', (nbin_source,), dtype='f')
        group.create_dataset('N_eff_2d', (1,), dtype='f')

        group.attrs['nbin_source'] = nbin_source
        group.attrs['catalog_type'] = self.config["shear_catalog_type"]
        for i in range(nbin_source):
            group.attrs[f'source_zmin_{i}'] = zbins[i]
            group.attrs[f'source_zmax_{i}'] = zbins[i+1]

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

    def write_global_values(self, outfile, calculators, number_density_stats):
        """
        Write out overall selection biases

        Parameters
        ----------

        outfile: h5py.File

        S: array of shape (nbin,2,2)
            Selection bias matrices
        """
        nbin_source = len(calculators) - 1
        means, variances = number_density_stats.collect()

        # Loop through the tomographic calculators.
        for i in range(nbin_source + 1):
            stats = self.compute_output_stats(calculators[i], means[i], variances[i])
            if self.rank == 0:
                stats.write_to(outfile, i if i < nbin_source else '2d')



    def select(self, data, bin_index):
        zbin = data['zbin']
        verbose = self.config['verbose']

        sel = self.select_2d(data, calling_from_select=True)
        sel &= zbin==bin_index
        f4 = sel.sum() / sel.size

        if verbose:
            print(f"{f4:.2%} z for bin {bin_index}")
            print("total tomo", sel.sum())

        return sel

    def select_2d(self, data, calling_from_select=False):
        # Select any objects that pass general WL cuts
        # The calling_from_select option just specifies whether we
        # are calling this function from within the select
        # method above, because the useful printed verbose
        # output is different in each case
        s2n_cut = self.config['s2n_cut']
        T_cut = self.config['T_cut']
        verbose = self.config['verbose']
        variant = data.suffix

        shear_prefix = self.config['shear_prefix']
        s2n = data[f'{shear_prefix}s2n']
        T = data[f'{shear_prefix}T']

        Tpsf = data[f'{shear_prefix}psf_T_mean']
        flag = data[f'{shear_prefix}flags']

        n0 = len(flag)
        sel  = flag==0
        f1 = sel.sum() / n0
        sel &= (T/Tpsf)>T_cut
        f2 = sel.sum() / n0
        sel &= s2n>s2n_cut
        f3 = sel.sum() / n0
        sel &= data['zbin'] >= 0
        f4 = sel.sum() / n0

        # Print out a message.  If we are selecting a 2D sample
        # this is the complete message.  Otherwise if we are about
        # to also apply a redshift bin cut about then the message will continue
        # as above
        if verbose and calling_from_select:
            print(f"Tomo selection ({variant}) {f1:.2%} flag, {f2:.2%} size, "
                  f"{f3:.2%} SNR, ", end="")
        elif verbose:
            print(f"2D selection ({variant}) {f1:.2%} flag, {f2:.2%} size, "
                  f"{f3:.2%} SNR, {f4:.2%} any z bin")
            print("total 2D", sel.sum())
        return sel

class TXSourceSelectorMetacal(TXSourceSelectorBase):
    name='TXSourceSelectorMetacal'
    def data_iterator(self):
        bands = self.config['bands']
        shear_cols = metacal_variants('mcal_T', 'mcal_s2n', 'mcal_g1', 'mcal_g2', 'mcal_flags')
        shear_cols += ['ra', 'dec', 'mcal_psf_T_mean', 'weight']
        shear_cols += band_variants(bands, 'mcal_mag', 'mcal_mag_err', shear_catalog_type='metacal')

        if self.config['input_pz']:
            shear_cols += metacal_variants('mean_z')
        elif self.config['true_z']:
            shear_cols += ['redshift_true']

        chunk_rows = self.config['chunk_rows']
        return self.iterate_hdf('shear_catalog', 'shear', shear_cols, chunk_rows)

    def setup_output(self):
        outfile = super().setup_output()
        n = outfile['tomography/source_bin'].size
        nbin_source = outfile['tomography/source_counts'].size
        group = outfile.create_group('response')
        group.create_dataset('R_gamma', (n,2,2), dtype='f')
        group.create_dataset('R_S', (nbin_source,2,2), dtype='f')
        group.create_dataset('R_gamma_mean', (nbin_source,2,2), dtype='f')
        group.create_dataset('R_total', (nbin_source,2,2), dtype='f')
        group.create_dataset('R_S_2d', (2,2), dtype='f')
        group.create_dataset('R_gamma_mean_2d', (2,2), dtype='f')
        group.create_dataset('R_total_2d', (2,2), dtype='f')
        return outfile

    def setup_response_calculators(self, nbin_source):
        delta_gamma = self.config['delta_gamma']
        calculators = [MetacalCalculator(self.select, delta_gamma) for i in range(nbin_source)]
        calculators.append(MetacalCalculator(self.select_2d, delta_gamma))
        return calculators

    def write_tomography(self, outfile, start, end, source_bin, R):
        super().write_tomography(outfile, start, end, source_bin, R)
        group = outfile['response']
        group['R_gamma'][start:end,:,:] = R

    def compute_per_object_response(self, data):
        delta_gamma = self.config['delta_gamma']
        n = data['mcal_g1_1p'].size
        R = np.zeros((n, 2, 2))
        R[:,0,0] = (data['mcal_g1_1p'] - data['mcal_g1_1m']) / delta_gamma
        R[:,0,1] = (data['mcal_g1_2p'] - data['mcal_g1_2m']) / delta_gamma
        R[:,1,0] = (data['mcal_g2_1p'] - data['mcal_g2_1m']) / delta_gamma
        R[:,1,1] = (data['mcal_g2_2p'] - data['mcal_g2_2m']) / delta_gamma
        return R

    def apply_simple_redshift_cut(self, data):
        # If we have the truth pz then we just need to do the binning once,
        # as in the parent class
        if not self.config['input_pz']:
            return super().apply_simple_redshift_cut(data)

        # Otherwise we have to do it once for each variant
        pz_data = {}
        variants = ['', '_1p', '_2p', '_1m', '_2m']
        for v in variants:
            zz = shear_data[f'mean_z{v}']

            pz_data_v = np.zeros(len(zz), dtype=int) -1
            for zi in range(len(self.config['source_zbin_edges'])-1):
                mask_zbin = (zz>=self.config['source_zbin_edges'][zi]) & (zz<self.config['source_zbin_edges'][zi+1])
                pz_data_v[mask_zbin] = zi

            pz_data[f'zbin{v}'] = pz_data_v

        return pz_data

    def compute_output_stats(self, calculator, mean, variance):
        R, S, N = calculator.collect(self.comm)
        calibrator = MetaCalibrator(R, S, mean, mu_is_calibrated=False)
        mean_e = calibrator.mu

        Rtot = R + S
        P = np.diag(np.linalg.inv(Rtot @ Rtot))
        # Apply to the variances to get sigma_e
        sigma_e = np.sqrt(0.5 * P @ variance)

        return BinStats(N, N, mean_e, sigma_e, calibrator)




class TXSourceSelectorMetadetect(TXSourceSelectorBase):
    name = "TXSourceSelectorMetadetect"
    def data_iterator(self):
        chunk_rows = self.config['chunk_rows']
        bands = self.config['bands']
        shear_cols = metadetect_variants(
                'T', 's2n', 'g1', 'g2',
                'ra', 'dec', 'mcal_psf_T_mean', 'weight', 'flags'
                )
        shear_cols += band_variants(bands, 'mag', 'mag_err', shear_catalog_type='metadetect')

        if self.config['input_pz']:
            shear_cols += metadetect_variants('mean_z')
        elif self.config['true_z']:
            shear_cols += metadetect_variants('redshift_true')

        renames = {}
        for prefix in ['00', '1p', '1m', '2p', '2m']:
            renames[f'{prefix}/mcal_psf_T_mean'] = f'{prefix}/psf_T_mean'

        it = self.iterate_hdf('shear_catalog', 'shear', shear_cols, chunk_rows, longest=True)
        return rename_iterated(it, renames)

    def setup_response_calculators(self, nbin_source):
        delta_gamma = self.config['delta_gamma']
        calculators = [MetaDetectCalculator(self.select, delta_gamma) for i in range(nbin_source)]
        calculators.append(MetaDetectCalculator(self.select_2d, delta_gamma))
        return calculators

    def apply_simple_redshift_cut(self, data):
        # If we have the truth pz then we just need to do the binning once,
        # as in the parent class
        if not self.config['input_pz']:
            return super().apply_simple_redshift_cut(data)

        # Otherwise we have to do it once for each variant
        pz_data = {}
        variants = ['', '_1p', '_2p', '_1m', '_2m']
        for v in variants:
            zz = shear_data[f'mean_z{v}']

            pz_data_v = np.zeros(len(zz), dtype=int) -1
            for zi in range(len(self.config['source_zbin_edges'])-1):
                mask_zbin = (zz>=self.config['source_zbin_edges'][zi]) & (zz<self.config['source_zbin_edges'][zi+1])
                pz_data_v[mask_zbin] = zi

            pz_data[f'zbin{v}'] = pz_data_v

        return pz_data


    def setup_output(self):
        outfile = super().setup_output()
        n = outfile['tomography/source_bin'].size
        nbin_source = outfile['tomography/source_counts'].size
        group = outfile.create_group('response')
        group.create_dataset('R', (nbin_source,2,2), dtype='f')
        group.create_dataset('R_2d', (2,2), dtype='f')
        return outfile

    def compute_output_stats(self, calculator, mean, variance):
        R, N = calculator.collect(self.comm, allgather=True)
        calibrator = MetaDetectCalibrator(R, mean, mu_is_calibrated=False)
        mean_e = calibrator.mu

        P = np.diag(np.linalg.inv(R @ R))
        # Apply to the variances to get sigma_e
        sigma_e = np.sqrt(0.5 * P @ variance)

        return BinStats(N, N, mean_e, sigma_e, calibrator)


class TXSourceSelectorLensfit(TXSourceSelectorBase):
    name = "TXSourceSelectorLensfit"
    def data_iterator(self):
        chunk_rows = self.config['chunk_rows']
        bands = self.config['bands']
        shear_cols = ['psf_T_mean', 'weight', 'flags', 'T', 's2n', 'g1', 'g2','weight','m']
        shear_cols += band_variants(bands, 'mag', 'mag_err', shear_catalog_type='lensfit')
        if self.config['input_pz']:
            shear_cols += ['mean_z']
        elif self.config['true_z']:
            shear_cols += ['redshift_true']
        return self.iterate_hdf('shear_catalog', 'shear', shear_cols, chunk_rows)

    def setup_response_calculators(self, nbin_source):
        calculators = [LensfitCalculator(self.select,self.config['input_m_is_weighted']) for i in range(nbin_source)]
        calculators.append(LensfitCalculator(self.select_2d,self.config['input_m_is_weighted']))
        return calculators

    def setup_output(self):
        outfile = super().setup_output()
        n = outfile['tomography/source_bin'].size
        nbin_source = outfile['tomography/source_counts'].size
        group = outfile.create_group('response')
        group.create_dataset('K', (nbin_source,), dtype='f')
        group.create_dataset('C', (nbin_source,2), dtype='f')
        group.create_dataset('K_2d', (1,), dtype='f')
        group.create_dataset('C_2d', (2), dtype='f')
        return outfile


    def compute_output_stats(self, calculator, mean, variance):
        K, C, N = calculator.collect(self.comm, allgather=True)
        calibrator = LensfitCalibrator(K, C)
        mean_e = C
        sigma_e = np.sqrt((0.5 * (variance[0] + variance[1]))) / (1 + K)

        return BinStats(N, N, mean_e, sigma_e, calibrator)


class TXSourceSelectorHSC(TXSourceSelectorBase):
    name = "TXSourceSelectorHSC"
    def data_iterator(self):
        chunk_rows = self.config['chunk_rows']
        bands = self.config['bands']
        shear_cols = ['psf_T_mean', 'weight', 'flags', 'T', 's2n', 'g1', 'g2','weight','m','c1','c2','sigma_e']
        shear_cols += band_variants(bands, 'mag', 'mag_err', shear_catalog_type='hsc')
        if self.config['input_pz']:
            shear_cols += ['mean_z']
        elif self.config['true_z']:
            shear_cols += ['redshift_true']
        return self.iterate_hdf('shear_catalog', 'shear', shear_cols, chunk_rows)

    def setup_output(self):
        outfile = super().setup_output()
        n = outfile['tomography/source_bin'].size
        nbin_source = outfile['tomography/source_counts'].size
        group = outfile.create_group('response')
        group.create_dataset('R', (n,), dtype='f')
        group.create_dataset('K', (nbin_source,), dtype='f')
        group.create_dataset('C', (nbin_source,2), dtype='f')
        group.create_dataset('R_mean', (nbin_source,), dtype='f')
        group.create_dataset('K_2d', (1,), dtype='f')
        group.create_dataset('C_2d', (2), dtype='f')
        group.create_dataset('R_mean_2d', (1,), dtype='f')
        return outfile

    def write_tomography(self, outfile, start, end, source_bin, R):
        super().write_tomography(outfile, start, end, source_bin, R)
        group = outfile['response']
        group['R'][start:end] = R

    def compute_per_object_response(self, data):
        w_tot = np.sum(data['weight'])
        R =  np.array([1. - np.sum(data['weight']*data['sigma_e'])/w_tot]*len(data['weight']))
        return R

    def compute_output_stats(self, calculator, mean, variance):
        raise NotImplementedError("HSC calib is broken")
        R, C, N = calculator.collect(self.comm, allgather=True)
        calibrator = HSCCalibrator(R, mean)
        mean_e = C
        sigma_e = np.sqrt((0.5 * (variance[0] + variance[1]))) / (1 + K[i])

        return BinStats(N, N, mean_e, sigma_e, calibrator)





if __name__ == '__main__':
    PipelineStage.main()
