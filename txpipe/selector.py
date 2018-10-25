from ceci import PipelineStage
from .data_types import MetacalCatalog, YamlFile, PhotozPDFFile, TomographyCatalog, HDFFile
from .utils import NumberDensityStats

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
        ('photoz_pdfs', PhotozPDFFile),
        ('photometry_catalog', HDFFile),
    ]

    outputs = [
        ('tomography_catalog', TomographyCatalog)
    ]

    config_options = {
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
        'r_i_cut':2.0
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
        import numpy as np

        output_file = self.setup_output()

        # Columns we need from the redshift data
        pz_cols = ['mu', 'mu_1p', 'mu_1m', 'mu_2p', 'mu_2m']

        # Colums we need from the photometry data
        phot_cols = ['mag_true_u_lsst', 'mag_true_g_lsst', 
                'mag_true_r_lsst', 'mag_true_i_lsst', 
                'mag_true_z_lsst']

        # Columns we need from the shear catalog
        cat_cols = ['mcal_flags', 'mcal_Tpsf']

        # Including all the metacalibration variants of these columns
        for c in ['mcal_T', 'mcal_s2n_r', 'mcal_g']:
            cat_cols += [c, c+"_1p", c+"_1m", c+"_2p", c+"_2m", ]


        # Input data.  These are iterators - they lazily load chunks
        # of the data one by one later when we do the for loop.
        # This code can be run in parallel, and different processes will
        # each get different chunks of the data 
        chunk_rows = self.config['chunk_rows']
        iter_pz    = self.iterate_hdf('photoz_pdfs', 'pdf', pz_cols, chunk_rows)
        iter_shear = self.iterate_fits('shear_catalog', 1, cat_cols, chunk_rows)
        iter_phot = self.iterate_hdf('photometry_catalog', 'photometry', phot_cols, chunk_rows)

        # We will collect the selection biases for each bin
        # as a matrix.  We will collect together the different
        # matrices for each chunk and do a weighted average at the end.
        nbin_source = len(self.config['zbins'])
        nbin_lens = 1

        selection_biases = []
        number_density_stats = NumberDensityStats(nbin_source, nbin_lens, self.comm)

        # Loop through the input data, processing it chunk by chunk
        for (start, end, pz_data), (_, _, shear_data), (_, _, phot_data) in zip(iter_pz, iter_shear, iter_phot):

            print(f"Process {self.rank} running selection for rows {start}-{end}")
            tomo_bin, R, S, counts = self.calculate_tomography(pz_data, shear_data)


            lens_gals = self.select_lens(phot_data)

            # Save the tomography for this chunk
            self.write_tomography(output_file, start, end, tomo_bin, lens_gals, R, lens_gals)

            number_density_stats.add_data(shear_data, tomo_bin, R, lens_gals)

            # The selection biases are the mean over all the data, so we
            # build them up as we go along and average them at the end.
            selection_biases.append((S,counts))

        # Do the selection bias averaging and output that too.
        S, source_counts = self.average_selection_bias(selection_biases)
        self.write_global_values(output_file, S, source_counts, number_density_stats)

        # Save and complete
        output_file.close()



    def calculate_tomography(self, pz_data, shear_data):
        """
        Select objects to go in each tomographic bin and their calibration.

        Parameters
        ----------

        pz_data: table or dict of arrays
            A chunk of input photo-z data containing mean values for each object
        shear_data: table or dict of arrays
            A chunk of input shear data with metacalibration variants.
        """
        import numpy as np

        g = shear_data['mcal_g']
        delta_gamma = self.config['delta_gamma']
        nbin = len(self.config['zbins'])

        n = len(shear_data)

        # The main output data - the tomographic
        # bin index for each object, or -1 for no bin.
        tomo_bin = np.repeat(-1, n)

        # The two biases - the response to shear R and the
        # selection bias S.
        R = np.zeros((n,2,2))
        S = np.zeros((nbin,2,2))

        # We also keep count of total count of objects in each bin
        counts = np.zeros(nbin, dtype=int)


        for i,(zmin, zmax) in enumerate(self.config['zbins']):

            # The main selection.  The select function below returns a
            # boolean array where True means "selected and in this bin"
            # and False means "cut or not in this bin".
            sel_00 = self.select(shear_data, pz_data, '', zmin, zmax)

            # The metacalibration selections, used to work out selection
            # biases
            sel_1p = self.select(shear_data, pz_data, '_1p', zmin, zmax)
            sel_2p = self.select(shear_data, pz_data, '_2p', zmin, zmax)
            sel_1m = self.select(shear_data, pz_data, '_1m', zmin, zmax)
            sel_2m = self.select(shear_data, pz_data, '_2m', zmin, zmax)

            # Assign these objects to this bin
            tomo_bin[sel_00] = i

            # Multiplicative estimator bias in this bin
            # One value per object
            R_1 = (shear_data['mcal_g_1p'][sel_00] - shear_data['mcal_g_1m'][sel_00]) / delta_gamma
            R_2 = (shear_data['mcal_g_2p'][sel_00] - shear_data['mcal_g_2m'][sel_00]) / delta_gamma

            # Selection bias for this chunk - we must average over these later.
            # For now there is one value per bin.
            S_1 = (g[sel_1p].mean(0) - g[sel_1m].mean(0)) / delta_gamma
            S_2 = (g[sel_2p].mean(0) - g[sel_2m].mean(0)) / delta_gamma

            # Save in a more useful ordering for the output - 
            # as a matrix
            R[sel_00,0,0] = R_1[:,0]
            R[sel_00,0,1] = R_1[:,1]
            R[sel_00,1,0] = R_2[:,0]
            R[sel_00,1,1] = R_2[:,1]


            # Also save the selection biases as a matrix.
            S[i,0,0] = S_1[0]
            S[i,0,1] = S_1[1]
            S[i,1,0] = S_2[0]
            S[i,1,1] = S_2[1]
            counts[i] = sel_00.sum()

        return tomo_bin, R, S, counts

        
    def average_selection_bias(self, selection_biases):
        """
        Compute the average selection bias.

        Average the selection biases, which are matrices
        of shape [nbin, 2, 2].  Each matrix comes from 
        a different chunk of data and we do a weighted
        average over them all.

        Parameters
        ----------

        selection_biases: list of pairs (bias, count)
            The selection biases

        Returns
        -------
        S: array (nbin,2,2)
            Average selection bias
        """
        import numpy as np

        if self.is_mpi():
            s = self.comm.gather(selection_biases)
            if self.rank!=0:
                return None,None
            selection_biases = flatten_list(s)


        S = 0.0
        N = 0

        # Accumulate the total and the counts
        # for each bin
        for S_i, count in selection_biases:
            S += count[:,np.newaxis,np.newaxis]*S_i
            N += count

        # And finally divide by the total count
        # to get the average
        for i,n_i in enumerate(N):
            S[i]/=n_i

        return S, N




    def setup_output(self):
        """
        Set up the output data file.

        Creates the data sets and groups to put module output
        in the tomography_catalog output file.
        """
        n = self.open_input('shear_catalog')[1].get_nrows()
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

    def write_global_values(self, outfile, S, source_counts, number_density_stats):
        """
        Write out overall selection biases

        Parameters
        ----------

        outfile: h5py.File

        S: array of shape (nbin,2,2)
            Selection bias matrices
        """
        sigma_e, N_eff, lens_counts = number_density_stats.collect()

        if self.rank==0:
            group = outfile['multiplicative_bias']
            group['R_S'][:,:,:] = S
            group = outfile['tomography']
            group['source_counts'][:] = source_counts
            group['lens_counts'][:] = lens_counts
            group['sigma_e'][:] = sigma_e
            group['N_eff'][:] = N_eff


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
        import numpy as np

        mag_i = phot_data['mag_true_i_lsst']
        mag_r = phot_data['mag_true_r_lsst']
        mag_g = phot_data['mag_true_g_lsst']

        # Mag cuts 
        cperp_cut_val = self.config['cperp_cut']
        r_cpar_cut_val = self.config['r_cpar_cut']
        r_lo_cut_val = self.config['r_lo_cut']
        r_hi_cut_val = self.config['r_hi_cut']
        i_lo_cut_val = self.config['i_lo_cut']
        i_hi_cut_val = self.config['i_hi_cut']
        r_i_cut_val = self.config['r_i_cut']
        print(r_lo_cut_val, r_hi_cut_val)

        n = len(mag_i)
        # HDF does not support bools, so we will prepare a binary array
        # where 0 is a lens and 1 is not
        lens_gals = np.repeat(1,n)

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

    def select(self, shear_data, pz_data, variant, zmin, zmax):
        n = len(shear_data)

        s2n_cut = self.config['s2n_cut']
        T_cut = self.config['T_cut']

        T_col = 'mcal_T' + variant
        s2n_col = 'mcal_s2n_r' + variant

        z_col = 'mu' + variant

        s2n = shear_data[s2n_col]
        T = shear_data[T_col]
        z = pz_data[z_col]

        Tpsf = shear_data['mcal_Tpsf']
        flag = shear_data['mcal_flags']

        sel  = flag==0
        sel &= (T/Tpsf)>T_cut
        sel &= s2n>s2n_cut
        sel &= z>=zmin
        sel &= z<zmax

        return sel


def flatten_list(lst):
    return [item for sublist in lst for item in sublist]


if __name__ == '__main__':
    PipelineStage.main()
