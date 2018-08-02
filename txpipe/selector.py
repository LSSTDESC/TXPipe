from ceci import PipelineStage
from descformats.tx import MetacalCatalog, YamlFile, PhotozPDFFile, TomographyCatalog, HDFFile



class TXSelector(PipelineStage):
    """
    Selects and constructs metacal calibrations for tomographic bins of objects

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
    config_options = {'T_cut':float, 's2n_cut':float, 'delta_gamma': float, 'max_rows':0,
                      'chunk_rows':10000, 'zbin_edges':[float]}

    def run(self):
        import numpy as np

        info = self.config
        output_file = self.setup_output(info)

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

        # Columns we need from the photometry catalog
        bands = self.config['bands']
        cols = [f'mag_{band}_lsst{suffix}' for band in bands]


        # Input data.  These are iterators - they lazily load chunks
        # of the data one by one later when we do the for loop
        chunk_rows = info['chunk_rows']
        iter_pz = self.iterate_hdf('photoz_pdfs', 'pdf', pz_cols, chunk_rows)
        iter_shear = self.iterate_fits('shear_catalog', 1, cat_cols, chunk_rows)
        iter_phot = self.iterate_hdf('photometry_catalog', 'photometry', cols, chunk_rows)

        selection_biases = []

        # Loop through the input data, processing it chunk by chunk
        for (start, end, pz_data), (_, _, shear_data), (_, _, phot_data) in zip(iter_pz, iter_shear, iter_phot):

            print(f"Process {self.rank} running selection for rows {start}-{end}")
            tomo_bin, R, S, counts = self.calculate_tomography(pz_data, shear_data, info)

            lens_gals = select_lens(phot_data)
            # Save the tomography for this chunk
            self.write_tomography(output_file, start, end, tomo_bin, R, lens_gals)

            # The selection biases are the mean over all the data, so we
            # build them up as we go along and average them at the end.
            selection_biases.append((S,counts))

        # Do the selection bias averaging and output that too.
        S = self.average_selection_bias(selection_biases)
        self.write_selection_bias(output_file, S)



        output_file.close()

        

    def calculate_tomography(self, pz_data, shear_data, info):
        # for each tomographic bin, select objects in that bin
        # under each metacal choice
        import numpy as np

        g = shear_data['mcal_g']
        delta_gamma = info['delta_gamma']
        nbin = len(info['zbins'])

        n = len(shear_data)
        tomo_bin = np.repeat(-1, n)
        R = np.zeros((n,2,2))
        S = np.zeros((nbin,2,2))
        counts = np.zeros(nbin, dtype=int)


        for i,(zmin, zmax) in enumerate(info['zbins']):
            info['zmin'] = zmin
            info['zmax'] = zmax

            # The main selection.
            sel_00 = select(shear_data, pz_data, info, '')
            # The metacalibration selections, used to work out selection
            # biases
            sel_1p = select(shear_data, pz_data, info, '_1p')
            sel_2p = select(shear_data, pz_data, info, '_2p')
            sel_1m = select(shear_data, pz_data, info, '_1m')
            sel_2m = select(shear_data, pz_data, info, '_2m')

            # Assign these objects to this bin
            tomo_bin[sel_00] = i

            # Multiplicative estimator bias in this bin
            # One value per object
            R_1 = (shear_data['mcal_g_1p'][sel_00] - shear_data['mcal_g_1m'][sel_00]) / delta_gamma
            R_2 = (shear_data['mcal_g_2p'][sel_00] - shear_data['mcal_g_2m'][sel_00]) / delta_gamma

            # Selection bias for this chunk - we must average over these later.
            # For now there is one value per bin
            S_1 = (g[sel_1p].mean(0) - g[sel_1m].mean(0)) / delta_gamma
            S_2 = (g[sel_2p].mean(0) - g[sel_2m].mean(0)) / delta_gamma

            # Save in a more useful ordering for the output
            R[sel_00,0,0] = R_1[:,0]
            R[sel_00,0,1] = R_1[:,1]
            R[sel_00,1,0] = R_2[:,0]
            R[sel_00,1,1] = R_2[:,1]

            S[i,0,0] = S_1[0]
            S[i,0,1] = S_1[1]
            S[i,1,0] = S_2[0]
            S[i,1,1] = S_2[1]
            counts[i] = sel_00.sum()
        return tomo_bin, R, S, counts

    def select_lens(self, phot_data):
        pass
        return
        
    def average_selection_bias(self, selection_biases):
        import numpy as np

        if self.is_mpi():
            s = self.comm.gather(selection_biases)
            if self.rank!=0:
                return
            selection_biases = flatten_list(s)


        S = 0.0
        N = 0

        for S_i, count in selection_biases:
            S += count[:,np.newaxis,np.newaxis]*S_i
            N += count
            # count (4) S_i (4,2,2)
        # For each bin
        for i,n_i in enumerate(N):
            S[i]/=n_i

        return S




    def setup_output(self, info):
        n = self.open_input('shear_catalog')[1].get_nrows()
        zbins = info['zbins']
        nbin = len(zbins)
        outfile = self.open_output('tomography_catalog', parallel=True)
        group = outfile.create_group('tomography')
        group.create_dataset('bin', (n,), dtype='i')
        group.create_dataset('lens', (n,), dtype='i')
        group.attrs['nbin'] = nbin
        for i in range(nbin):
            group.attrs[f'zmin_{i}'] = zbins[i][0]
            group.attrs[f'zmax_{i}'] = zbins[i][1]
        group = outfile.create_group('multiplicative_bias')
        group.create_dataset('R_gamma', (n,2,2), dtype='i')
        group.create_dataset('R_S', (nbin,2,2), dtype='i')
        return outfile

    def write_tomography(self, outfile, start, end, tomo_bin, R, lens_gals):
        group = outfile['tomography']
        group['bin'][start:end] = tomo_bin
        group['lens'][start:end] = lens_gals
        group = outfile['multiplicative_bias']
        group['R_gamma'][start:end,:,:] = R

    def write_selection_bias(self, outfile, S):
        if self.rank==0:
            group = outfile['multiplicative_bias']
            group['R_S'][:,:,:] = S


    def read_config(self, args):
        config = super().read_config(args)
        zbin_edges = config['zbin_edges']
        zbins = list(zip(zbin_edges[:-1], zbin_edges[1:]))
        config['zbins'] = zbins
        return config


def select_lens(phot_data):

    """Photometry cuts based on the BOSS Galaxy Target Selection:
    http://www.sdss3.org/dr9/algorithms/boss_galaxy_ts.php
    """

    mag_i = phot_data['mag_i_lsst'].value
    mag_r = phot_data['mag_r_lsst'].value
    mag_g = phot_data['mag_g_lsst'].value

    cpar = 0.7 * (mag_g - mag_r) + 1.2 * ((mag_r - mag_i) - 0.18)
    cperp = (mag_r - mag_i) - ((mag_g - mag_r) / 4.0) - 0.18
    dperp = (mag_r - mag_i) - ((mag_g - mag_r) / 8.0)

    # LOWZ
    cperp_cut = np.abs(cperp) < 0.2
    r_cpar_cut = mag_r < 13.5 + cpar / 0.3
    r_lo_cut = mag_r > 16.0
    r_hi_cut = mag_r < 19.6

    lowz_cut = (cperp_cut) & (r_cpar_cut) & (r_lower_cut) & (r_upper_cut)

    # CMASS
    i_lo_cut = mag_i > 17.5
    i_hi_cut = mag_i < 19.9
    r_i_cut = (mag_r - mag_i) < 2.0
    #dperp_cut = dperp > 0.55 # this cut did not return any sources...

    cmass_cut = (i_lower_cut) & (i_upper_cut) & (r_i_cut)

    return lowz_cut | cmass_cut


def select(shear_data, pz_data, phot_data, cuts, variant):
    n = len(shear_data)

    s2n_cut = cuts['s2n_cut']
    T_cut = cuts['T_cut']

    s2n_col = 'mcal_T' + variant
    T_col = 'mcal_s2n_r' + variant
    z_col = 'mu' + variant

    s2n = shear_data[s2n_col]
    T = shear_data[T_col]
    z = pz_data[z_col]

    Tpsf = shear_data['mcal_Tpsf']
    flag = shear_data['mcal_flags']

    zmin = cuts['zmin']
    zmax = cuts['zmax']

    
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
