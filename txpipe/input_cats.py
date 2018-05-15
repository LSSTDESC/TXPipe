from ceci import PipelineStage
from descformats.tx import MetacalCatalog, HDFFile

# could also just load /global/projecta/projectdirs/lsst/groups/CS/descqa/catalog/ANL_AlphaQ_v3.0.hdf5


class TXDProtoDC2Mock(PipelineStage):
    """


    """
    name='TXDProtoDC2Mock'
    inputs = [
    ]
    outputs = [
        ('metacal_catalog', MetacalCatalog),
        ('photometry_catalog', HDFFile),
    ]
    config_options = {'cat_name':'protoDC2_test', 'visits_per_band':165}

    def data_iterator(self, gc):

        cols = ['mag_true_u_lsst', 'mag_true_g_lsst', 
                'mag_true_r_lsst', 'mag_true_i_lsst', 
                'mag_true_z_lsst',
                'ellipticity_1_true', 'ellipticity_2_true',
                'shear_1', 'shear_2',
                'size_true',
                'galaxy_id',
                ]

        it = gc.get_quantities(cols, return_iterator=True)
        for data in it:
            yield data

    def get_catalog_size(self, gc):
        f = h5py.File(gc.get_catalog_info()['filename'])
        n = ['galaxyProperties/ra'].size
        f.close()
        return n

    def run(self):
        import GCRCatalogs
        cat_name = self.config['cat_name']
        gc = GCRCatalogs.load_catalog(cat_name)
        N = self.get_size(gc)

        metacal_file = self.open_output('metacal_catalog', parallel=False)
        photo_file = self.open_output('photometry_catalog', parallel=False)


        start = 0
        for data in self.data_iterator(gc):
            end = start + len(data.values()[0])

            mock_photometry = self.make_mock_photometry(data)
            mock_metacal = self.make_mock_metacal(data, mock_photometry)

            self.write_photometry(photo_file, mock_photometry, start, end)
            self.write_metacal(metacal_file, mock_metacal, start, end)



    def make_mock_photometry(self, data):
        bands = ('u','g', 'r', 'i', 'z')
        n_visit = self.config['visits_per_band']
        photo = make_mock_photometry(n_visit, bands, data)
        return photo

    def make_mock_metacal(self, data, photo):
        """
        Generate a mock metacal table with noise added
        """
        # These are the numbers from figure F1 of the DES Y1 shear catalog paper
        # (this version is not yet public but is awaiting a second referee response)
        import numpy as np
        import scipy.interpolate

        # strategy here.
        # we have a S/N per band for each object.
        # get the total S/N in r,i,z since most shapes are measured there.
        # if it's > 5 then make a fake R_mean value based on the signal to noise and size
        # Then generate R_mean from R with some spread
        # 
        # Just use the true shear * R_mean for the estimated shear
        # (the noise on R will do the job of noise on shear)
        # Use R11 = R22 and R12 = R21 = 0

        # wasteful - we are making this every chunk of data
        spline_snr = np.log10([0.01,  5.7,   7.4,   9.7,  12.6,  16.5,  21.5,  28. ,  36.5,  47.5,
                    61.9,  80.7, 105.2, 137.1, 178.7, 232.9, 303.6, 395.7, 515.7,
                    672.1, 875.9])
        spline_R = array([0.001,  0.07, 0.15, 0.25, 0.35, 0.43, 0.5 , 0.54, 0.56, 0.58, 0.59, 0.59,
                    0.6 , 0.6 , 0.59, 0.57, 0.55, 0.52, 0.5 , 0.48, 0.46])

        spline = scipy.interpolate.interp1d(spline_snr, spline_R, kind='cubic')

        # Now we need the SNR of the object.




def make_mock_photometry(n_visit, bands, data):
    """
    Generate a mock photometric table with noise added

    This is mostly from LSE‐40 by 
    Zeljko Ivezic, Lynne Jones, and Robert Lupton
    retrieved here:
    http://faculty.washington.edu/ivezic/Teaching/Astr511/LSST_SNRdoc.pdf
    """

    import numpy as np


    # Sky background, seeing, and system throughput, 
    # all from table 2 of Ivezic, Jones, & Lupton
    B_b = np.array([85.07, 467.9, 1085.2, 1800.3, 2775.7])
    fwhm = np.array([0.77, 0.73, 0.70, 0.67, 0.65])
    T_b = np.array([0.0379, 0.1493, 0.1386, 0.1198, 0.0838])


    # effective pixels size for a Gaussian PSF, from equation
    # 27 of Ivezic, Jones, & Lupton
    pixel = 0.2 # arcsec
    N_eff = 2.436 * (fwhm/pixel)**2


    # other numbers from Ivezic, Jones, & Lupton
    sigma_inst2 = 10.0**2  #instrumental noise in photons per pixel, just below eq 42
    gain = 1  # ADU units per photon, also just below eq 42
    D = 8.4 # primary mirror diameter in meters, from LSST key numbers page
    time = 30. # seconds per exposure, from LSST key numbers page
    sigma_b2 = 0.0 # error on background, just above eq 42

    # combination of these  used below, from various equations
    factor = 5455./gain * (D/6.5)**2 * (time/30)

    for band, b_b, t_b, n_eff in zip(bands, B_b, T_b, N_eff):
        # truth magnitude
        mag = data[f'mag_true_{band}_lsst']

        # expected signal photons, over all visits
        c_b = factor * 10**(0.4*(25-mag)) * t_b * n_visit

        # expectedbackground photons, over all visits
        background = (b_b + sigma_inst2 + sigma_b2)*n_eff * n_visit
        # total expected photons
        mu = c_b + background
        
        # Observed number of photons in excess of the expected background.
        # This can go negative for faint magnitudes, indicating that the object is
        # not going to be detected
        n_obs = np.random.poisson(mu) - background

        # signal to noise, true and estimated values
        true_snr = mu / background**0.5
        obs_snr = n_obs / background

        # observed magnitude from inverting c_b expression above
        mag_obs = 25 - 2.5*np.log10(n_obs/factor/t_b)

        visible = np.isfinite(mag_obs)

        output[f'true_snr_{band}'] = true_snr
        output[f'snr_{band}'] = obs_snr
        output[f'mag_{band}_lsst'] = mag_obs

    return output


