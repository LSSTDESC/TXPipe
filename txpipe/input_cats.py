from ceci import PipelineStage
from descformats.tx import MetacalCatalog, HDFFile

# could also just load /global/projecta/projectdirs/lsst/groups/CS/descqa/catalog/ANL_AlphaQ_v3.0.hdf5


class TXDProtoDC2Mock(PipelineStage):
    """


    """
    name='TXDProtoDC2Mock'
    inputs = [
        ('response_model', HDFFile)
    ]
    outputs = [
        ('shear_catalog', MetacalCatalog),
        ('photometry_catalog', HDFFile),
    ]
    config_options = {
        'cat_name':'protoDC2_test', 
        'visits_per_band':165, 
        'snr_limit':4.0,
        'max_size': 99999999999999
        }

    def data_iterator(self, gc):

        cols = ['mag_true_u_lsst', 'mag_true_g_lsst', 
                'mag_true_r_lsst', 'mag_true_i_lsst', 
                'mag_true_z_lsst',
                'ra', 'dec',
                'ellipticity_1_true', 'ellipticity_2_true',
                'shear_1', 'shear_2',
                'size_true',
                'galaxy_id',
                ]

        it = gc.get_quantities(cols, return_iterator=True)
        for data in it:
            yield data

    def get_catalog_size(self, gc):
        import h5py
        filename = gc.get_catalog_info()['filename']
        print(f"Reading catalog size directly from {filename}")
        f = h5py.File(filename)
        n = f['galaxyProperties/ra'].size
        f.close()
        n = min(n, self.config['max_size'])
        return n

    def run(self):
        import GCRCatalogs
        cat_name = self.config['cat_name']
        self.bands = ('u','g', 'r', 'i', 'z')

        gc = GCRCatalogs.load_catalog(cat_name)
        N = self.get_catalog_size(gc)
        self.cat_size = N
        metacal_file = self.open_output('shear_catalog', clobber=True)
        photo_file = self.open_output('photometry_catalog', parallel=False)

        # This is the kind of thing that should go into
        # the DESCFormats stuff
        self.setup_photometry_output(photo_file)
        self.load_metacal_response_model()
        self.current_index = 0
        for data in self.data_iterator(gc):
            if len(data['galaxy_id'])+self.current_index > self.cat_size:
                cut = self.cat_size - self.current_index
                for name in list(data.keys()):
                    data[name] = data[name][:cut]
            mock_photometry = self.make_mock_photometry(data)
            mock_metacal = self.make_mock_metacal(data, mock_photometry)
            self.remove_undetected(mock_photometry, mock_metacal)
            self.write_photometry(photo_file, mock_photometry)
            self.write_metacal(metacal_file, mock_metacal)
            if self.current_index >= self.cat_size:
                break

            
        # Tidy up
        self.truncate_photometry(photo_file)
        photo_file.close()
        metacal_file.close()


    def setup_photometry_output(self, photo_file):
        # Get a list of all the column names
        cols = ['ra', 'dec']
        for band in self.bands:
            cols.append(f'mag_true_{band}_lsst')
            cols.append(f'true_snr_{band}')
            cols.append(f'mag_{band}_lsst')
            cols.append(f'mag_{band}_lsst_1p')
            cols.append(f'mag_{band}_lsst_1m')
            cols.append(f'mag_{band}_lsst_2p')
            cols.append(f'mag_{band}_lsst_2m')
            cols.append(f'snr_{band}')
            cols.append(f'snr_{band}_1p')
            cols.append(f'snr_{band}_1m')
            cols.append(f'snr_{band}_2p')
            cols.append(f'snr_{band}_2m')

        # Make group for all the photometry
        group = photo_file.create_group('photometry')

        # Extensible columns becase we don't know the size yet.
        # We will cut down the size at the end.
        for col in cols:
            group.create_dataset(col, (self.cat_size,), maxshape=(self.cat_size,), dtype='f8')

        # The only non-float column for now
        group.create_dataset('galaxy_id', (self.cat_size,), maxshape=(self.cat_size,), dtype='i8')
    

    def load_metacal_response_model(self):
        import scipy.interpolate
        import numpy as np
        model_file = self.open_input("response_model")
        snr_centers = model_file['R_model/log10_snr'][:]
        sz_centers = model_file['R_model/size'][:]
        R_mean = model_file['R_model/R_mean'][:]
        R_std = model_file['R_model/R_std'][:]
        model_file.close()

        snr_grid, sz_grid = np.meshgrid(snr_centers, sz_centers)
        self.R_spline=scipy.interpolate.SmoothBivariateSpline(snr_grid.T.flatten(), sz_grid.T.flatten(), R_mean.flatten(), w=R_std.flatten())
        self.Rstd_spline=scipy.interpolate.SmoothBivariateSpline(snr_grid.T.flatten(), sz_grid.T.flatten(), R_std.flatten())        


    def write_photometry(self, photo_file, mock_photometry):
        # Work out the range of data to output (since we will be
        # doing this in chunks)
        start = self.current_index
        n = len(mock_photometry['galaxy_id'])
        end = start + n

        # Save each column
        for name, col in mock_photometry.items():
            photo_file[f'photometry/{name}'][start:end] = col

        # Update starting point for next round
        self.current_index += n

    def write_metacal(self, metacal_file, metacal_data):
        import numpy as np
        dtype = [(name,val.dtype,val[0].shape) for (name,val) in sorted(metacal_data.items())]
        nobj = metacal_data['R'].size
        data = np.zeros(nobj, dtype)
        for key, val in metacal_data.items():
            data[key] = val

        already_created_ext = len(metacal_file)==2

        if already_created_ext:
            metacal_file[-1].append(data)
        else:
            metacal_file.write(data)
            


    def make_mock_photometry(self, data):
        # The visit count affects the overall noise levels
        n_visit = self.config['visits_per_band']
        # Do all the work in the function below
        photo = make_mock_photometry(n_visit, self.bands, data)
        return photo



    def make_mock_metacal(self, data, photo):
        """
        Generate a mock metacal table with noise added
        """

        # TODO: Write

        # These are the numbers from figure F1 of the DES Y1 shear catalog paper
        # (this version is not yet public but is awaiting a second referee response)
        import numpy as np

        # strategy here.
        # we have a S/N per band for each object.
        # get the total S/N in r,i,z since most shapes are measured there.
        # if it's > 5 then make a fake R_mean value based on the signal to noise and size
        # Then generate R_mean from R with some spread
        # 
        # Just use the true shear * R_mean for the estimated shear
        # (the noise on R will do the job of noise on shear)
        # Use R11 = R22 and R12 = R21 = 0

        # Overall SNR for the three bands usually used for shape measurement
        # We use the true SNR not the estimated one, though these are pretty close
        snr = (photo['snr_r']**2 + photo['snr_i'] + photo['snr_z'])**0.5
        snr_1p = (photo['snr_r_1p']**2 + photo['snr_i_1p'] + photo['snr_z_1p'])**0.5
        snr_1m = (photo['snr_r_1m']**2 + photo['snr_i_1m'] + photo['snr_z_1m'])**0.5
        snr_2p = (photo['snr_r_2p']**2 + photo['snr_i_2p'] + photo['snr_z_2p'])**0.5
        snr_2m = (photo['snr_r_2m']**2 + photo['snr_i_2m'] + photo['snr_z_2m'])**0.5

        nobj = snr.size

        log10_snr = np.log10(snr)

        size_hlr = data['size_true']
        size_sigma = size_hlr / np.sqrt(2*np.log(2))
        size_T = 2 * size_sigma**2

        psf_fwhm = 0.75
        psf_sigma = psf_fwhm/(2*np.sqrt(2*np.log(2)))
        psf_T = 2 * psf_sigma**2

        R_mean = self.R_spline(log10_snr, size_T, grid=False)
        R_std = self.Rstd_spline(log10_snr, size_T, grid=False)
        

#        response_mean = np.array([R_mean, R_mean])
        rho = 0.2  # correlation between size response and shear response.  chosen arbitrarily
#        response_covmat = np.array([[R_std**2,rho*R_std**2],[rho*R_std**2,R_std**2]])
#        R, R_size = R_mean + np.random.multivariate_normal(response_mean, response_covmat, nobj).T
        f = np.random.multivariate_normal([0.0,0.0], [[1.0,rho],[rho,1.0]], nobj).T
        R, R_size = f * R_std + R_mean

        flux_r = 10**0.4*(27 - photo['mag_r_lsst'])
        flux_i = 10**0.4*(27 - photo['mag_i_lsst'])
        flux_z = 10**0.4*(27 - photo['mag_z_lsst'])

        delta_gamma = 0.01
        
        shape_noise = 0.26
        eps  = np.random.normal(0,shape_noise,nobj) + 1.j * np.random.normal(0,shape_noise,nobj)
        g1 = data['shear_1']
        g2 = data['shear_2']
        g = g1 + 1j*g2
        e = (eps + g) / (1+g.conj()*eps)
        e1 = e.real
        e2 = e.imag
       

        output = {
            "R":R,
            "true_g": np.array([g1, g2]).T,
            "mcal_g": np.array([e1*R, e2*R]).T,
            "mcal_g_1p": np.array([(e1+delta_gamma)*R, e2*R]).T,
            "mcal_g_1m": np.array([(e1-delta_gamma)*R, e2*R]).T,
            "mcal_g_2p": np.array([e1*R, (e2+delta_gamma)*R]).T,
            "mcal_g_2m": np.array([e1*R, (e2-delta_gamma)*R]).T,
            "mcal_T": size_T,
            "mcal_T_1p": size_T + R_size*delta_gamma,
            "mcal_T_1m": size_T - R_size*delta_gamma,
            "mcal_T_2p": size_T + R_size*delta_gamma,
            "mcal_T_2m": size_T - R_size*delta_gamma,
            "mcal_s2n_r": snr,
            "mcal_s2n_r_1p": snr_1p,
            "mcal_s2n_r_1m": snr_1m,
            "mcal_s2n_r_2p": snr_2p,
            "mcal_s2n_r_2m": snr_2m,
            'mcal_mag': np.array([photo['mag_r_lsst'], photo['mag_i_lsst'], photo['mag_z_lsst']]).T,
            'mcal_flux': np.array([flux_r, flux_i, flux_z]).T,
            # not sure if this is right
            'mcal_flux_s2n': np.array([photo['snr_r'], photo['snr_i'], photo['snr_z']]).T,
            # These appear to be all zeros in the tract files.
            # possibly they should in fact all be ones.
            'mcal_weight': np.zeros(nobj),
            'mcal_gpsf': np.zeros(nobj),
            'mcal_Tpsf': np.repeat(psf_T, nobj),
            # Everything below here is wrong
            "mcal_g_cov": np.zeros((nobj,2,2)),
            "mcal_g_cov_1p": np.zeros((nobj,2,2)),
            "mcal_g_cov_1m": np.zeros((nobj,2,2)),
            "mcal_g_cov_2p": np.zeros((nobj,2,2)),
            "mcal_g_cov_2m": np.zeros((nobj,2,2)),
            "mcal_pars":np.zeros((nobj,6)),
            "mcal_pars_1p":np.zeros((nobj,6)),
            "mcal_pars_1m":np.zeros((nobj,6)),
            "mcal_pars_2p":np.zeros((nobj,6)),
            "mcal_pars_2m":np.zeros((nobj,6)),
            "mcal_pars_cov":np.zeros((nobj,6,6)),
            "mcal_pars_cov_1p":np.zeros((nobj,6,6)),
            "mcal_pars_cov_1m":np.zeros((nobj,6,6)),
            "mcal_pars_cov_2p":np.zeros((nobj,6,6)),
            "mcal_pars_cov_2m":np.zeros((nobj,6,6)),
            "mcal_flux_cov": np.zeros((nobj,2,2)),
            "mcal_T_err": np.zeros(nobj),
            "mcal_T_err_1p": np.zeros(nobj),
            "mcal_T_err_1m": np.zeros(nobj),
            "mcal_T_err_2p": np.zeros(nobj),
            "mcal_T_err_2m": np.zeros(nobj),
            "mcal_T_r": size_T,
            "mcal_T_r_1p": size_T,
            "mcal_T_r_1m": size_T,
            "mcal_T_r_2p": size_T,
            "mcal_T_r_2m": size_T,
            "mcal_logsb": np.zeros(nobj),

            }

        return output



    def remove_undetected(self, photo, metacal):
        import numpy as np
        snr_limit = self.config['snr_limit']
        detected = False

        # Check if detected in any band.  Makes a boolean array
        # Even though we started with just a single False.
        for band in self.bands:
            detected_in_band = photo[f'snr_{band}'] > snr_limit
            not_detected_in_band = ~detected_in_band
            # Set objects not detected in one band that are detected in another
            # to inf magnitude in that band, and the SNR to zero.
            photo[f'snr_{band}'][not_detected_in_band] = 0.0
            photo[f'mag_{band}_lsst'][not_detected_in_band] = np.inf

            # Record that we have detected this object at all
            detected |= detected_in_band


        # the protoDC2 sims have an edge with zero shear.
        # Remove it.
        print(metacal['true_g'].shape)
        zero_shear_edge = (abs(metacal['true_g'][:,0])==0) & (abs(metacal['true_g'][:,1])==0)
        print("Removing {} objects with identically zero shear in both terms".format(zero_shear_edge.sum()))

        detected &= (~zero_shear_edge)

        ndet = detected.sum()
        ntot = detected.size
        fract = ndet*100./ntot

        print(f"Detected {ndet} out of {ntot} objects ({fract:.1f}%)")
        # Remove all objects not detected in *any* band
        # make a copy of the keys with photo.keys so we are not
        # modifying during the iteration
        for key in list(photo.keys()): 
            photo[key] = photo[key][detected]

        for key in list(metacal.keys()):
            metacal[key] = metacal[key][detected]


    def truncate_photometry(self, photo_file):
        group = photo_file['photometry']
        cols = list(group.keys())
        for col in cols:
            group[col].resize((self.current_index,))


def make_mock_photometry(n_visit, bands, data):
    """
    Generate a mock photometric table with noise added

    This is mostly from LSE‐40 by 
    Zeljko Ivezic, Lynne Jones, and Robert Lupton
    retrieved here:
    http://faculty.washington.edu/ivezic/Teaching/Astr511/LSST_SNRdoc.pdf
    """

    import numpy as np

    output = {}
    nobj = data['galaxy_id'].size
    output['ra'] = data['ra']
    output['dec'] = data['dec']
    output['galaxy_id'] = data['galaxy_id']


    # Sky background, seeing FWHM, and system throughput, 
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
    D = 6.5 # primary mirror diameter in meters, from LSST key numbers page (effective clear diameter)
    # Not sure what effective clear diameter means but it's closer to the number used in the paper
    time = 30. # seconds per exposure, from LSST key numbers page
    sigma_b2 = 0.0 # error on background, just above eq 42

    # combination of these  used below, from various equations
    factor = 5455./gain * (D/6.5)**2 * (time/30.)

    nband = len(bands)
    mu = np.zeros(nband) # seems approx mean of response across bands, from HSC tract
    rho = 0.25  #  approx correlation between response in bands, from HSC tract
    sigma2 = 1.7**2 # approx variance of response, from HSC tract
    covmat = np.full((nband,nband), rho*sigma2)
    np.fill_diagonal(covmat, sigma2)
    mag_responses = np.random.multivariate_normal(mu, covmat, nobj).T
    delta_gamma = 0.01  # this is the half-delta gamma, i.e. gamma_+ - gamma_0
    # that's the right thing to use here because we are doing m+ = m0 + dm/dy*dy
    # Use the same response for gamma1 and gamma2


    for band, b_b, t_b, n_eff, mag_resp in zip(bands, B_b, T_b, N_eff, mag_responses):
        # truth magnitude
        mag = data[f'mag_true_{band}_lsst']
        output[f'mag_true_{band}_lsst'] = mag

        # expected signal photons, over all visits
        c_b = factor * 10**(0.4*(25-mag)) * t_b * n_visit

        # expected background photons, over all visits
        background = np.sqrt((b_b + sigma_inst2 + sigma_b2) * n_eff * n_visit)
        # total expected photons
        mu = c_b + background
        
        # Observed number of photons in excess of the expected background.
        # This can go negative for faint magnitudes, indicating that the object is
        # not going to be detected
        n_obs = np.random.poisson(mu) - background

        # signal to noise, true and estimated values
        true_snr = c_b / background
        obs_snr = n_obs / background

        # observed magnitude from inverting c_b expression above
        mag_obs = 25 - 2.5*np.log10(n_obs/factor/t_b/n_visit)

        output[f'true_snr_{band}'] = true_snr
        output[f'snr_{band}'] = obs_snr
        output[f'mag_{band}_lsst'] = mag_obs

        mag_obs_1p = mag_obs + mag_resp*delta_gamma
        mag_obs_1m = mag_obs - mag_resp*delta_gamma
        mag_obs_2p = mag_obs + mag_resp*delta_gamma
        mag_obs_2m = mag_obs - mag_resp*delta_gamma

        output[f'mag_{band}_lsst_1p'] = mag_obs_1p
        output[f'mag_{band}_lsst_1m'] = mag_obs_1m
        output[f'mag_{band}_lsst_2p'] = mag_obs_2p
        output[f'mag_{band}_lsst_2m'] = mag_obs_2m

        output[f'snr_{band}_1p'] = obs_snr * 10**(0.4*(mag_obs - mag_obs_1p))
        output[f'snr_{band}_1m'] = obs_snr * 10**(0.4*(mag_obs - mag_obs_1m))
        output[f'snr_{band}_2p'] = obs_snr * 10**(0.4*(mag_obs - mag_obs_2p))
        output[f'snr_{band}_2m'] = obs_snr * 10**(0.4*(mag_obs - mag_obs_2m))


    return output



def test():
    import numpy as np
    import pylab
    data = {
        'ra':None,
        'dec':None,
        'galaxy_id':None,
    }
    bands = ('u','g','r','i','z')
    n_visit=165
    M5 = [24.22, 25.17, 24.74, 24.38, 23.80]
    for b,m5 in zip(bands, M5):
        data[f'mag_true_{b}_lsst'] = np.repeat(m5, 10000)
    results = make_mock_photometry(n_visit, bands, data)
    pylab.hist(results['snr_r'], bins=50, histtype='step')
    pylab.savefig('snr_r.png')
