from ceci import PipelineStage
from .data_types import MetacalCatalog, HDFFile


class TXProtoDC2Mock(PipelineStage):
    """
    This stage simulates metacal data and metacalibrated
    photometry measurements, starting from a cosmology catalogs
    of the kind used as an input to DC2 image and obs-catalog simulations.

    This is mainly useful for testing infrastructure in advance
    of the DC2 catalogs being available, but might also be handy
    for starting from a purer simulation.
    """
    name='TXProtoDC2Mock'

    inputs = [
        ('response_model', HDFFile)
    ]

    outputs = [
        ('shear_catalog', MetacalCatalog),
        ('photometry_catalog', HDFFile),
    ]

    config_options = {
        'cat_name':'protoDC2_test',
        'visits_per_band':165,  # used in the noise simulation
        'snr_limit':4.0,  # used to decide what objects to cut out
        'max_size': 99999999999999  #for testing on smaller catalogs
        }

    def data_iterator(self, gc):

        # Columns we need from the cosmo simulation
        cols = ['mag_true_u_lsst', 'mag_true_g_lsst', 
                'mag_true_r_lsst', 'mag_true_i_lsst', 
                'mag_true_z_lsst', 'mag_true_y_lsst',
                'ra', 'dec',
                'ellipticity_1_true', 'ellipticity_2_true',
                'shear_1', 'shear_2',
                'size_true',
                'galaxy_id',
                ]

        it = gc.get_quantities(cols, return_iterator=True)
        for data in it:
            yield data

    def run(self):
        import GCRCatalogs
        cat_name = self.config['cat_name']
        print(f"Loading from catalog {cat_name}")
        self.bands = ('u','g', 'r', 'i', 'z', 'y')

        # Load the input catalog (this is lazy)
        gc = GCRCatalogs.load_catalog(cat_name)

        # Get the size, and optionally cut down to a smaller
        # size if we want to test
        N = len(gc)
        self.cat_size = min(N, self.config['max_size'])

        select_fraction = (1.0 * self.cat_size)/N

        if self.cat_size != N:
            print("Will select a fraction of approx {select_fraction:.2f} of objects")

        # Prepare output files
        metacal_file = self.open_output('shear_catalog', clobber=True)
        photo_file = self.open_output('photometry_catalog', parallel=False)
        self.setup_photometry_output(photo_file)

        # Load the metacal response file
        self.load_metacal_response_model()

        # Keep track of catalog position
        self.current_index = 0

        # Loop through chunks of 
        for data in self.data_iterator(gc):
            some_col = list(data.keys())[0]
            chunk_size = len(data[some_col])
            s = self.current_index
            e = s + chunk_size
            print(f"Read chunk {s} - {e} or {self.cat_size}")
            # Select a random fraction of the catalog
            if self.cat_size != N:
                select = np.random.binomial(chunk_size, select_fraction)
                for name in list(data.keys()):
                    data[name] = data[name][select]

            # Simulate the various output data sets
            mock_photometry = self.make_mock_photometry(data)
            mock_metacal = self.make_mock_metacal(data, mock_photometry)

            # Cut out any objects too faint to be detected and measured
            self.remove_undetected(mock_photometry, mock_metacal)

            # Save all output
            self.write_photometry(photo_file, mock_photometry)
            self.write_metacal(metacal_file, mock_metacal)

            # Break early if we are cutting down the
            # catalog
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
            cols.append(f'mag_err_{band}_lsst')
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
        group.create_dataset('id', (self.cat_size,), maxshape=(self.cat_size,), dtype='i8')
    

    def load_metacal_response_model(self):
        """
        Load an HDF file containing the response model
        R(log10(snr), size)
        R_std(log10(snr), size)

        where R is the mean metacal response in a bin and
        R_std is its standard deviation.

        So far only one of these files exists!

        """
        import scipy.interpolate
        import numpy as np
        model_file = self.open_input("response_model")
        snr_centers = model_file['R_model/log10_snr'][:]
        sz_centers = model_file['R_model/size'][:]
        R_mean = model_file['R_model/R_mean'][:]
        R_std = model_file['R_model/R_std'][:]
        model_file.close()

        # Save a 2D spline
        snr_grid, sz_grid = np.meshgrid(snr_centers, sz_centers)
        self.R_spline=scipy.interpolate.SmoothBivariateSpline(snr_grid.T.flatten(), sz_grid.T.flatten(), R_mean.flatten(), w=R_std.flatten())
        self.Rstd_spline=scipy.interpolate.SmoothBivariateSpline(snr_grid.T.flatten(), sz_grid.T.flatten(), R_std.flatten())        


    def write_photometry(self, photo_file, mock_photometry):
        """
        Save the photometry we have just simulated to disc

        Parameters
        ----------

        photo_file: HDF File object

        mock_photometry: dict
            Dictionary of simulated photometry

        """
        # Work out the range of data to output (since we will be
        # doing this in chunks)
        start = self.current_index
        n = len(mock_photometry['id'])
        end = start + n

        # Save each column
        for name, col in mock_photometry.items():
            photo_file[f'photometry/{name}'][start:end] = col

        # Update starting point for next round
        self.current_index += n

    def write_metacal(self, metacal_file, metacal_data):
        """
        Save the metacal data we have just simulated to disc

        Parameters
        ----------

        metacal_file: fitsio FITS File object

        metacal_data: dict
            Dictionary of arrays of simulated metacal values

        """
        import numpy as np

        # Get the name and data type for eah column
        # We sort these so that they are always in the same order,
        # because we will just be appending the data for
        # subsequent calls
        dtype = [(name,val.dtype,val[0].shape) for (name,val) in sorted(metacal_data.items())]
        nobj = metacal_data['R'].size

        # Make a numpy structured array for this data
        # and fill it in with our values
        data = np.zeros(nobj, dtype)
        for key, val in metacal_data.items():
            data[key] = val

        # The first time we call this we will make an extension
        # to contain the data.  Subsequent times we just append
        # to that extension.
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
        Generate a mock metacal table.
        This is a long and complicated function unfortunately.

        Throughout we assume a single R = R11 = R22, with R12 = R21 = 0
        
        Parameters
        ----------
        data: dict
            Dictionary of arrays read from the input cosmo catalog
        photo: dict
            Dictionary of arrays generated to simulate photometry
        """


        # These are the numbers from figure F1 of the DES Y1 shear catalog paper
        # (this version is not yet public but is awaiting a second referee response)
        import numpy as np

        # Overall SNR for the three bands usually used for shape measurement
        # We use the true SNR not the estimated one, though these are pretty close
        snr = (
            photo['snr_r']**2 + 
            photo['snr_i']**2 + 
            photo['snr_z']**2
            )**0.5
        snr_1p = (photo['snr_r_1p']**2 + photo['snr_i_1p'] + photo['snr_z_1p'])**0.5
        snr_1m = (photo['snr_r_1m']**2 + photo['snr_i_1m'] + photo['snr_z_1m'])**0.5
        snr_2p = (photo['snr_r_2p']**2 + photo['snr_i_2p'] + photo['snr_z_2p'])**0.5
        snr_2m = (photo['snr_r_2m']**2 + photo['snr_i_2m'] + photo['snr_z_2m'])**0.5

        nobj = snr.size

        log10_snr = np.log10(snr)

        # Convert from the half-light radius which is in the 
        # input catalogs to a sigma.  Do this by pretending
        # that it is a Gaussian.  This is clearly wrong, and
        # if this causes major errors in the size cuts we may
        # have to modify this
        size_hlr = data['size_true']
        size_sigma = size_hlr / np.sqrt(2*np.log(2))
        size_T = 2 * size_sigma**2

        # Use a fixed PSF across all the objects
        psf_fwhm = 0.75
        psf_sigma = psf_fwhm/(2*np.sqrt(2*np.log(2)))
        psf_T = 2 * psf_sigma**2

        # Use the response model to get a reasonable response
        # value for this size and SNR
        R_mean = self.R_spline(log10_snr, size_T, grid=False)
        R_std = self.Rstd_spline(log10_snr, size_T, grid=False)
        
        # Assume a 0.2 correlation between the size response
        # and the shear response.
        rho = 0.2
        f = np.random.multivariate_normal([0.0,0.0], [[1.0,rho],[rho,1.0]], nobj).T
        R, R_size = f * R_std + R_mean

        # Convert magnitudes to fluxes according to the baseline
        # use in the metacal numbers
        flux_r = 10**0.4*(27 - photo['mag_r_lsst'])
        flux_i = 10**0.4*(27 - photo['mag_i_lsst'])
        flux_z = 10**0.4*(27 - photo['mag_z_lsst'])

        # Note that this is delta_gamma not 2*delta_gamma, because
        # of how we use it below
        delta_gamma = 0.01
        
        # Use a fixed shape noise per component to generate 
        # an overall 
        shape_noise = 0.26
        eps  = np.random.normal(0,shape_noise,nobj) + 1.j * np.random.normal(0,shape_noise,nobj)
        # True shears without shape noise
        g1 = data['shear_1']
        g2 = data['shear_2']
        # Do the full combination of (g,epsilon) -> e, not the approx one
        g = g1 + 1j*g2
        e = (eps + g) / (1+g.conj()*eps)
        e1 = e.real
        e2 = e.imag
    
        # Now collect together everything to go into the metacal
        # file
        output = {
            # Basic values
            "R":R,
            'id': photo['id'],
            'ra': photo['ra'],
            'dec': photo['dec'],
            # Keep the truth value just in case
            "true_g": np.array([g1, g2]).T,
            # Shears, taken by scaling the overall ellipticity by the response
            # I think this is the correct thing to use here
            "mcal_g": np.array([e1*R, e2*R]).T,
            # metacalibrated variants - we add the shear and then 
            # apply the response
            "mcal_g_1p": np.array([(e1+delta_gamma)*R, e2*R]).T,
            "mcal_g_1m": np.array([(e1-delta_gamma)*R, e2*R]).T,
            "mcal_g_2p": np.array([e1*R, (e2+delta_gamma)*R]).T,
            "mcal_g_2m": np.array([e1*R, (e2-delta_gamma)*R]).T,
            # Size parameter and metacal variants
            "mcal_T": size_T,
            "mcal_T_1p": size_T + R_size*delta_gamma,
            "mcal_T_1m": size_T - R_size*delta_gamma,
            "mcal_T_2p": size_T + R_size*delta_gamma,
            "mcal_T_2m": size_T - R_size*delta_gamma,
            # Rounded SNR values, as used in the selection,
            # together with the metacal variants
            "mcal_s2n_r": snr,
            "mcal_s2n_r_1p": snr_1p,
            "mcal_s2n_r_1m": snr_1m,
            "mcal_s2n_r_2p": snr_2p,
            "mcal_s2n_r_2m": snr_2m,
            # Magntiudes and fluxes, just copied from the inputs.
            # Note that we won't use these directly later as we instead
            # use stuff from the photometry file
            'mcal_mag': np.array([
                photo['mag_r_lsst'], 
                photo['mag_i_lsst'], 
                photo['mag_z_lsst']
                ]).T,
            'mcal_flux': np.array([
                flux_r,
                flux_i,
                flux_z,
            ]).T,
            # not sure if this is right
            'mcal_flux_s2n': np.array([
                photo['snr_r'], 
                photo['snr_i'], 
                photo['snr_z']
            ]).T,
            # mcal_weight appears to be all zeros in the tract files.
            # possibly they should in fact all be ones.
            'mcal_weight': np.zeros(nobj),
            # Fixed PSF parameters - all round with same size
            'mcal_gpsf': np.zeros((nobj,2)),
            'mcal_Tpsf': np.repeat(psf_T, nobj),
            # Everything that gets this far should be used, so flag=0
            'mcal_flags': np.zeros(nobj, dtype=int),
            # Pretend these are the same as the standard sizes
            # actually they are measured on a rounded version
            "mcal_T_r": size_T,
            "mcal_T_r_1p": size_T + R_size*delta_gamma,
            "mcal_T_r_1m": size_T - R_size*delta_gamma,
            "mcal_T_r_2p": size_T + R_size*delta_gamma,
            "mcal_T_r_2m": size_T - R_size*delta_gamma,
            # Everything below here is wrong
            # Some of these are probably important!
            # For now I'm leaving them all as zero 
            # because we're not currently using them in the
            # main pipeline
            "mcal_g_cov": np.zeros((nobj,2,2)),
            "mcal_g_cov_1p": np.zeros((nobj,2,2)),
            "mcal_g_cov_1m": np.zeros((nobj,2,2)),
            "mcal_g_cov_2p": np.zeros((nobj,2,2)),
            "mcal_g_cov_2m": np.zeros((nobj,2,2)),
            # These are fit parameters and their covariances
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
            # This is likely to be important
            "mcal_T_err": np.zeros(nobj),
            "mcal_T_err_1p": np.zeros(nobj),
            "mcal_T_err_1m": np.zeros(nobj),
            "mcal_T_err_2p": np.zeros(nobj),
            "mcal_T_err_2m": np.zeros(nobj),
            # Surface brightness
            "mcal_logsb": np.zeros(nobj),

            }

        return output



    def remove_undetected(self, photo, metacal):
        """
        Strip out any undetected objects from the two
        simulated data sets.

        Use a configuration parameter snr_limit to decide
        on the detection limit.
        """
        import numpy as np
        snr_limit = self.config['snr_limit']

        # This will become a boolean array in a minute when
        # we OR it with an array
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

        # Print out interesting information
        ndet = detected.sum()
        ntot = detected.size
        fract = ndet*100./ntot
        print(f"Detected {ndet} out of {ntot} objects ({fract:.1f}%)")

        # Remove all objects not detected in *any* band
        # make a copy of the keys with list(photo.keys()) so we are not
        # modifying during the iteration
        for key in list(photo.keys()): 
            photo[key] = photo[key][detected]

        for key in list(metacal.keys()):
            metacal[key] = metacal[key][detected]


    def truncate_photometry(self, photo_file):
        """
        Cut down the output photometry file to its final 
        size.

        Not sure why I did this as the size should be
        the same.  I must have had a reason.
        """
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
    output['id'] = data['galaxy_id']


    # Sky background, seeing FWHM, and system throughput, 
    # all from table 2 of Ivezic, Jones, & Lupton
    B_b = np.array([85.07, 467.9, 1085.2, 1800.3, 2775.7, 3614.3])
    fwhm = np.array([0.77, 0.73, 0.70, 0.67, 0.65, 0.63])
    T_b = np.array([0.0379, 0.1493, 0.1386, 0.1198, 0.0838, 0.0413])


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

    # Fake some metacal responses
    mag_responses = generate_mock_metacal_mag_responses(bands, nobj)

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
        n_obs_err = np.sqrt(mu)

        # signal to noise, true and estimated values
        true_snr = c_b / background
        obs_snr = n_obs / background

        # observed magnitude from inverting c_b expression above
        mag_obs = 25 - 2.5*np.log10(n_obs/factor/t_b/n_visit)

        # converting error on n_obs to error on mag
        mag_err = 2.5/np.log(10.) / obs_snr

        output[f'true_snr_{band}'] = true_snr
        output[f'snr_{band}'] = obs_snr
        output[f'mag_{band}_lsst'] = mag_obs
        output[f'mag_err_{band}_lsst'] = mag_err

        mag_obs_1p = mag_obs + mag_resp*delta_gamma
        mag_obs_1m = mag_obs - mag_resp*delta_gamma
        mag_obs_2p = mag_obs + mag_resp*delta_gamma
        mag_obs_2m = mag_obs - mag_resp*delta_gamma

        output[f'mag_{band}_lsst_1p'] = mag_obs_1p
        output[f'mag_{band}_lsst_1m'] = mag_obs_1m
        output[f'mag_{band}_lsst_2p'] = mag_obs_2p
        output[f'mag_{band}_lsst_2m'] = mag_obs_2m

        # Scale the SNR values according the to change in magnitude.r
        output[f'snr_{band}_1p'] = obs_snr * 10**(0.4*(mag_obs - mag_obs_1p))
        output[f'snr_{band}_1m'] = obs_snr * 10**(0.4*(mag_obs - mag_obs_1m))
        output[f'snr_{band}_2p'] = obs_snr * 10**(0.4*(mag_obs - mag_obs_2p))
        output[f'snr_{band}_2m'] = obs_snr * 10**(0.4*(mag_obs - mag_obs_2m))


    return output



def generate_mock_metacal_mag_responses(bands, nobj):
    import numpy as np
    nband = len(bands)
    mu = np.zeros(nband) # seems approx mean of response across bands, from HSC tract
    rho = 0.25  #  approx correlation between response in bands, from HSC tract
    sigma2 = 1.7**2 # approx variance of response, from HSC tract
    covmat = np.full((nband,nband), rho*sigma2)
    np.fill_diagonal(covmat, sigma2)
    mag_responses = np.random.multivariate_normal(mu, covmat, nobj).T
    return mag_responses


class TXGCRMockMetacal(PipelineStage):
    """
    Use real photometry from the DRP (output of DM stack) in merged form,
    but fake metacal responses for the magnitudes.

    TODO: Shapes!
    """
    name = "TXGCRMockMetacal"

    inputs = [
        ('response_model', HDFFile),
    ]

    outputs = [
        ('photometry_catalog', HDFFile),
    ]

    config_options = {
        'cat_name':'dc2_coadd_run1.1p',
        'snr_limit':4.0,  # used to decide what objects to cut out
        'max_size': 99999999999999  #for testing on smaller catalogs
        }

    def count_rows(self, cat_name):
        counts = {
            'dc2_coadd_run1.1p':6892380,
        }
        n = counts.get(cat_name)
        if n is None:
            raise ValueError("Sorry - there is no way to count the number of rows in a GCR catalog yet, so we have to hard-code them.  And your catalog we don't know.")
        return n

    def run(self):
        #input_filename="/global/cscratch1/sd/jchiang8/desc/HSC/merged_tract_8524.hdf5"
        import GCRCatalogs
        cat_name = self.config['cat_name']
        self.bands = 'ugrizy'

        self.load_metacal_response_model()

        # Load the input catalog (this is lazy)
        gc = GCRCatalogs.load_catalog(cat_name, {'md5': None})


        n = self.count_rows(cat_name)
        print(f"Found {n} objects in catalog {cat_name}")


        # Put everything under the "photometry" section
        output_photo = self.open_output('photometry_catalog')
        photo = output_photo.create_group('photometry')

        # The merged DRP file is already split into chunks internally - we load those 
        # chunks one by one
        for (start, end, data) in self.generate_fake_metacalibrated_photometry(gc):

            # For the first chunk we create the output data space, 
            # because then we know what all the columns are
            if start==0:
                for colname,col in data.items():
                    photo.create_dataset(colname, (n,), dtype=col.dtype)
                    print(f" Column: {colname}")

            # Output this chunk
            print(f" Saving rows {start}:{end}")
            for colname,col in data.items():
                photo[colname][start:end] = col

        output_photo.close()


    def generate_fake_metacalibrated_photometry(self, gc):
        import h5py
        import pandas
        import warnings


        mag_names = [f'mag_{b}' for b in self.bands]
        err_names = [f'magerr_{b}' for b in self.bands]
        snr_names = [f'snr_{b}_cModel' for b in self.bands]
        # Columns we need from the cosmo simulation
        cols = mag_names + err_names + snr_names
        start = 0
        delta_gamma = 0.01

        for data in gc.get_quantities(cols, return_iterator=True):
            n = len(data[mag_names[0]])
            end = start + n

            # Fake magnitude responses
            mag_responses = generate_mock_metacal_mag_responses(self.bands, n)

            output = {}
            for band, mag_resp in zip(self.bands, mag_responses):
                # Mags and SNR are in the catalog already
                mag_obs = data[f'mag_{band}']
                snr     = data[f'snr_{band}_cModel']
                mag_err = data[f'magerr_{band}']

                output[f'mag_{band}']    = mag_obs
                output[f'magerr_{band}'] = mag_err
                output[f'snr_{band}']    = snr

                # Generate the metacalibrated values
                mag_obs_1p = mag_obs + mag_resp*delta_gamma
                mag_obs_1m = mag_obs - mag_resp*delta_gamma
                mag_obs_2p = mag_obs + mag_resp*delta_gamma
                mag_obs_2m = mag_obs - mag_resp*delta_gamma

                # Save them, but we will also need them to generate
                # metacalibrated SNRs below
                output[f'mag_{band}_1p'] = mag_obs_1p
                output[f'mag_{band}_1m'] = mag_obs_1m
                output[f'mag_{band}_2p'] = mag_obs_2p
                output[f'mag_{band}_2m'] = mag_obs_2m

                # Scale the SNR values according the to change in magnitude.
                output[f'snr_{band}_1p'] = snr * 10**(0.4*(mag_obs - mag_obs_1p))
                output[f'snr_{band}_1m'] = snr * 10**(0.4*(mag_obs - mag_obs_1m))
                output[f'snr_{band}_2p'] = snr * 10**(0.4*(mag_obs - mag_obs_2p))
                output[f'snr_{band}_2m'] = snr * 10**(0.4*(mag_obs - mag_obs_2m))

            yield start, end, output
            start = end





    def load_metacal_response_model(self):
        """
        Load an HDF file containing the response model
        R(log10(snr), size)
        R_std(log10(snr), size)

        where R is the mean metacal response in a bin and
        R_std is its standard deviation.

        So far only one of these files exists!

        """
        import scipy.interpolate
        import numpy as np
        model_file = self.open_input("response_model")
        snr_centers = model_file['R_model/log10_snr'][:]
        sz_centers = model_file['R_model/size'][:]
        R_mean = model_file['R_model/R_mean'][:]
        R_std = model_file['R_model/R_std'][:]
        model_file.close()

        # Save a 2D spline
        snr_grid, sz_grid = np.meshgrid(snr_centers, sz_centers)
        self.R_spline=scipy.interpolate.SmoothBivariateSpline(snr_grid.T.flatten(), sz_grid.T.flatten(), R_mean.flatten(), w=R_std.flatten())
        self.Rstd_spline=scipy.interpolate.SmoothBivariateSpline(snr_grid.T.flatten(), sz_grid.T.flatten(), R_std.flatten())        



def test():
    import numpy as np
    import pylab
    data = {
        'ra':None,
        'dec':None,
        'galaxy_id':None,
    }
    bands = 'ugrizy'
    n_visit=165
    M5 = [24.22, 25.17, 24.74, 24.38, 23.80]
    for b,m5 in zip(bands, M5):
        data[f'mag_true_{b}_lsst'] = np.repeat(m5, 10000)
    results = make_mock_photometry(n_visit, bands, data)
    pylab.hist(results['snr_r'], bins=50, histtype='step')
    pylab.savefig('snr_r.png')
