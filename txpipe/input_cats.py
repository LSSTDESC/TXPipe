from .base_stage import PipelineStage
from .data_types import MetacalCatalog, HDFFile
from .utils.metacal import metacal_band_variants, metacal_variants
import numpy as np
from .utils.timer import Timer

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
        'max_size': 99999999999999,  #for testing on smaller catalogs
        'extra_cols': "", # string-separated list of columns to include
        'max_npix':99999999999999,
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
        # Add any extra requestd columns
        cols += self.config['extra_cols'].split()

        it = gc.get_quantities(cols, return_iterator=True)
        nfile = len(gc._file_list) if hasattr(gc, '_file_list') else 0

        for i, data in enumerate(it):
            if nfile:
                j = i+1
                print(f"Loading chunk {j}/{nfile}")
            yield data

    def run(self):
        import GCRCatalogs
        cat_name = self.config['cat_name']
        self.bands = ('u', 'g', 'r', 'i', 'z', 'y')

        if self.rank == 0:
            print(f"Loading from catalog {cat_name}")
            # Load the input catalog (this is lazy)
            # For testing we may want to cut down to a smaller number of pixels.
            # This is separate from the split by processor later on
            all_healpix_pixels = GCRCatalogs.available_catalogs[cat_name]['healpix_pixels']
            max_npix = self.config['max_npix']
            if max_npix != 99999999999999:
                print(f"Cutting down initial catalog to {max_npix} healpix pixels")
                all_healpix_pixels = all_healpix_pixels[:max_npix]
            # complete_cat = GCRCatalogs.load_catalog(cat_name, {'healpix_pixels':all_healpix_pixels})
            # print(f"Loaded overall catalog {cat_name}")

            # # Get the size, and optionally cut down to a smaller
            # # size if we want to test
            # N = len(complete_cat)
            # print(f"Measured catalog length: {N}")

        else:
            N = 0
            all_healpix_pixels = None

        if self.comm:
            # Split up the pixels to load among the processors
            all_healpix_pixels = self.comm.bcast(all_healpix_pixels)
            all_npix = len(all_healpix_pixels)
            my_healpix_pixels = all_healpix_pixels[self.rank::self.size]
            my_npix = len(my_healpix_pixels)

            # Load the catalog for this processor
            print(f"Rank {self.rank} loading catalog with {my_npix} pixels from total {all_npix}.")
            gc = GCRCatalogs.load_catalog(cat_name, {'healpix_pixels':my_healpix_pixels})

            # Work out my local length and the total length (from the sum of all the local lengths)
            my_N = len(gc)
            N = self.comm.allreduce(my_N)
            print(f"Rank {self.rank}: loaded. Have {my_N} objects from total {N}")

        else:
            all_npix = len(all_healpix_pixels)
            print(f"Rank {self.rank} loading catalog with all {all_npix} pixels.")
            gc = GCRCatalogs.load_catalog(cat_name, {'healpix_pixels':all_healpix_pixels})
            N = my_N = len(gc)
            print(f"Rank {self.rank} loaded: length = {N}.")

        target_size = min(N, self.config['max_size'])
        select_fraction = target_size / N

        if target_size != N:
            print(f"Will select approx {100*select_fraction:.2f}% of objects ({target_size})")


        # Prepare output files
        metacal_file = self.open_output('shear_catalog', parallel=self.is_mpi())
        photo_file = self.open_output('photometry_catalog', parallel=self.is_mpi())
        photo_cols = self.setup_photometry_output(photo_file, target_size)
        metacal_cols = self.setup_metacal_output(metacal_file, target_size)


        # Load the metacal response file
        self.load_metacal_response_model()

        # Keep track of catalog position
        start = 0
        count = 0

        # Loop through chunks of 
        for data in self.data_iterator(gc):
            # The initial chunk size, of all the input data.
            # This will be reduced later as we remove objects
            some_col = list(data.keys())[0]
            chunk_size = len(data[some_col])
            print(f"Process {self.rank} read chunk {count} - {count+chunk_size} of {my_N}")
            count += chunk_size
            # Select a random fraction of the catalog if we are cutting down
            # We can't just take the earliest galaxies because they are ordered
            # by redshift
            if target_size != N:
                select = np.random.uniform(size=chunk_size) < select_fraction
                nselect = select.sum()
                print(f"Cutting down to {nselect}/{chunk_size} objects")
                for name in list(data.keys()):
                    data[name] = data[name][select]

            # Simulate the various output data sets
            mock_photometry = self.make_mock_photometry(data)
            mock_metacal = self.make_mock_metacal(data, mock_photometry)

            # Cut out any objects too faint to be detected and measured
            self.remove_undetected(data, mock_photometry, mock_metacal)
            # The chunk size has now changed
            some_col = list(mock_photometry.keys())[0]
            chunk_size = len(mock_photometry[some_col])


            # start is where this process should start writing this
            # chunk of data.  end is where the final process will finish
            # writing, becoming the starting point for the whole next
            # chunk over all the processes
            start, end = self.next_output_indices(start, chunk_size)

            # Save all output
            self.write_output(start, target_size, photo_cols, metacal_cols, photo_file, mock_photometry, metacal_file, mock_metacal)

            # The next iteration starts writing where the current one ends.
            start = end
            
            if start >= target_size:
                break


            
        # Tidy up
        #self.truncate_output(photo_file, metacal_file, end)
        photo_file.close()
        metacal_file.close()

    def next_output_indices(self, start, chunk_size):
        if self.comm is None:
            end = start + chunk_size
        else:
            all_indices = self.comm.allgather(chunk_size)
            starting_points = np.concatenate(([0], np.cumsum(all_indices)))
            # use the old start to find the end point.
            # the final starting point (not used below, since it is larger
            # than the largest self.rank value) is the total data length
            end = start + starting_points[-1]
            start = start + starting_points[self.rank]
        print(f"- Rank {self.rank} writing output to {start}-{start+chunk_size}")
        return start, end



    def setup_photometry_output(self, photo_file, target_size):
        from .utils.hdf_tools import create_dataset_early_allocated
        # Get a list of all the column names
        cols = ['ra', 'dec']
        for band in self.bands:
            cols.append(f'{band}_mag')
            cols.append(f'{band}_mag_err')
            cols.append(f'snr_{band}')

        for col in self.config['extra_cols'].split():
            cols.append(col)

        # Make group for all the photometry
        group = photo_file.create_group('photometry')

        # Extensible columns becase we don't know the size yet.
        # We will cut down the size at the end.
        for col in cols:
            create_dataset_early_allocated(group, col, target_size, 'f8')

        # The only non-float column for now
        create_dataset_early_allocated(group, 'id', target_size, 'i8')

        return cols + ['id']



    def setup_metacal_output(self, metacal_file, target_size):
        from .utils.hdf_tools import create_dataset_early_allocated

        # Get a list of all the column names
        cols = (
            ['ra', 'dec', 'mcal_psf_g1', 'mcal_psf_g2', 'mcal_psf_T_mean']
            + metacal_variants('mcal_g1', 'mcal_g2', 'mcal_T', 'mcal_s2n',  'mcal_T_err')
            + metacal_band_variants('riz', 'mcal_mag', 'mcal_mag_err')
        )

        cols += ['true_g1', 'true_g2']

        # Make group for all the photometry
        group = metacal_file.create_group('metacal')

        # Extensible columns becase we don't know the size yet.
        # We will cut down the size at the end.
        

        for col in cols:
            create_dataset_early_allocated(group, col, target_size, 'f8')

        create_dataset_early_allocated(group, 'id', target_size, 'i8')
        create_dataset_early_allocated(group, 'mcal_flags', target_size, 'i4')

        return cols + ['id',  'mcal_flags']
        

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


    def write_output(self, start, target_size, photo_cols, metacal_cols, photo_file, photo_data, metacal_file, metacal_data):
        """
        Save the photometry we have just simulated to disc

        Parameters
        ----------

        photo_file: HDF File object

        metacal_file: HDF File object

        photo_data: dict
            Dictionary of simulated photometry

        metacal_data: dict
            Dictionary of simulated metacal data

        """
        # Work out the range of data to output (since we will be
        # doing this in chunks). If we have cut down to a random
        # subset of the catalog then we may have gone over the
        # target length, depending on the random selection
        n = len(photo_data['id'])
        end = min(start + n, target_size)

        assert photo_data['id'].min()>0

        # Save each column
        for name in photo_cols:
            photo_file[f'photometry/{name}'][start:end] = photo_data[name]

        for name in metacal_cols:
            metacal_file[f'metacal/{name}'][start:end] = metacal_data[name]

    def make_mock_photometry(self, data):
        # The visit count affects the overall noise levels
        n_visit = self.config['visits_per_band']
        # Do all the work in the function below
        photo = make_mock_photometry(n_visit, self.bands, data)

        for col in self.config['extra_cols'].split():
            photo[col] = data[col]
        
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
        flux_r = 10**0.4*(27 - photo['r_mag'])
        flux_i = 10**0.4*(27 - photo['i_mag'])
        flux_z = 10**0.4*(27 - photo['z_mag'])

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
    
        zero = np.zeros(nobj)
        # Now collect together everything to go into the metacal
        # file
        output = {
            # Basic values
            'id': photo['id'],
            'ra': photo['ra'],
            'dec': photo['dec'],
            # Keep the truth value just in case
            "true_g1": g1,
            "true_g2": g2,

            # g1
            "mcal_g1": e1*R,
            "mcal_g1_1p": (e1+delta_gamma)*R,
            "mcal_g1_1m": (e1-delta_gamma)*R,
            "mcal_g1_2p": e1*R,
            "mcal_g1_2m": e1*R,

            # g2
            "mcal_g2": e1*R,
            "mcal_g2_1p": e2*R,
            "mcal_g2_1m": e2*R,
            "mcal_g2_2p": (e2+delta_gamma)*R,
            "mcal_g2_2m": (e2-delta_gamma)*R,

            # T
            "mcal_T": size_T,
            "mcal_T_1p": size_T + R_size*delta_gamma,
            "mcal_T_1m": size_T - R_size*delta_gamma,
            "mcal_T_2p": size_T + R_size*delta_gamma,
            "mcal_T_2m": size_T - R_size*delta_gamma,

            # Terr
            "mcal_T_err":    zero,
            "mcal_T_err_1p": zero,
            "mcal_T_err_1m": zero,
            "mcal_T_err_2p": zero,
            "mcal_T_err_2m": zero,

            # size 
            "mcal_s2n": snr,
            "mcal_s2n_1p": snr_1p,
            "mcal_s2n_1m": snr_1m,
            "mcal_s2n_2p": snr_2p,
            "mcal_s2n_2m": snr_2m,

            # Magntiudes and fluxes, just copied from the inputs.
            'mcal_mag_r': photo['r_mag'],
            'mcal_mag_i': photo['i_mag'],
            'mcal_mag_z': photo['z_mag'],

            'mcal_mag_err_r': photo['r_mag_err'],
            'mcal_mag_err_i': photo['i_mag_err'],
            'mcal_mag_err_z': photo['z_mag_err'],

            'mcal_mag_r_1p': photo['r_mag_1p'],
            'mcal_mag_r_2p': photo['r_mag_2p'],
            'mcal_mag_r_1m': photo['r_mag_1m'],
            'mcal_mag_r_2m': photo['r_mag_2m'],

            'mcal_mag_i_1p': photo['i_mag_1p'],
            'mcal_mag_i_2p': photo['i_mag_2p'],
            'mcal_mag_i_1m': photo['i_mag_1m'],
            'mcal_mag_i_2m': photo['i_mag_2m'],
            
            'mcal_mag_z_1p': photo['z_mag_1p'],
            'mcal_mag_z_2p': photo['z_mag_2p'],
            'mcal_mag_z_1m': photo['z_mag_1m'],
            'mcal_mag_z_2m': photo['z_mag_2m'],

            'mcal_mag_err_r_1p': photo['r_mag_err'],
            'mcal_mag_err_r_2p': photo['r_mag_err'],
            'mcal_mag_err_r_1m': photo['r_mag_err'],
            'mcal_mag_err_r_2m': photo['r_mag_err'],

            'mcal_mag_err_i_1p': photo['i_mag_err'],
            'mcal_mag_err_i_2p': photo['i_mag_err'],
            'mcal_mag_err_i_1m': photo['i_mag_err'],
            'mcal_mag_err_i_2m': photo['i_mag_err'],

            'mcal_mag_err_z_1p': photo['z_mag_err'],
            'mcal_mag_err_z_2p': photo['z_mag_err'],
            'mcal_mag_err_z_1m': photo['z_mag_err'],
            'mcal_mag_err_z_2m': photo['z_mag_err'],

            # Fixed PSF parameters - all round with same size
            'mcal_psf_g1': zero,
            'mcal_psf_g2': zero,
            'mcal_psf_T_mean' : np.repeat(psf_T, nobj),

            # Everything that gets this far should be used, so flag=0
            'mcal_flags': np.zeros(nobj, dtype=np.int32),
            }

        return output



    def remove_undetected(self, data, photo, metacal):
        """
        Strip out any undetected objects from the two
        simulated data sets.

        Use a configuration parameter snr_limit to decide
        on the detection limit.
        """
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
            photo[f'{band}_mag'][not_detected_in_band] = np.inf

            # Record that we have detected this object at all
            detected |= detected_in_band


        # the protoDC2 sims have an edge with zero shear.
        # Remove it.
        zero_shear_edge = (abs(data['shear_1'])==0) & (abs(data['shear_2'])==0)
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


    def truncate_output(self, photo_file, metacal_file, end):
        """
        Cut down the output photometry file to its final 
        size.
        """
        group = photo_file['photometry']
        cols = list(group.keys())
        for col in cols:
            group[col].resize((end,))

        group = metacal_file['metacal']
        cols = list(group.keys())
        for col in cols:
            group[col].resize((end,))


def make_mock_photometry(n_visit, bands, data):
    """
    Generate a mock photometric table with noise added

    This is mostly from LSE‐40 by 
    Zeljko Ivezic, Lynne Jones, and Robert Lupton
    retrieved here:
    http://faculty.washington.edu/ivezic/Teaching/Astr511/LSST_SNRdoc.pdf
    """

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
        output[f'{band}_mag'] = mag_obs
        output[f'mag_err_{band}'] = mag_err
        output[f'{band}_mag_err'] = mag_err

        m = mag_resp*delta_gamma

        m1 = mag_obs + m
        m2 = mag_obs - m
        mag_obs_1p = m1
        mag_obs_1m = m2


        output[f'{band}_mag_1p'] = m1
        output[f'{band}_mag_1m'] = m2
        output[f'{band}_mag_2p'] = m1
        output[f'{band}_mag_2m'] = m2

        # Scale the SNR values according the to change in magnitude.r
        s = np.power(10., -0.4*m)

        snr1 = obs_snr * s
        snr2 = obs_snr / s
        output[f'snr_{band}_1p'] = snr1
        output[f'snr_{band}_1m'] = snr2
        output[f'snr_{band}_2p'] = snr1
        output[f'snr_{band}_2m'] = snr2



    return output



def generate_mock_metacal_mag_responses(bands, nobj):
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
        err_names = [f'{b}_mag_err' for b in self.bands]
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
                mag_obs = data[f'{band}_mag']
                snr     = data[f'snr_{band}_cModel']
                mag_err = data[f'{band}_mag_err']

                output[f'{band}_mag']    = mag_obs
                output[f'{band}_mag_err'] = mag_err
                output[f'snr_{band}']    = snr

                # Generate the metacalibrated values
                mag_obs_1p = mag_obs + mag_resp*delta_gamma
                mag_obs_1m = mag_obs - mag_resp*delta_gamma
                mag_obs_2p = mag_obs + mag_resp*delta_gamma
                mag_obs_2m = mag_obs - mag_resp*delta_gamma

                # Save them, but we will also need them to generate
                # metacalibrated SNRs below
                output[f'{band}_mag_1p'] = mag_obs_1p
                output[f'{band}_mag_1m'] = mag_obs_1m
                output[f'{band}_mag_2p'] = mag_obs_2p
                output[f'{band}_mag_2m'] = mag_obs_2m

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
