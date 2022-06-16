from .base_stage import PipelineStage
from .data_types import ShearCatalog, HDFFile
from .utils.calibration_tools import (
    band_variants,
    metacal_variants,
    metadetect_variants,
)
import numpy as np
from .utils.timer import Timer


class TXCosmoDC2Mock(PipelineStage):
    """
    Simulate mock shear and photometry measurements from CosmoDC2 (or similar)

    This stage simulates metacal data and metacalibrated
    photometry measurements, starting from a cosmology catalogs
    of the kind used as an input to DC2 image and obs-catalog simulations.

    This is mainly useful for testing infrastructure in advance
    of the DC2 catalogs being available, but might also be handy
    for starting from a purer simulation.
    """

    name = "TXCosmoDC2Mock"
    parallel = False
    inputs = [("response_model", HDFFile)]

    outputs = [
        ("shear_catalog", ShearCatalog),
        ("photometry_catalog", HDFFile),
    ]

    config_options = {
        "cat_name": "cosmoDC2",
        "visits_per_band": 165,  # used in the noise simulation
        "snr_limit": 4.0,  # used to decide what objects to cut out
        "max_size": 99999999999999,  # for testing on smaller catalogs
        "extra_cols": "",  # string-separated list of columns to include
        "max_npix": 99999999999999,
        "unit_response": False,
        "cat_size": 0,
        "flip_g2": True,  # this matches the metacal definition, and the treecorr/namaster one
        "apply_mag_cut": False,  # used when comparing to descqa measurements
        "Mag_r_limit": -19,  # used to decide what objects to cut out
        "metadetect": True,  # Alternatively we will mock a  metacal catalog
        "add_shape_noise": True, 
    }

    def data_iterator(self, gc):

        # Columns we need from the cosmo simulation
        cols = [
            "mag_true_u_lsst",
            "mag_true_g_lsst",
            "mag_true_r_lsst",
            "mag_true_i_lsst",
            "mag_true_z_lsst",
            "mag_true_y_lsst",
            "ra",
            "dec",
            "ellipticity_1_true",
            "ellipticity_2_true",
            "shear_1",
            "shear_2",
            "size_true",
            "galaxy_id",
            "redshift_true",
        ]
        # Add any extra requestd columns
        cols += self.config["extra_cols"].split()

        it = gc.get_quantities(cols, return_iterator=True)
        nfile = len(gc._file_list) if hasattr(gc, "_file_list") else 0

        for i, data in enumerate(it):
            if nfile:
                j = i + 1
                print(f"Loading chunk {j}/{nfile}")
            yield data

    def load_catalog(self):

        import GCRCatalogs

        cat_name = self.config["cat_name"]
        self.bands = ("u", "g", "r", "i", "z", "y")

        print(f"Loading from catalog {cat_name}")

        gc = GCRCatalogs.load_catalog(cat_name)

        # GCR sometimes tries to read the entire catalog
        # to measure its length rather than looking at metadata
        # this can take a very long time.
        # allow the user to say that already know it.
        N = self.config["cat_size"]
        if N == 0:
            N = len(gc)

        return gc, N

    def run(self):

        gc, N = self.load_catalog()

        print(f"Rank {self.rank} loaded: length = {N}.")

        target_size = min(N, self.config["max_size"])
        select_fraction = target_size / N

        if target_size != N:
            print(
                f"Will select approx {100*select_fraction:.2f}% of objects ({target_size})"
            )

        # Prepare output files
        metacal_file = self.open_output("shear_catalog", parallel=self.is_mpi())
        photo_file = self.open_output("photometry_catalog", parallel=self.is_mpi())
        photo_cols = self.setup_photometry_output(photo_file, target_size)
        if self.config["metadetect"]:
            metacal_cols = self.setup_metadetect_output(metacal_file, target_size)
        else:
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
            print(f"Process {self.rank} read chunk {count} - {count+chunk_size} of {N}")
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

            # Cut out any objects too faint to be detected and measured.
            # We have to do this after the photometry, so that we know if
            # the object is detected, but we can do it before making the mock
            # metacal info, saving us some time simulating un-needed objects
            if (
                self.config["snr_limit"] > 0
            ):  # otherwise there is no need to run this function which is slow
                self.remove_undetected(data, mock_photometry)

            if self.config["apply_mag_cut"]:
                self.apply_magnitude_cut(data)

            if self.config["metadetect"]:
                mock_shear = self.make_mock_metadetect(data, mock_photometry)
            else:
                mock_shear = self.make_mock_metacal(data, mock_photometry)

            # The chunk size has now changed
            some_col = list(mock_photometry.keys())[0]
            chunk_size = len(mock_photometry[some_col])

            # start is where this process should start writing this
            # chunk of data.  end is where the final process will finish
            # writing, becoming the starting point for the whole next
            # chunk over all the processes
            start, end = self.next_output_indices(start, chunk_size)

            # Save all output
            self.write_output(
                start,
                target_size,
                photo_cols,
                metacal_cols,
                photo_file,
                mock_photometry,
                metacal_file,
                mock_shear,
            )

            # The next iteration starts writing where the current one ends.
            start = end

            if start >= target_size:
                break
        # Tidy up

        photo_file.close()
        metacal_file.close()

        self.truncate_outputs(end)

    def truncate_outputs(self, n):
        import h5py

        if self.comm is not None:
            self.comm.Barrier()

        def visitor(name, node):
            if isinstance(node, h5py.Dataset):
                print(f"Resizing {name}")
                node.resize((n,))

        if self.rank == 0:
            # all files should now be closed for all procs
            print(f"Resizing all outupts to size {n}")

            with h5py.File(self.get_output("photometry_catalog"), "r+") as f:
                f["photometry"].visititems(visitor)

            with h5py.File(self.get_output("shear_catalog"), "r+") as f:
                f["shear"].visititems(visitor)

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
        # Get a list of all the column names
        cols = ["ra", "dec", "extendedness"]
        for band in self.bands:
            cols.append(f"mag_{band}")
            cols.append(f"mag_{band}_err")
            cols.append(f"snr_{band}")

        for col in self.config["extra_cols"].split():
            cols.append(col)

        # Make group for all the photometry
        group = photo_file.create_group("photometry")

        # Extensible columns becase we don't know the size yet.
        # We will cut down the size at the end.
        for col in cols:
            group.create_dataset(
                col, (target_size,), maxshape=(target_size,), dtype="f8"
            )

        # The only non-float column for now
        group.create_dataset("id", (target_size,), maxshape=(target_size,), dtype="i8")

        return cols + ["id"]

    def setup_metadetect_output(self, metacal_file, target_size):
        # Get a list of all the column names
        cols = metadetect_variants(
            "g1",
            "g2",
            "T",
            "s2n",
            "T_err",
            "ra",
            "dec",
            "psf_g1",
            "psf_g2",
            "mcal_psf_g1",
            "mcal_psf_g2",
            "mcal_psf_T_mean",
            "weight",
        ) + band_variants("riz", "mag", "mag_err", shear_catalog_type="metadetect")

        cols += ["true_g1", "true_g2", "redshift_true"]

        # Make group for all the photometry
        group = metacal_file.create_group("shear")

        # Extensible columns becase we don't know the size yet.
        # We will cut down the size at the end.
        for col in cols:
            group.create_dataset(
                col, (target_size,), maxshape=(target_size,), dtype="f8"
            )

        # Integer columns
        int_cols = metadetect_variants("id", "flags")
        for col in int_cols:
            group.create_dataset(
                col, (target_size,), maxshape=(target_size,), dtype="i8"
            )

        return cols + int_cols

    def setup_metacal_output(self, metacal_file, target_size):
        # Get a list of all the column names
        cols = (
            [
                "ra",
                "dec",
                "psf_g1",
                "psf_g2",
                "mcal_psf_g1",
                "mcal_psf_g2",
                "mcal_psf_T_mean",
            ]
            + metacal_variants("mcal_g1", "mcal_g2", "mcal_T", "mcal_s2n", "mcal_T_err")
            + band_variants(
                "riz", "mcal_mag", "mcal_mag_err", shear_catalog_type="metacal"
            )
            + ["weight"]
        )

        cols += ["true_g1", "true_g2", "redshift_true"]

        # Make group for all the photometry
        group = metacal_file.create_group("shear")

        # Extensible columns becase we don't know the size yet.
        # We will cut down the size at the end.

        for col in cols:
            group.create_dataset(
                col, (target_size,), maxshape=(target_size,), dtype="f8"
            )

        group.create_dataset("id", (target_size,), maxshape=(target_size,), dtype="i8")

        for col in metacal_variants("mcal_flags"):
            group.create_dataset(
                col, (target_size,), maxshape=(target_size,), dtype="i8"
            )

        return cols + ["id", "mcal_flags"]

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

        if self.config["unit_response"]:
            return

        model_file = self.open_input("response_model")
        snr_centers = model_file["R_model/log10_snr"][:]
        sz_centers = model_file["R_model/size"][:]
        R_mean = model_file["R_model/R_mean"][:]
        R_std = model_file["R_model/R_std"][:]
        model_file.close()

        # Save a 2D spline
        snr_grid, sz_grid = np.meshgrid(snr_centers, sz_centers)
        self.R_spline = scipy.interpolate.SmoothBivariateSpline(
            snr_grid.T.flatten(),
            sz_grid.T.flatten(),
            R_mean.flatten(),
            w=R_std.flatten(),
        )
        self.Rstd_spline = scipy.interpolate.SmoothBivariateSpline(
            snr_grid.T.flatten(), sz_grid.T.flatten(), R_std.flatten()
        )

    def write_output(
        self,
        start,
        target_size,
        photo_cols,
        metacal_cols,
        photo_file,
        photo_data,
        metacal_file,
        metacal_data,
    ):
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
        n = len(photo_data["id"])
        end = min(start + n, target_size)

        # assert photo_data['id'].min()>0

        # Save each column
        for name in photo_cols:
            photo_file[f"photometry/{name}"][start:end] = photo_data[name]

        for name in metacal_cols:
            metacal_file[f"shear/{name}"][start:end] = metacal_data[name]

    def make_mock_photometry(self, data):
        # The visit count affects the overall noise levels
        n_visit = self.config["visits_per_band"]
        # Do all the work in the function below
        photo = make_mock_photometry(
            n_visit, self.bands, data, self.config["unit_response"]
        )

        for col in self.config["extra_cols"].split():
            photo[col] = data[col]

        return photo

    def make_mock_metadetect(self, data, photo):
        """
        !!! This function is not working properly yet!!!
        Generate a mock metadetect table.
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
        snr = (photo["snr_r"] ** 2 + photo["snr_i"] ** 2 + photo["snr_z"] ** 2) ** 0.5
        snr_1p = (
            photo["snr_r_1p"] ** 2 + photo["snr_i_1p"] ** 2 + photo["snr_z_1p"] ** 2
        ) ** 0.5
        snr_1m = (
            photo["snr_r_1m"] ** 2 + photo["snr_i_1m"] ** 2 + photo["snr_z_1m"] ** 2
        ) ** 0.5
        snr_2p = (
            photo["snr_r_2p"] ** 2 + photo["snr_i_2p"] ** 2 + photo["snr_z_2p"] ** 2
        ) ** 0.5
        snr_2m = (
            photo["snr_r_2m"] ** 2 + photo["snr_i_2m"] ** 2 + photo["snr_z_2m"] ** 2
        ) ** 0.5

        if self.config["unit_response"]:
            assert np.allclose(snr, snr_1p)
            assert np.allclose(snr, snr_2m)

        nobj = snr.size

        log10_snr = np.log10(snr)

        # Convert from the half-light radius which is in the
        # input catalogs to a sigma.  Do this by pretending
        # that it is a Gaussian.  This is clearly wrong, and
        # if this causes major errors in the size cuts we may
        # have to modify this
        size_hlr = data["size_true"]
        size_sigma = size_hlr / np.sqrt(2 * np.log(2))
        size_T = 2 * size_sigma**2

        # Use a fixed PSF across all the objects
        psf_fwhm = 0.75
        psf_sigma = psf_fwhm / (2 * np.sqrt(2 * np.log(2)))
        psf_T = 2 * psf_sigma**2

        if self.config["unit_response"]:
            R = 1.0
            R_size = 0.0

        else:
            # Use the response model to get a reasonable response
            # value for this size and SNR
            R_mean = self.R_spline(log10_snr, size_T, grid=False)
            R_std = self.Rstd_spline(log10_snr, size_T, grid=False)

            # Assume a 0.2 correlation between the size response
            # and the shear response.
            rho = 0.2
            f = np.random.multivariate_normal(
                [0.0, 0.0], [[1.0, rho], [rho, 1.0]], nobj
            ).T
            R, R_size = f * R_std + R_mean

        # Convert magnitudes to fluxes according to the baseline
        # use in the metacal numbers
        flux_r = 10**0.4 * (27 - photo["mag_r"])
        flux_i = 10**0.4 * (27 - photo["mag_i"])
        flux_z = 10**0.4 * (27 - photo["mag_z"])

        # Note that this is delta_gamma not 2*delta_gamma, because
        # of how we use it below
        delta_gamma = 0.01

        # Use a fixed shape noise per component to generate
        # an overall
        if self.config["add_shape_noise"]:
            shape_noise = 0.26
        else:
            shape_noise = 0.
            
        eps = np.random.normal(0, shape_noise, nobj) + 1.0j * np.random.normal(
            0, shape_noise, nobj
        )
        # True shears without shape noise
        g1 = data["shear_1"]
        g2 = data["shear_2"]

        if self.config["flip_g2"]:
            g2 *= -1

        # Do the full combination of (g,epsilon) -> e, not the approx one
        g = g1 + 1j * g2
        e = (eps + g) / (1 + g.conj() * eps)
        e1 = e.real
        e2 = e.imag

        zero = np.zeros(nobj)
        # Now collect together everything to go into the metacal
        # file
        output = {
            # Basic values
            "00/id": photo["id"],
            "1p/id": photo["id"],
            "1m/id": photo["id"],
            "2p/id": photo["id"],
            "2m/id": photo["id"],
            "00/ra": photo["ra"],
            "1p/ra": photo["ra"],
            "1m/ra": photo["ra"],
            "2p/ra": photo["ra"],
            "2m/ra": photo["ra"],
            "00/dec": photo["dec"],
            "1p/dec": photo["dec"],
            "1m/dec": photo["dec"],
            "2p/dec": photo["dec"],
            "2m/dec": photo["dec"],
            # Keep the truth values for use in some testing paths
            # We are not pretending to do meta
            "true_g1": g1,
            "true_g2": g2,
            "redshift_true": photo["redshift_true"],
            # g1
            "00/g1": e1 * R,
            "1p/g1": (e1 + delta_gamma) * R,
            "1m/g1": (e1 - delta_gamma) * R,
            "2p/g1": e1 * R,
            "2m/g1": e1 * R,
            # g2
            "00/g2": e2 * R,
            "1p/g2": e2 * R,
            "1m/g2": e2 * R,
            "2p/g2": (e2 + delta_gamma) * R,
            "2m/g2": (e2 - delta_gamma) * R,
            # T
            "00/T": size_T,
            "1p/T": size_T + R_size * delta_gamma,
            "1m/T": size_T - R_size * delta_gamma,
            "2p/T": size_T + R_size * delta_gamma,
            "2m/T": size_T - R_size * delta_gamma,
            # Terr
            "00/T_err": zero,
            "1p/T_err": zero,
            "1m/T_err": zero,
            "2p/T_err": zero,
            "2m/T_err": zero,
            # size
            "00/s2n": snr,
            "1p/s2n": snr_1p,
            "1m/s2n": snr_1m,
            "2p/s2n": snr_2p,
            "2m/s2n": snr_2m,
            # Magntiudes and fluxes, just copied from the inputs.
            "00/mag_r": photo["mag_r"],
            "00/mag_i": photo["mag_i"],
            "00/mag_z": photo["mag_z"],
            "00/mag_err_r": photo["mag_r_err"],
            "00/mag_err_i": photo["mag_i_err"],
            "00/mag_err_z": photo["mag_z_err"],
            "1p/mag_r": photo["mag_r_1p"],
            "2p/mag_r": photo["mag_r_2p"],
            "1m/mag_r": photo["mag_r_1m"],
            "2m/mag_r": photo["mag_r_2m"],
            "1p/mag_i": photo["mag_i_1p"],
            "2p/mag_i": photo["mag_i_2p"],
            "1m/mag_i": photo["mag_i_1m"],
            "2m/mag_i": photo["mag_i_2m"],
            "1p/mag_z": photo["mag_z_1p"],
            "2p/mag_z": photo["mag_z_2p"],
            "1m/mag_z": photo["mag_z_1m"],
            "2m/mag_z": photo["mag_z_2m"],
            "1p/mag_err_r": photo["mag_r_err"],
            "2p/mag_err_r": photo["mag_r_err"],
            "1m/mag_err_r": photo["mag_r_err"],
            "2m/mag_err_r": photo["mag_r_err"],
            "1p/mag_err_i": photo["mag_i_err"],
            "2p/mag_err_i": photo["mag_i_err"],
            "1m/mag_err_i": photo["mag_i_err"],
            "2m/mag_err_i": photo["mag_i_err"],
            "1p/mag_err_z": photo["mag_z_err"],
            "2p/mag_err_z": photo["mag_z_err"],
            "1m/mag_err_z": photo["mag_z_err"],
            "2m/mag_err_z": photo["mag_z_err"],
            # Fixed PSF parameters - all round with same size
            "00/mcal_psf_g1": zero,
            "1p/mcal_psf_g1": zero,
            "1m/mcal_psf_g1": zero,
            "2p/mcal_psf_g1": zero,
            "2m/mcal_psf_g1": zero,
            "00/mcal_psf_g2": zero,
            "1p/mcal_psf_g2": zero,
            "1m/mcal_psf_g2": zero,
            "2p/mcal_psf_g2": zero,
            "2m/mcal_psf_g2": zero,
            "00/psf_g1": zero,
            "1p/psf_g1": zero,
            "1m/psf_g1": zero,
            "2p/psf_g1": zero,
            "2m/psf_g1": zero,
            "00/psf_g2": zero,
            "1p/psf_g2": zero,
            "1m/psf_g2": zero,
            "2p/psf_g2": zero,
            "2m/psf_g2": zero,
            "00/mcal_psf_T_mean": np.repeat(psf_T, nobj),
            "1p/mcal_psf_T_mean": np.repeat(psf_T, nobj),
            "1m/mcal_psf_T_mean": np.repeat(psf_T, nobj),
            "2p/mcal_psf_T_mean": np.repeat(psf_T, nobj),
            "2m/mcal_psf_T_mean": np.repeat(psf_T, nobj),
            # Everything that gets this far should be used, so flag=0
            "00/flags": np.zeros(nobj, dtype=np.int32),
            "1p/flags": np.zeros(nobj, dtype=np.int32),
            "1m/flags": np.zeros(nobj, dtype=np.int32),
            "2p/flags": np.zeros(nobj, dtype=np.int32),
            "2m/flags": np.zeros(nobj, dtype=np.int32),
            # we use weights of one for everything for metacal
            # if that ever changes we may also need to add
            # weight_1p, etc.
            "00/weight": np.ones(nobj),
            "1p/weight": np.ones(nobj),
            "1m/weight": np.ones(nobj),
            "2p/weight": np.ones(nobj),
            "2m/weight": np.ones(nobj),
        }

        return output

    def make_mock_metacal(self, data, photo):
        """
        I copied the old version here to use if we do not want metadetect.
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
        snr = (photo["snr_r"] ** 2 + photo["snr_i"] ** 2 + photo["snr_z"] ** 2) ** 0.5
        snr_1p = (
            photo["snr_r_1p"] ** 2 + photo["snr_i_1p"] ** 2 + photo["snr_z_1p"] ** 2
        ) ** 0.5
        snr_1m = (
            photo["snr_r_1m"] ** 2 + photo["snr_i_1m"] ** 2 + photo["snr_z_1m"] ** 2
        ) ** 0.5
        snr_2p = (
            photo["snr_r_2p"] ** 2 + photo["snr_i_2p"] ** 2 + photo["snr_z_2p"] ** 2
        ) ** 0.5
        snr_2m = (
            photo["snr_r_2m"] ** 2 + photo["snr_i_2m"] ** 2 + photo["snr_z_2m"] ** 2
        ) ** 0.5

        if self.config["unit_response"]:
            assert np.allclose(snr, snr_1p)
            assert np.allclose(snr, snr_2m)

        nobj = snr.size

        log10_snr = np.log10(snr)

        # Convert from the half-light radius which is in the
        # input catalogs to a sigma.  Do this by pretending
        # that it is a Gaussian.  This is clearly wrong, and
        # if this causes major errors in the size cuts we may
        # have to modify this
        size_hlr = data["size_true"]
        size_sigma = size_hlr / np.sqrt(2 * np.log(2))
        size_T = 2 * size_sigma**2

        # Use a fixed PSF across all the objects
        psf_fwhm = 0.75
        psf_sigma = psf_fwhm / (2 * np.sqrt(2 * np.log(2)))
        psf_T = 2 * psf_sigma**2

        if self.config["unit_response"]:
            R = 1.0
            R_size = 0.0

        else:
            # Use the response model to get a reasonable response
            # value for this size and SNR
            R_mean = self.R_spline(log10_snr, size_T, grid=False)
            R_std = self.Rstd_spline(log10_snr, size_T, grid=False)

            # Assume a 0.2 correlation between the size response
            # and the shear response.
            rho = 0.2
            f = np.random.multivariate_normal(
                [0.0, 0.0], [[1.0, rho], [rho, 1.0]], nobj
            ).T
            R, R_size = f * R_std + R_mean

        # Convert magnitudes to fluxes according to the baseline
        # use in the metacal numbers
        flux_r = 10**0.4 * (27 - photo["mag_r"])
        flux_i = 10**0.4 * (27 - photo["mag_i"])
        flux_z = 10**0.4 * (27 - photo["mag_z"])

        # Note that this is delta_gamma not 2*delta_gamma, because
        # of how we use it below
        delta_gamma = 0.01

        # Use a fixed shape noise per component to generate
        # an overall
        if self.config["add_shape_noise"]:
            shape_noise = 0.26
        else:
            shape_noise = 0.
            
        eps = np.random.normal(0, shape_noise, nobj) + 1.0j * np.random.normal(
            0, shape_noise, nobj
        )
        # True shears without shape noise
        g1 = data["shear_1"]
        g2 = data["shear_2"]

        if self.config["flip_g2"]:
            g2 *= -1

        # Do the full combination of (g,epsilon) -> e, not the approx one
        g = g1 + 1j * g2
        e = (eps + g) / (1 + g.conj() * eps)
        e1 = e.real
        e2 = e.imag

        zero = np.zeros(nobj)
        # Now collect together everything to go into the metacal
        # file
        output = {
            # Basic values
            "id": photo["id"],
            "ra": photo["ra"],
            "dec": photo["dec"],
            # Keep the truth value just in case
            "true_g1": g1,
            "true_g2": g2,
            # add true redshift since it is used in source selector
            "redshift_true": photo["redshift_true"],
            # g1
            "mcal_g1": e1 * R,
            "mcal_g1_1p": (e1 + delta_gamma) * R,
            "mcal_g1_1m": (e1 - delta_gamma) * R,
            "mcal_g1_2p": e1 * R,
            "mcal_g1_2m": e1 * R,
            # g2
            "mcal_g2": e2 * R,
            "mcal_g2_1p": e2 * R,
            "mcal_g2_1m": e2 * R,
            "mcal_g2_2p": (e2 + delta_gamma) * R,
            "mcal_g2_2m": (e2 - delta_gamma) * R,
            # T
            "mcal_T": size_T,
            "mcal_T_1p": size_T + R_size * delta_gamma,
            "mcal_T_1m": size_T - R_size * delta_gamma,
            "mcal_T_2p": size_T + R_size * delta_gamma,
            "mcal_T_2m": size_T - R_size * delta_gamma,
            # Terr
            "mcal_T_err": zero,
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
            "mcal_mag_r": photo["mag_r"],
            "mcal_mag_i": photo["mag_i"],
            "mcal_mag_z": photo["mag_z"],
            "mcal_mag_err_r": photo["mag_r_err"],
            "mcal_mag_err_i": photo["mag_i_err"],
            "mcal_mag_err_z": photo["mag_z_err"],
            "mcal_mag_r_1p": photo["mag_r_1p"],
            "mcal_mag_r_2p": photo["mag_r_2p"],
            "mcal_mag_r_1m": photo["mag_r_1m"],
            "mcal_mag_r_2m": photo["mag_r_2m"],
            "mcal_mag_i_1p": photo["mag_i_1p"],
            "mcal_mag_i_2p": photo["mag_i_2p"],
            "mcal_mag_i_1m": photo["mag_i_1m"],
            "mcal_mag_i_2m": photo["mag_i_2m"],
            "mcal_mag_z_1p": photo["mag_z_1p"],
            "mcal_mag_z_2p": photo["mag_z_2p"],
            "mcal_mag_z_1m": photo["mag_z_1m"],
            "mcal_mag_z_2m": photo["mag_z_2m"],
            "mcal_mag_err_r_1p": photo["mag_r_err"],
            "mcal_mag_err_r_2p": photo["mag_r_err"],
            "mcal_mag_err_r_1m": photo["mag_r_err"],
            "mcal_mag_err_r_2m": photo["mag_r_err"],
            "mcal_mag_err_i_1p": photo["mag_i_err"],
            "mcal_mag_err_i_2p": photo["mag_i_err"],
            "mcal_mag_err_i_1m": photo["mag_i_err"],
            "mcal_mag_err_i_2m": photo["mag_i_err"],
            "mcal_mag_err_z_1p": photo["mag_z_err"],
            "mcal_mag_err_z_2p": photo["mag_z_err"],
            "mcal_mag_err_z_1m": photo["mag_z_err"],
            "mcal_mag_err_z_2m": photo["mag_z_err"],
            # Fixed PSF parameters - all round with same size
            "mcal_psf_g1": zero,
            "mcal_psf_g2": zero,
            "psf_g1": zero,
            "psf_g2": zero,
            "mcal_psf_T_mean": np.repeat(psf_T, nobj),
            # Everything that gets this far should be used, so flag=0
            "mcal_flags": np.zeros(nobj, dtype=np.int32),
            "mcal_flags_1p": np.zeros(nobj, dtype=np.int32),
            "mcal_flags_1m": np.zeros(nobj, dtype=np.int32),
            "mcal_flags_2p": np.zeros(nobj, dtype=np.int32),
            "mcal_flags_2m": np.zeros(nobj, dtype=np.int32),
            # we use weights of one for everything for metacal
            # if that ever changes we may also need to add
            # weight_1p, etc.
            "weight": np.ones(nobj),
        }

        return output

    def apply_magnitude_cut(self, data):
        """
        Allow for a cut in absolute magnitude.
        """
        mag_limit = self.config["Mag_r_limit"]
        sel = data["Mag_true_r_sdss_z0"] < mag_limit

        ndet = sel.sum()
        ntot = sel.size
        fract = ndet * 100.0 / ntot
        print(f"{ndet} objects pass magnitude cut out of {ntot} objects ({fract:.1f}%)")

        # Remove all objects not selected
        for key in list(data.keys()):
            data[key] = data[key][sel]

    def remove_undetected(self, data, photo):
        """
        Strip out any undetected objects from the two
        simulated data sets.

        Use a configuration parameter snr_limit to decide
        on the detection limit.
        """
        snr_limit = self.config["snr_limit"]

        # This will become a boolean array in a minute when
        # we OR it with an array
        detected = False

        # Check if detected in any band.  Makes a boolean array
        # Even though we started with just a single False.
        for band in self.bands:
            detected_in_band = photo[f"snr_{band}"] > snr_limit
            not_detected_in_band = ~detected_in_band
            # Set objects not detected in one band that are detected in another
            # to inf magnitude in that band, and the SNR to zero.
            # We have to do this for each of the variants also, because otherwise
            # we end up with wildly different final SNR values later.
            # This is the metadetection issue really!
            for v in metacal_variants(f"snr_{band}"):
                photo[v][not_detected_in_band] = 0.0
            for v in metacal_variants(f"mag_{band}"):
                photo[v][not_detected_in_band] = np.inf

            # Record that we have detected this object at all
            detected |= detected_in_band

        # the protoDC2 sims have an edge with zero shear.
        # Remove it.
        zero_shear_edge = (abs(data["shear_1"]) == 0) & (abs(data["shear_2"]) == 0)
        print(
            "Removing {} objects with identically zero shear in both terms".format(
                zero_shear_edge.sum()
            )
        )

        detected &= ~zero_shear_edge

        # Print out interesting information
        ndet = detected.sum()
        ntot = detected.size
        fract = ndet * 100.0 / ntot
        print(f"Detected {ndet} out of {ntot} objects ({fract:.1f}%)")

        # Remove all objects not detected in *any* band
        # make a copy of the keys with list(photo.keys()) so we are not
        # modifying during the iteration
        for key in list(photo.keys()):
            photo[key] = photo[key][detected]

        for key in list(data.keys()):
            data[key] = data[key][detected]


class TXBuzzardMock(TXCosmoDC2Mock):
    """
    Simulate mock photometry from Buzzard.

    May be obsolete.
    """

    name = "TXBuzzardMock"
    parallel = False
    inputs = [("response_model", HDFFile)]

    outputs = [
        ("shear_catalog", ShearCatalog),
        ("photometry_catalog", HDFFile),
    ]

    config_options = {
        "cat_name": "buzzard",
        "visits_per_band": 165,  # used in the noise simulation
        "snr_limit": 4.0,  # used to decide what objects to cut out
        "max_size": 99999999999999,  # for testing on smaller catalogs
        "extra_cols": "",  # string-separated list of columns to include
        "max_npix": 99999999999999,
        "unit_response": False,
        "flip_g2": True,  # this matches the metacal definition, and the treecorr/namaster one
    }


class TXGaussianSimsMock(TXCosmoDC2Mock):
    """
    Simulate mock photometry from gaussian simulations

    This stage simulates metacal data and metacalibrated
    photometry measurements, starting from simple Gaussian simulations
    produced starting from CCL power spectra and poission sampling galaxies
    from it.

    This is mainly useful for testing infrastructure
    starting from a purer simulation.
    """

    name = "TXGaussianSimsMock"
    parallel = False
    inputs = [("response_model", HDFFile)]

    outputs = [
        ("shear_catalog", ShearCatalog),
        ("photometry_catalog", HDFFile),
    ]

    config_options = {
        "cat_name": "GaussianSims",
        "visits_per_band": 165,  # used in the noise simulation
        "snr_limit": 0.0,  # we want to keep all input objects here
        "max_size": 99999999999999,  # for testing on smaller catalogs
        "extra_cols": "",  # string-separated list of columns to include
        "max_npix": 99999999999999,
        "unit_response": True,
        "cat_size": 0,
        "flip_g2": False,  # this matches the metacal definition, and the treecorr/namaster one
        "apply_mag_cut": False,  # used when comparing to descqa measurements
        "metadetect": True,  # Alternatively we will mock a  metacal catalog
        "add_shape_noise": False, # the input cats already have shape noise included
    }

    def data_iterator(self, cat):

        # all cols we need
        (
            ra,
            dec,
            g1,
            g2,
            z,
            m_u,
            m_g,
            m_r,
            m_i,
            m_z,
            m_y,
            etrue1,
            etrue2,
            size,
            galaxy_id,
        ) = cat

        # figuring out number of chunks
        chunk_size = 1_000_000
        nchunk = len(ra) // chunk_size
        if nchunk * chunk_size < len(ra):
            nchunk += 1

        # main loop
        for i in range(nchunk):
            # start and end index of this chunk in full data
            s = chunk_size * i
            e = s + chunk_size

            # make dict just for this chunk of data
            data_chunk = {}
            data_chunk["ra"] = ra[s:e]
            data_chunk["dec"] = dec[s:e]
            data_chunk["shear_1"] = g1[s:e]
            data_chunk["shear_2"] = g2[s:e]
            data_chunk["size_true"] = size[s:e]
            data_chunk["galaxy_id"] = galaxy_id[s:e]
            data_chunk["redshift_true"] = z[s:e]
            data_chunk["ellipticity_1_true"] = etrue1[s:e]
            data_chunk["ellipticity_2_true"] = etrue2[s:e]
            data_chunk["mag_true_u_lsst"] = m_u[s:e]
            data_chunk["mag_true_g_lsst"] = m_g[s:e]
            data_chunk["mag_true_r_lsst"] = m_r[s:e]
            data_chunk["mag_true_i_lsst"] = m_i[s:e]
            data_chunk["mag_true_z_lsst"] = m_z[s:e]
            data_chunk["mag_true_y_lsst"] = m_y[s:e]

            # send this chunk of data back to caller
            yield data_chunk

    def load_catalog(self):
        """
        This method loads the Gaussian sims produced in this notebook
        https://github.com/LSSTDESC/txpipe-cosmodc2/blob/master/notebooks/MaskAreaGaussianSims_and_FakeTXPipeInput.ipynb
        """
        cat_name = self.config["cat_name"]
        self.bands = ("u", "g", "r", "i", "z", "y")

        print(f"Loading from catalog {cat_name}")

        cat = np.load(cat_name, allow_pickle = True)
    
        N = len(cat[0])

        return cat, N


def make_mock_photometry(n_visit, bands, data, unit_response):
    """
    Generate a mock photometric table with noise added

    This is mostly from LSE‐40 by
    Zeljko Ivezic, Lynne Jones, and Robert Lupton
    retrieved here:
    http://faculty.washington.edu/ivezic/Teaching/Astr511/LSST_SNRdoc.pdf
    """

    output = {}
    nobj = data["galaxy_id"].size
    output["ra"] = data["ra"]
    output["dec"] = data["dec"]
    output["id"] = data["galaxy_id"]
    output["extendedness"] = np.ones(nobj)

    # Sky background, seeing FWHM, and system throughput,
    # all from table 2 of Ivezic, Jones, & Lupton
    B_b = np.array([85.07, 467.9, 1085.2, 1800.3, 2775.7, 3614.3])
    fwhm = np.array([0.77, 0.73, 0.70, 0.67, 0.65, 0.63])
    T_b = np.array([0.0379, 0.1493, 0.1386, 0.1198, 0.0838, 0.0413])

    # effective pixels size for a Gaussian PSF, from equation
    # 27 of Ivezic, Jones, & Lupton
    pixel = 0.2  # arcsec
    N_eff = 2.436 * (fwhm / pixel) ** 2

    # other numbers from Ivezic, Jones, & Lupton
    sigma_inst2 = 10.0**2  # instrumental noise in photons per pixel, just below eq 42
    gain = 1  # ADU units per photon, also just below eq 42
    D = 6.5  # primary mirror diameter in meters, from LSST key numbers page (effective clear diameter)
    # Not sure what effective clear diameter means but it's closer to the number used in the paper
    time = 30.0  # seconds per exposure, from LSST key numbers page
    sigma_b2 = 0.0  # error on background, just above eq 42

    # combination of these  used below, from various equations
    factor = 5455.0 / gain * (D / 6.5) ** 2 * (time / 30.0)

    # Fake some metacal responses
    if unit_response:
        mag_responses = [0.0 for i in bands]
    else:
        mag_responses = generate_mock_metacal_mag_responses(bands, nobj)

    delta_gamma = 0.01  # this is the half-delta gamma, i.e. gamma_+ - gamma_0
    # that's the right thing to use here because we are doing m+ = m0 + dm/dy*dy
    # Use the same response for gamma1 and gamma2

    for band, b_b, t_b, n_eff, mag_resp in zip(bands, B_b, T_b, N_eff, mag_responses):
        # truth magnitude
        mag = data[f"mag_true_{band}_lsst"]
        output[f"mag_true_{band}_lsst"] = mag

        # expected signal photons, over all visits
        c_b = factor * 10 ** (0.4 * (25 - mag)) * t_b * n_visit

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
        mag_obs = 25 - 2.5 * np.log10(n_obs / factor / t_b / n_visit)

        # converting error on n_obs to error on mag
        mag_err = 2.5 / np.log(10.0) / obs_snr

        output[f"true_snr_{band}"] = true_snr
        output[f"snr_{band}"] = obs_snr
        output[f"mag_{band}"] = mag_obs
        output[f"mag_err_{band}"] = mag_err
        output[f"mag_{band}_err"] = mag_err

        m = mag_resp * delta_gamma

        m1 = mag_obs + m
        m2 = mag_obs - m
        mag_obs_1p = m1
        mag_obs_1m = m2

        output[f"mag_{band}_1p"] = m1
        output[f"mag_{band}_1m"] = m2
        output[f"mag_{band}_2p"] = m1
        output[f"mag_{band}_2m"] = m2

        # Scale the SNR values according the to change in magnitude.r
        s = np.power(10.0, -0.4 * m)

        snr1 = obs_snr * s
        snr2 = obs_snr / s
        output[f"snr_{band}_1p"] = snr1
        output[f"snr_{band}_1m"] = snr2
        output[f"snr_{band}_2p"] = snr1
        output[f"snr_{band}_2m"] = snr2

    return output


def generate_mock_metacal_mag_responses(bands, nobj):
    nband = len(bands)
    mu = np.zeros(nband)  # seems approx mean of response across bands, from HSC tract
    rho = 0.25  #  approx correlation between response in bands, from HSC tract
    sigma2 = 1.7**2  # approx variance of response, from HSC tract
    covmat = np.full((nband, nband), rho * sigma2)
    np.fill_diagonal(covmat, sigma2)
    mag_responses = np.random.multivariate_normal(mu, covmat, nobj).T
    return mag_responses


def test():
    import pylab

    data = {
        "ra": None,
        "dec": None,
        "galaxy_id": None,
    }
    bands = "ugrizy"
    n_visit = 165
    M5 = [24.22, 25.17, 24.74, 24.38, 23.80]
    for b, m5 in zip(bands, M5):
        data[f"mag_true_{b}_lsst"] = np.repeat(m5, 10000)
    results = make_mock_photometry(n_visit, bands, data, True)
    pylab.hist(results["snr_r"], bins=50, histtype="step")
    pylab.savefig("snr_r.png")
