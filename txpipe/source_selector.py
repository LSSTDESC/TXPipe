from .base_stage import PipelineStage
from .data_types import (
    ShearCatalog,
    YamlFile,
    PhotozPDFFile,
    TomographyCatalog,
    HDFFile,
    TextFile,
)
from .utils import SourceNumberDensityStats, rename_iterated
from .utils.calibration_tools import read_shear_catalog_type, apply_metacal_response
from .utils.calibration_tools import (
    metacal_variants,
    metadetect_variants,
    band_variants,
    MetacalCalculator,
    LensfitCalculator,
    HSCCalculator,
    MetaDetectCalculator,
)
from .utils.calibrators import (
    MetaCalibrator,
    MetaDetectCalibrator,
    LensfitCalibrator,
    HSCCalibrator,
)
from .binning import build_tomographic_classifier, apply_classifier
import numpy as np
import warnings

class BinStats:
    """
    This is a small helper class to store and write the statistics of a
    single tomographic bin. It helps simplify some of the code below.
    """

    def __init__(self, source_count, N_eff, mean_e, sigma_e, calibrator):
        """
        Parameters
        ----------
        source_count: int
            The raw number of objects
        N_eff: int
            The effective number of objects
        mean_e: array or list
            Length 2. The mean ellipticity e1 and e2 in the bin
        sigma_e: float
            The ellipticity dispersion
        calibrator: Calibrator
            A Calibrator subclass instance that calibrates this bin
        """
        self.source_count = source_count
        self.N_eff = N_eff
        self.mean_e = mean_e
        self.sigma_e = sigma_e
        self.calibrator = calibrator

    def write_to(self, outfile, i):
        """
        Writes the bin statistics to an HDF5 file in the right place

        Parameters
        ----------
        outfile: an open HDF5 file object
            The output file
        i: int or str
            The index for this tomographic bin, or "2d"
        """
        group = outfile["tomography"]
        if i == "2d":
            group["counts_2d"][:] = self.source_count
            group["N_eff_2d"][:] = self.N_eff
            # This might get saved by the calibrator also
            # but in case not we do it here.
            group["mean_e1_2d"][:] = self.mean_e[0]
            group["mean_e2_2d"][:] = self.mean_e[1]
            group["sigma_e_2d"][:] = self.sigma_e
        else:
            group["counts"][i] = self.source_count
            group["N_eff"][i]   = self.N_eff
            group["mean_e1"][i] = self.mean_e[0]
            group["mean_e2"][i] = self.mean_e[1]
            group["sigma_e"][i] = self.sigma_e

        self.calibrator.save(outfile, i)


class TXSourceSelectorBase(PipelineStage):
    """
    Base stage for source selection using S/N, size, and flag cuts and tomography

    The subclasses of this pipeline stage select
    objects to be used as the source sample for
    the shear-shear and shear-position calibrations.

    Each subclass is specialised to a specific kind
    of catalog and calibration scheme. You cannot
    use this parent class.

    All the versions apply some general cuts based
    on object flags, and size and S/N cuts
    based on the configuration file.

    It also splits those objects into tomographic
    bins using either a random forest algorithm, with
    the training data in calibration_table, or according to
    the true shear (if present), according to the choice the
    user makes in the configuration.

    Once these selections are made it constructs
    the quantities needed to calibrate each bin,
    generating a set of Calibrator objects.
    """

    name = "TXSourceSelector"

    inputs = [
        ("shear_catalog", ShearCatalog),
        ("calibration_table", TextFile),
    ]

    outputs = [("shear_tomography_catalog", TomographyCatalog)]

    config_options = {
        "input_pz": False,
        "true_z": False,
        "bands": "riz",  # bands from metacal to use
        "verbose": False,
        "T_cut": float,
        "s2n_cut": float,
        "chunk_rows": 10000,
        "source_zbin_edges": [float],
        "random_seed": 42,
    }

    def run(self):
        import astropy.table
        import sklearn.ensemble

        # This base class should no longer be used, so to avoid people
        # accidentally doing so we give a clear message if they try.
        if self.name == "TXSourceSelector":
            raise ValueError(
                "Do not use the class TXSourceSelector any more. "
                "Use one of the subclasses like TXSourceSelectorMetacal"
            )

        # Suppress some warnings from numpy that are not relevant
        original_warning_settings = np.seterr(all="ignore")

        # Are we using a metacal or lensfit catalog?
        shear_catalog_type = read_shear_catalog_type(self)
        bands = self.config["bands"]

        # The output file we will put the tomographic
        # information into
        output_file = self.setup_output()

        # The iterator that will loop through the data.
        # Set it up here so that we can find out if there are any
        # problems with it before we get run the slow classifier.
        it = self.data_iterator()

        # Build a classifier used to put objects into tomographic bins
        if not (self.config["input_pz"] or self.config["true_z"]):
            classifier, features = build_tomographic_classifier(
                bands,
                self.get_input("calibration_table"),
                self.config["source_zbin_edges"],
                self.config["random_seed"],
                self.comm,
            )

        # We will collect the selection biases for each bin
        # as a matrix.  We will collect together the different
        # matrices for each chunk and do a weighted average at the end.
        nbin_source = len(self.config["source_zbin_edges"]) - 1

        number_density_stats = SourceNumberDensityStats(
            nbin_source, comm=self.comm, shear_type=shear_catalog_type
        )

        calculators = self.setup_response_calculators(nbin_source)

        # Loop through the input data, processing it chunk by chunk
        for (start, end, shear_data) in it:
            print(f"Process {self.rank} running selection for rows {start:,}-{end:,}")

            # Either apply a simple z cut if we have an input PZ estimate or
            # the truth value (in simulations)
            if self.config["true_z"] or self.config["input_pz"]:
                pz_data = self.apply_simple_redshift_cut(shear_data)

            else:
                # or select most likely tomographic source bin, with a random forest
                pz_data = apply_classifier(
                    classifier,
                    features,
                    bands,
                    shear_catalog_type,
                    shear_data,
                )
            # Combine this selection with size and snr cuts to produce a source selection
            # and calculate the shear bias it would generate
            tomo_bin, R, counts = self.calculate_tomography(
                pz_data, shear_data, calculators
            )

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

    def apply_simple_redshift_cut(self, shear_data):
        pz_data = {}
        if self.config["input_pz"]:
            zz = shear_data["mean_z"]
        else:
            zz = shear_data["redshift_true"]

        pz_data_bin = np.zeros(len(zz), dtype=int) - 1
        for zi in range(len(self.config["source_zbin_edges"]) - 1):
            mask_zbin = (zz >= self.config["source_zbin_edges"][zi]) & (
                zz < self.config["source_zbin_edges"][zi + 1]
            )
            pz_data_bin[mask_zbin] = zi

        return {"zbin": pz_data_bin}

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
        nbin = len(self.config["source_zbin_edges"]) - 1
        n = len(list(shear_data.values())[0])

        # The main output data - the tomographic
        # bin index for each object, or -1 for no bin.
        tomo_bin = np.repeat(-1, n)

        # We also keep count of total count of objects in each bin,
        # and also the overall count for all the bins (the last entry)
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

        return tomo_bin[tomo_bin>=0], R[tomo_bin>=0], counts

    def compute_per_object_response(self, data):
        # The default implementation has no per-object response
        # Some subclasses supply it.
        return None

    def setup_output(self):
        """
        Set up the output data file.

        Creates the data sets and groups to put module output
        in the shear_tomography_catalog output file.
        """
        cat_type = read_shear_catalog_type(self)
        with self.open_input("shear_catalog", wrapper=True) as f:
            n = f.get_size()

        zbins = self.config["source_zbin_edges"]
        nbin_source = len(zbins) - 1

        output = self.open_output("shear_tomography_catalog", parallel=True, wrapper=True)
        outfile = output.file
        group = outfile.create_group("tomography")
        group.attrs['catalog_type'] = cat_type
        output.write_zbins(zbins)
        group.create_dataset("bin", (n,), dtype="i")
        group.create_dataset("counts", (nbin_source,), dtype="i")
        group.create_dataset("counts_2d", (1,), dtype="i")
        group.create_dataset("sigma_e", (nbin_source,), dtype="f")
        group.create_dataset("sigma_e_2d", (1,), dtype="f")
        group.create_dataset("mean_e1", (nbin_source,), dtype="f")
        group.create_dataset("mean_e2", (nbin_source,), dtype="f")
        group.create_dataset("mean_e1_2d", (1,), dtype="f")
        group.create_dataset("mean_e2_2d", (1,), dtype="f")
        group.create_dataset("N_eff", (nbin_source,), dtype="f")
        group.create_dataset("N_eff_2d", (1,), dtype="f")

        group.attrs["nbin"] = nbin_source

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
        group = outfile["tomography"]
        group["bin"][start:end] = source_bin

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
                stats.write_to(outfile, i if i < nbin_source else "2d")

    def select(self, data, bin_index):
        """
        Select which objects are to be chosen in this tomographic bin.
        We do this by calling out to the 2D selector, which does the
        cuts on size and SNR, and then combining this with a cut on tomographic bin.

        Note that we don't call this method directly in this class. Instead
        we pass it to the Calculator objects that call it, sometimes on different
        columns of data.
        """
        zbin = data["zbin"]
        verbose = self.config["verbose"]

        sel = self.select_2d(data, calling_from_select=True)
        sel &= zbin == bin_index
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
        s2n_cut = self.config["s2n_cut"]
        T_cut = self.config["T_cut"]
        verbose = self.config["verbose"]
        variant = data.suffix

        shear_prefix = self.config["shear_prefix"]
        s2n  = data[f"{shear_prefix}s2n{variant}"]
        T    = data[f"{shear_prefix}T{variant}"]
        Tpsf = data[f"{shear_prefix}psf_T_mean"]
        flag = data[f"{shear_prefix}flags{variant}"]

        # Apply our cuts.  We keep track of the number of objects
        # reject by each cut in case it's important.
        # First we require flag = 0
        n0 = len(flag)
        sel = flag == 0
        f1 = sel.sum() / n0

        # Next we required a minimum object size compared to the PSF
        sel &= (T / Tpsf) > T_cut
        f2 = sel.sum() / n0

        # Then we require a signal-to-noise minimum
        sel &= s2n > s2n_cut
        f3 = sel.sum() / n0

        # Finally we want objects that have been put into any of our other
        # tomographic bins
        sel &= data["zbin"] >= 0
        f4 = sel.sum() / n0

        # Print out a message.  If we are selecting a 2D sample
        # this is the complete message.  Otherwise if we are about
        # to also apply a redshift bin cut about then the message will continue
        # as above
        if verbose and calling_from_select:
            print(
                f"Tomo selection ({variant}) {f1:.2%} flag, {f2:.2%} size, "
                f"{f3:.2%} SNR, ",
                end="",
            )
        elif verbose:
            print(
                f"2D selection ({variant}) {f1:.2%} flag, {f2:.2%} size, "
                f"{f3:.2%} SNR, {f4:.2%} any z bin"
            )
            print("total 2D", sel.sum())
        return sel


class TXSourceSelectorMetacal(TXSourceSelectorBase):
    """
    Source selection and tomography for metacal catalogs

    This selector subclass is designed for metacal-type catalogs like those
    used in Dark Energy Survey Y1 and Y3 data releases. In DESC they are
    superseded by MetaDetect, see below.
    """

    name = "TXSourceSelectorMetacal"

    # add one option to the base class configuration
    config_options = {
        **TXSourceSelectorBase.config_options,
        "delta_gamma": float,
        "use_mean_diag": False
    }


    # The main differences between the classes are to do with how the data is read
    # and what output response values are generated.
    def data_iterator(self):
        """
        This iterator returns chunks of data in dictionaries one by one.

        We call to a parent class method to do the main iteration; the work here is
        just choosing which columns to read.
        """
        bands = self.config["bands"]
        shear_cols = metacal_variants(
            "mcal_T", "mcal_s2n", "mcal_g1", "mcal_g2", "mcal_flags", "weight"
        )
        shear_cols += ["ra", "dec", "mcal_psf_T_mean"]
        shear_cols += band_variants(
            bands, "mcal_mag", "mcal_mag_err", shear_catalog_type="metacal"
        )

        if self.config["input_pz"]:
            shear_cols += metacal_variants("mean_z")
        elif self.config["true_z"]:
            shear_cols += ["redshift_true"]

        chunk_rows = self.config["chunk_rows"]
        return self.iterate_hdf("shear_catalog", "shear", shear_cols, chunk_rows)#, longest=True)

    def setup_output(self):
        """
        Prepare the output columns for the response values generated by metacal.
        We save:
            R_gamma: the per-object estimator response
            R_S: the per-bin selection response
            R_gamma_mean: the mean per-bin estimator response
            R_total: the complete per-bin response

        and the 2D versions of the per-bin values.
        """
        # This call to the super-class method defined above sets up most of the output
        # here, so the rest of this method only does things specific to this
        # calibration scheme
        outfile = super().setup_output()
        n = outfile["tomography/bin"].size
        nbin_source = outfile["tomography/counts"].size
        group = outfile.create_group("response")
        group.create_dataset("R_gamma", (n, 2, 2), dtype="f")
        group.create_dataset("R_S", (nbin_source, 2, 2), dtype="f")
        group.create_dataset("R_gamma_mean", (nbin_source, 2, 2), dtype="f")
        group.create_dataset("R_total", (nbin_source, 2, 2), dtype="f")
        group.create_dataset("R_S_2d", (2, 2), dtype="f")
        group.create_dataset("R_gamma_mean_2d", (2, 2), dtype="f")
        group.create_dataset("R_total_2d", (2, 2), dtype="f")
        return outfile

    def setup_response_calculators(self, nbin_source):
        delta_gamma = self.config["delta_gamma"]
        use_diagonal_response = self.config["use_diagonal_response"]
        calculators = [
            MetacalCalculator(self.select, delta_gamma,use_diagonal_response) for i in range(nbin_source)
        ]
        calculators.append(MetacalCalculator(self.select_2d, delta_gamma,use_diagonal_response))
        return calculators

    def write_tomography(self, outfile, start, end, source_bin, R):
        # This write the per-object response values as well in addition
        # to the general values that are written in the base class.
        super().write_tomography(outfile, start, end, source_bin, R)
        group = outfile["response"]
        group["R_gamma"][start:end, :, :] = R

    def compute_per_object_response(self, data):
        delta_gamma = self.config["delta_gamma"]
        n = data["mcal_g1_1p"].size
        R = np.zeros((n, 2, 2))
        R[:, 0, 0] = (data["mcal_g1_1p"] - data["mcal_g1_1m"]) / delta_gamma
        R[:, 0, 1] = (data["mcal_g1_2p"] - data["mcal_g1_2m"]) / delta_gamma
        R[:, 1, 0] = (data["mcal_g2_1p"] - data["mcal_g2_1m"]) / delta_gamma
        R[:, 1, 1] = (data["mcal_g2_2p"] - data["mcal_g2_2m"]) / delta_gamma
        return R

    def apply_simple_redshift_cut(self, data):
        # If we have the truth pz then we just need to do the binning once,
        # as in the parent class
        if self.config["true_z"]:
            return super().apply_simple_redshift_cut(data)

        # Otherwise we have to do it once for each variant, because
        # we are using a point estimate and should have different
        # values of it for each shear.

        pz_data = {}
        variants = ["", "_1p", "_2p", "_1m", "_2m"]
        for v in variants:
            zz = data[f"mean_z{v}"]

            pz_data_v = np.zeros(len(zz), dtype=int) - 1
            for zi in range(len(self.config["source_zbin_edges"]) - 1):
                mask_zbin = (zz >= self.config["source_zbin_edges"][zi]) & (
                    zz < self.config["source_zbin_edges"][zi + 1]
                )
                pz_data_v[mask_zbin] = zi

            pz_data[f"zbin{v}"] = pz_data_v

        return pz_data

    def compute_output_stats(self, calculator, mean, variance):
        """
        Collect the per-bin response values, and the shear means and variances.
        These are calculated in a distributed way across different processes,
        so here we bring them together.

        We collate these into a BinStats object for clarity.
        """
        R, S, N, Neff = calculator.collect(self.comm, allgather=True)
        calibrator = MetaCalibrator(R, S, mean, mu_is_calibrated=False)
        mean_e = calibrator.mu.copy()

        Rtot = R + S
        P = np.diag(np.linalg.inv(Rtot @ Rtot))

        # Apply to the variances to get sigma_e
        sigma_e = np.sqrt(0.5 * P @ variance)

        # In metacal all weights are unity, so the effective N is the same
        # as the raw N.
        return BinStats(N, Neff, mean_e, sigma_e, calibrator)


class TXSourceSelectorMetadetect(TXSourceSelectorBase):
    """
    Source selection and tomography for metadetect catalogs

    This subclass selects for MetaDetect catalogs, which is expected to be used for
    Rubin data. It computes the selection bias due to object detection by repeating
    the detection process under different applied shears.

    As a consequence the different calibration columns have different lengths, since
    different objects are detected in each case.
    """

    name = "TXSourceSelectorMetadetect"

    # add one option to the base class configuration
    config_options = {
        **TXSourceSelectorBase.config_options,
        "delta_gamma": float
    }


    def data_iterator(self):
        # As above, this is where we work out which columns we need.
        chunk_rows = self.config["chunk_rows"]
        bands = self.config["bands"]

        # Core quantities we need
        shear_cols = metadetect_variants(
            "T", "s2n", "g1", "g2", "ra", "dec", "mcal_psf_T_mean", "weight", "flags"
        )

        # Magnitudes and errors
        shear_cols += band_variants(
            bands, "mag", "mag_err", shear_catalog_type="metadetect"
        )

        # We need truth shears and/or PZ point-estimates for each shear too
        if self.config["input_pz"]:
            shear_cols += metadetect_variants("mean_z")
        elif self.config["true_z"]:
            shear_cols += ["redshift_true"]

        renames = {}
        for prefix in ["00", "1p", "1m", "2p", "2m"]:
            renames[f"{prefix}/mcal_psf_T_mean"] = f"{prefix}/psf_T_mean"

        # This is a parent ceci.PipelineStage method.
        # It returns an iterator we loop through
        it = self.iterate_hdf(
            "shear_catalog", "shear", shear_cols, chunk_rows, longest=True
        )
        return rename_iterated(it, renames)

    def setup_response_calculators(self, nbin_source):
        delta_gamma = self.config["delta_gamma"]
        calculators = [
            MetaDetectCalculator(self.select, delta_gamma) for i in range(nbin_source)
        ]
        calculators.append(MetaDetectCalculator(self.select_2d, delta_gamma))
        return calculators

    def apply_simple_redshift_cut(self, data):
        # If we have the truth pz then we just need to do the binning once,
        # as in the parent class
        if self.config["true_z"]:
            return super().apply_simple_redshift_cut(data)

        # Otherwise we have to do it once for each variant
        pz_data = {}
        variants = ["00/", "1p/", "2p/", "1m/", "2m/"]
        for v in variants:
            zz = data[f"{v}mean_z"]

            pz_data_v = np.zeros(len(zz), dtype=int) - 1
            for zi in range(len(self.config["source_zbin_edges"]) - 1):
                mask_zbin = (zz >= self.config["source_zbin_edges"][zi]) & (
                    zz < self.config["source_zbin_edges"][zi + 1]
                )
                pz_data_v[mask_zbin] = zi

            pz_data[f"{v}zbin"] = pz_data_v

        return pz_data

    def setup_output(self):
        """
        MetaDetect outputs do not include per-object calibration values,
        only the per-bin values.
        """
        # This call to the super-class method defined above sets up most of the output
        # here, so the rest of this method only does things specific to this
        # calibration scheme
        outfile = super().setup_output()
        n = outfile["tomography/bin"].size
        nbin_source = outfile["tomography/counts"].size
        group = outfile.create_group("response")

        # Per-bin 2x2 calibration matrix
        group.create_dataset("R", (nbin_source, 2, 2), dtype="f")
        # Global calibration matrix
        group.create_dataset("R_2d", (2, 2), dtype="f")
        return outfile

    def compute_output_stats(self, calculator, mean, variance):
        # Collate calibration values
        R, N, Neff = calculator.collect(self.comm, allgather=True)
        calibrator = MetaDetectCalibrator(R, mean, mu_is_calibrated=False)
        mean_e = calibrator.mu.copy()

        # Apply to the variances to get sigma_e
        P = np.diag(np.linalg.inv(R @ R))
        sigma_e = np.sqrt(0.5 * P @ variance)

        # Like metacal, N_eff = N for metadetect
        return BinStats(N, Neff, mean_e, sigma_e, calibrator)


class TXSourceSelectorLensfit(TXSourceSelectorBase):
    """
    Source selection and tomography for lensfit catalogs

    This selector class is for Lensfit catalogs like those used in KIDS.

    It is a simpler calibration scheme than the above two, and does not involve
    variant catalogs, just taking the mean of a value for one catalog.
    """

    name = "TXSourceSelectorLensfit"

    # add one option to the base class configuration
    config_options = {
        **TXSourceSelectorBase.config_options,
        "input_m_is_weighted": bool
    }


    def data_iterator(self):
        chunk_rows = self.config["chunk_rows"]
        bands = self.config["bands"]
        shear_cols = [
            "psf_T_mean",
            "weight",
            "flags",
            "T",
            "s2n",
            "g1",
            "g2",
            "weight",
            "m",
        ]
        shear_cols += band_variants(
            bands, "mag", "mag_err", shear_catalog_type="lensfit"
        )
        if self.config["input_pz"]:
            shear_cols += ["mean_z"]
        elif self.config["true_z"]:
            shear_cols += ["redshift_true"]
        return self.iterate_hdf("shear_catalog", "shear", shear_cols, chunk_rows)

    def setup_response_calculators(self, nbin_source):
        calculators = [
            LensfitCalculator(self.select, self.config["input_m_is_weighted"])
            for i in range(nbin_source)
        ]
        calculators.append(
            LensfitCalculator(self.select_2d, self.config["input_m_is_weighted"])
        )
        return calculators

    def setup_output(self):
        # This call to the super-class method defined above sets up most of the output
        # here, so the rest of this method only does things specific to this
        # calibration scheme
        outfile = super().setup_output()
        n = outfile["tomography/bin"].size
        nbin_source = outfile["tomography/counts"].size
        group = outfile.create_group("response")
        group.create_dataset("K", (nbin_source,), dtype="f")
        group.create_dataset("C", (nbin_source, 2), dtype="f")
        group.create_dataset("K_2d", (1,), dtype="f")
        group.create_dataset("C_2d", (2), dtype="f")
        return outfile

    def compute_output_stats(self, calculator, mean, variance):
        K, C, N, Neff = calculator.collect(self.comm, allgather=True)
        calibrator = LensfitCalibrator(K, C)
        mean_e = C.copy()
        sigma_e = np.sqrt((0.5 * (variance[0] + variance[1]))) / (1 + K)

        return BinStats(N, Neff, mean_e, sigma_e, calibrator)


class TXSourceSelectorHSC(TXSourceSelectorBase):
    """
    Source selection and tomography for HSC catalogs

    This subclass is for selecting objects on catalogs of the form made by HSC.

    This scheme is quite similar to the one used by lensfit. The main difference is in
    the per-object response.

    TODO: The HSC calibrator is currently broken, and will crash when it gets
    to compute_output_stats
    """

    name = "TXSourceSelectorHSC"
    config_options = TXSourceSelectorBase.config_options.copy()
    config_options["max_shear_cut"] = 0.0
    
    def data_iterator(self):
        chunk_rows = self.config["chunk_rows"]
        bands = self.config["bands"]

        # Select columns we need.
        shear_cols = [
            "psf_T_mean",
            "weight",
            "flags",
            "T",
            "s2n",
            "g1",
            "g2",
            "weight",
            "m",
            "c1",
            "c2",
            "sigma_e",
        ]
        shear_cols += band_variants(bands, "mag", "mag_err", shear_catalog_type="hsc")
        if self.config["input_pz"]:
            shear_cols += ["mean_z"]
        elif self.config["true_z"]:
            shear_cols += ["redshift_true"]

        # Iterate using parent class method
        return self.iterate_hdf("shear_catalog", "shear", shear_cols, chunk_rows)

    def setup_output(self):
        # This call to the super-class method defined above sets up most of the output
        # here, so the rest of this method only does things specific to this
        # calibration scheme
        outfile = super().setup_output()
        n = outfile["tomography/bin"].size
        nbin_source = outfile["tomography/counts"].size
        group = outfile.create_group("response")

        # There is a single scalar per-object value for this scheme
        group.create_dataset("R", (n,), dtype="f")

        # and a set of additive and multiplicative factors.
        # The K and R values are degenerate.
        group.create_dataset("K", (nbin_source,), dtype="f")
        group.create_dataset("C", (nbin_source, 2), dtype="f")
        group.create_dataset("R_mean", (nbin_source,), dtype="f")
        group.create_dataset("K_2d", (1,), dtype="f")
        group.create_dataset("C_2d", (2), dtype="f")
        group.create_dataset("R_mean_2d", (1,), dtype="f")
        return outfile

    def write_tomography(self, outfile, start, end, source_bin, R):
        super().write_tomography(outfile, start, end, source_bin, R)
        group = outfile["response"]
        group["R"][start:end] = R

    def compute_per_object_response(self, data):
        w_tot = np.sum(data["weight"])
        R = np.array(
            [1.0 - np.sum(data["weight"] * data["sigma_e"]) / w_tot]
            * len(data["weight"])
        )
        return R

    def compute_output_stats(self, calculator, mean, variance):
        R, K, N, Neff = calculator.collect(self.comm, allgather=True)
        calibrator = HSCCalibrator(R, K)
        sigma_e = np.sqrt((0.5 * (variance[0] + variance[1]))) / (1 + K)
        return BinStats(N, Neff, mean, sigma_e, calibrator)

    def setup_response_calculators(self, nbin_source):
        calculators = [
            HSCCalculator(self.select)
            for i in range(nbin_source)
        ]
        calculators.append(
            HSCCalculator(self.select_2d)
        )
        return calculators

    def select_2d(self, data, calling_from_select=False):
        """
        Add an additional cut to the parent class, if specified, on the max shear.
        HSM DP0.2 catalogs seem to contain occasional very large shears that skew peaks.
        This removes those. This is only really for testing.
        """
        sel = super().select_2d(data, calling_from_select=calling_from_select)
        shear_cut = self.config["max_shear_cut"]
        if shear_cut:
            g = np.sqrt(data["g1"] ** 2 + data["g2"] ** 2)
            cut = g < shear_cut
            p = 100 * (1 - (cut.sum() / cut.size))
            print(f" shear cut removes {p:.2f}% of objects")
            sel &= cut
            p = sel.sum() / sel.size * 100
            print(f" after shear cut retain {p:.2f}% of objects")
        return sel

if __name__ == "__main__":
    PipelineStage.main()
