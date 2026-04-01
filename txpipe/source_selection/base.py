from ..base_stage import PipelineStage
from ..data_types import ShearCatalog, TomographyCatalog, HDFFile, PickleFile
from .utils import SourceNumberDensityStats
from ..utils.calibration_tools import read_shear_catalog_type
from ..binning import build_tomographic_classifier, apply_classifier, read_training_data
from ceci.config import StageParameter
import numpy as np


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
        group = outfile["counts"]
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
            group["N_eff"][i] = self.N_eff
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
    the training data in a spectroscopic_catalog, or according to
    the true shear (if present), according to the choice the
    user makes in the configuration.

    Once these selections are made it constructs
    the quantities needed to calibrate each bin,
    generating a set of Calibrator objects.
    """

    name = "TXSourceSelector"

    inputs = [
        ("shear_catalog", ShearCatalog),
        ("spectroscopic_catalog", HDFFile),
        ("shear_tomography_classifier", PickleFile),
    ]

    outputs = [("shear_tomography_catalog", TomographyCatalog)]

    config_options = {
        "input_pz": StageParameter(bool, False, msg="Whether to use input photo-z posteriors"),
        "true_z": StageParameter(bool, False, msg="Whether to use true redshifts instead of photo-z"),
        "bands": StageParameter(list, ["r", "i", "z"], msg="Bands from the catalog to use for selection"),
        "verbose": StageParameter(bool, False, msg="Whether to print verbose output"),
        "T_cut": StageParameter(float, required=True, msg="Size cut threshold for object selection"),
        "s2n_cut": StageParameter(
            float,
            required=True,
            msg="Signal-to-noise cut threshold for object selection",
        ),
        "chunk_rows": StageParameter(int, 10000, msg="Number of rows to process in each chunk"),
        "source_zbin_edges": StageParameter(list, required=True, msg="Redshift bin edges for source tomography"),
    }

    def run(self):
        import astropy.table
        import sklearn.ensemble

        # This base class should no longer be used, so to avoid people
        # accidentally doing so we give a clear message if they try.
        if self.name == "TXSourceSelector":
            raise ValueError(
                "Do not use the class TXSourceSelector any more. Use one of the subclasses like TXSourceSelectorMetacal"
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
            classifier, features = self.read_tomography_classifier()

        # We will collect the selection biases for each bin
        # as a matrix.  We will collect together the different
        # matrices for each chunk and do a weighted average at the end.
        nbin_source = len(self.config["source_zbin_edges"]) - 1

        number_density_stats = SourceNumberDensityStats(nbin_source, comm=self.comm, shear_type=shear_catalog_type)

        calculators = self.setup_response_calculators(nbin_source)

        # Loop through the input data, processing it chunk by chunk
        for start, end, shear_data in it:
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
            # and calculate the shear bias it would generate.
            # Note that the per-object response can be None for methods like MetaDetect that
            # only calibrate the entire bin
            tomo_bin, per_object_response = self.calculate_tomography(pz_data, shear_data, calculators)

            # Save the tomography for this chunk and accumulate the number
            # density information.
            self.accumulate_statistics(output_file, shear_data, start, end, tomo_bin, per_object_response, number_density_stats)


        # Do the selection bias averaging and output that too.
        self.write_global_values(output_file, calculators, number_density_stats)

        # Save and complete
        output_file.close()

        # Restore the original warning settings in case we are being called from a library
        np.seterr(**original_warning_settings)


    def accumulate_statistics(self, output_file, shear_data, start, end, tomo_bin, per_object_response, number_density_stats):
        # Save the tomography for this chunk
        self.write_tomography(output_file, start, end, tomo_bin, per_object_response)

        # Accumulate information on the number counts and the selection biases.
        # These will be brought together at the end.
        number_density_stats.add_data(shear_data, tomo_bin)  # check this

    def read_tomography_classifier(self):
        # Read the tomography classifier from file
        with self.open_input("shear_tomography_classifier", wrapper=True) as infile:
            pickle_data = infile.read()

        # Check that the tomographer used the same configuration
        # as we have here.
        if not pickle_data["bands"] == self.config["bands"]:
            raise ValueError(
                "Bands used in tomography classifier do not match those "
                "in source selector configuration."
            )
        # Also check bin edges are close enough
        if not np.allclose(pickle_data["source_zbin_edges"], self.config["source_zbin_edges"]):
            raise ValueError(
                "Source redshift bin edges used in tomography classifier do not match those "
                "in source selector configuration."
            )

        # This is all that is actually needed in this class
        classifier = pickle_data["classifier"]
        features = pickle_data["features"]
        return classifier, features


    def apply_simple_redshift_cut(self, shear_data):
        if self.config["input_pz"]:
            zz = shear_data["mean_z"]
        else:
            zz = shear_data["redshift_true"]

        pz_data_bin = np.zeros(len(zz), dtype=int) - 1
        for zi in range(len(self.config["source_zbin_edges"]) - 1):
            mask_zbin = (zz >= self.config["source_zbin_edges"][zi]) & (zz < self.config["source_zbin_edges"][zi + 1])
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

        data = {**pz_data, **shear_data}

        R = self.compute_per_object_response(data)

        for i in range(nbin):
            sel_00 = calculators[i].add_data(data, self.config, i)
            tomo_bin[sel_00] = i

        # and calibrate the 2D sample.
        # This calibrator refers to select_weak_lensing_sample
        calculators[-1].add_data(data, self.config)

        return tomo_bin, R

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
        group.attrs["catalog_type"] = cat_type
        output.write_zbins(zbins)
        group.create_dataset("bin", (n,), dtype="i")

        group_count = outfile.create_group("counts")
        group_count.create_dataset("counts", (nbin_source,), dtype="i")
        group_count.create_dataset("counts_2d", (1,), dtype="i")
        group_count.create_dataset("sigma_e", (nbin_source,), dtype="f")
        group_count.create_dataset("sigma_e_2d", (1,), dtype="f")
        group_count.create_dataset("mean_e1", (nbin_source,), dtype="f")
        group_count.create_dataset("mean_e2", (nbin_source,), dtype="f")
        group_count.create_dataset("mean_e1_2d", (1,), dtype="f")
        group_count.create_dataset("mean_e2_2d", (1,), dtype="f")
        group_count.create_dataset("N_eff", (nbin_source,), dtype="f")
        group_count.create_dataset("N_eff_2d", (1,), dtype="f")

        group.attrs["nbin"] = nbin_source

        return outfile

    def write_tomography(self, outfile, start, end, source_bin, per_object_response):
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

        per_object_response: array of shape (nrow,2,2)
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



def select_tomographic_weak_lensing_sample(data, config, bin_index):
    """
    Select which objects are to be chosen in this tomographic bin.
    We do this by calling out to the 2D selector, which does the
    cuts on size and SNR, and then combining this with a cut on tomographic bin.

    Note that we don't call this method directly in the selector classes. Instead
    we pass it to the Calculator objects that call it, sometimes on different
    columns of data.
    """
    zbin = data["zbin"]
    verbose = config["verbose"]

    sel = select_weak_lensing_sample(data, config, calling_from_select=True)
    sel &= zbin == bin_index
    f4 = sel.sum() / sel.size

    if verbose:
        print(f"{f4:.2%} z for bin {bin_index}")
        print("total tomo", sel.sum())

    return sel

def select_weak_lensing_sample(data, config, calling_from_select=False):
    # Select any objects that pass general WL cuts
    # The calling_from_select option just specifies whether we
    # are calling this function from within the select
    # method above, because the useful printed verbose
    # output is different in each case
    s2n_cut = config["s2n_cut"]
    T_cut = config["T_cut"]
    verbose = config["verbose"]
    variant = data.suffix

    shear_prefix = config["shear_prefix"]
    s2n = data[f"{shear_prefix}s2n{variant}"]
    T = data[f"{shear_prefix}T{variant}"]
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
            f"Tomo selection ({variant}) {f1:.2%} flag, {f2:.2%} size, {f3:.2%} SNR, ",
            end="",
        )
    elif verbose:
        print(f"2D selection ({variant}) {f1:.2%} flag, {f2:.2%} size, {f3:.2%} SNR, {f4:.2%} any z bin")
        print("total 2D", sel.sum())
    return sel
