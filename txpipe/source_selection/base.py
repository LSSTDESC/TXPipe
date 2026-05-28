from ..base_stage import PipelineStage
from ..data_types import ShearCatalog, TomographyCatalog, HDFFile, PickleFile
from ..utils import read_shear_catalog_type
from ..binning import apply_classifier
from ceci.config import StageParameter
import numpy as np



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

            # Save the tomography for this chunk
            self.write_tomography(output_file, start, end, tomo_bin, per_object_response)


        # Do the selection bias averaging and output that too.
        self.write_global_values(output_file, calculators)

        # Save and complete
        output_file.close()

        # Restore the original warning settings in case we are being called from a library
        np.seterr(**original_warning_settings)

    def read_tomography_classifier(self):
        # Read the tomography classifier from file
        with self.open_input("shear_tomography_classifier", wrapper=True) as infile:
            pickle_data = infile.read()

        # Check that the tomographer used the same configuration
        # as we have here.
        if not sorted(list(pickle_data["bands"])) == sorted(list(self.config["bands"])):
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

    def write_global_values(self, outfile, calculators):
        """
        Write out overall selection biases

        Parameters
        ----------

        outfile: h5py.File

        calculators: list of Calculator objects, one for each tomographic bin and one for the 2D sample,
                     that have been fed all the data and are ready to have their results collected and written out.
        """

        for i, calculator in enumerate(calculators):
            stats = calculator.collect(self.comm, allgather=True)
            if self.rank == 0:
                stats.write_to(outfile, i if i < len(calculators) - 1 else "2d")



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

    s2n = data[f"s2n{variant}"]
    T = data[f"T{variant}"]
    Tpsf = data[f"psf_T_mean"]
    flag = data[f"flags{variant}"]

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
