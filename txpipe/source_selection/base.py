from ..base_stage import PipelineStage
from ..data_types import ShearCatalog, TomographyCatalog, HDFFile, PickleFile, CatalogSelection, BinAssignments
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
        ("shear_bin_assignments", BinAssignments),
        ("shear_selection", CatalogSelection),
    ]

    outputs = [("shear_tomography_catalog", TomographyCatalog)]

    config_options = {
        "input_pz": StageParameter(bool, False, msg="Whether to use input photo-z posteriors"),
        "true_z": StageParameter(bool, False, msg="Whether to use true redshifts instead of photo-z"),
        "bands": StageParameter(list, ["r", "i", "z"], msg="Bands from the catalog to use for selection"),
        "verbose": StageParameter(bool, False, msg="Whether to print verbose output"),
        "chunk_rows": StageParameter(int, 10000, msg="Number of rows to process in each chunk"),
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

        nbin_source = -1
        self.config["nbin_source"] = nbin_source

        # We will collect the selection biases for each bin
        # as a matrix.  We will collect together the different
        # matrices for each chunk and do a weighted average at the end.
        calculators = self.setup_response_calculators(nbin_source)

        # The output file we will put the tomographic
        # information into
        output_file = self.setup_output(nbin_source)

        # The iterator that will loop through the data.
        # Set it up here so that we can find out if there are any
        # problems with it before we get run the slow classifier.
        it = self.data_iterator()


        # Loop through the input data, processing it chunk by chunk
        for start, end, shear_data in it:
            print(f"Process {self.rank} running selection for rows {start:,}-{end:,}")

            # Combine this selection with size and snr cuts to produce a source selection
            # and calculate the shear bias it would generate.
            # Note that the per-object response can be None for methods like MetaDetect that
            # only calibrate the entire bin
            per_object_response = self.accumulate_selection(pz_data, shear_data, calculators)

            # Save the tomography for this chunk
            self.write_tomography(output_file, start, end, tomo_bin, per_object_response)


        # Do the selection bias averaging and output that too.
        self.write_global_values(output_file, calculators)

        # Save and complete
        output_file.close()

        # Restore the original warning settings in case we are being called from a library
        np.seterr(**original_warning_settings)

    def select(self, data, bin_index):
        zbin = data["zbin"]
        sel = data["selected"]
        return (zbin == bin_index) & sel

    def select_any_bin(self, data):
        zbin = data["zbin"]
        sel = data["selected"]
        return (zbin >= 0) & sel

    def columns(self):
        return [
            {
                "shear_catalog": ["g1", "g2", "weight"],
                "shear_bin_assignments": ["zbin"],
                "shear_selection": ["selection"],
            }
        ]



    def columns(self):
        # for metadetect
        return [
            {
                "shear_catalog": [f"{v}/g1", f"{v}/g2", f"{v}/weight"],
                "shear_bin_assignments": f"{v}/zbin",
                "shear_selection": f"{v}/selection",
            }
            for v in META_VARIANTS
        ]

    def column_groups(self):
        # for scalar_metadetect
        return [
            {
                "shear_catalog": [f"{v}/g1", f"{v}/g2", f"{v}/weight"],
                "shear_bin_assignments": f"{v}/zbin",
                "shear_selection": f"{v}/selection",
            }
            for v in SCALAR_META_VARIANTS
        ]




    def data_iterator(self):
        # iterator over:
        # - the "sel" columns in the selection catalog
        # - the "zbin" columns in the tomo assignments
        # - the g1, g2, and weight columns from the shear catalog
        # - any other columns from the shear catalog that subclassses need
        with self.open_input("shear_catalog", wrapper=True) as f:
            calib_cols = f.get_extra_calibration_columns()


        for col_group in self.column_groups():
            for (s, e, data) in self.combined_iterators(
                chunk_rows,
                "shear_catalog",
                "shear",
                # fortunately the metadetect and metacal catalogs don't have any additional calib_cols
                # or this would not quite work
                col_group["shear_catalog"] + calib_cols,
                "shear_bin_assignments",
                "/",
                col_group["shear_bin_assignments"],
                "shear_selection",
                "/",
                col_group["shear_selection"],
            ):
                yield (s, e, data)

    def accumulate_selection(self, pz_data, shear_data, calculators):
        """
        Select objects to go in each tomographic bin and their calibration.

        Parameters
        ----------

        pz_data: table or dict of arrays
            A chunk of input photo-z data containing mean values for each object
        shear_data: table or dict of arrays
            A chunk of input shear data with metacalibration variants.
        """
        nbin = self.config['nbin_source']
        n = len(list(shear_data.values())[0])

        # The main output data - the tomographic
        # bin index for each object, or -1 for no bin.
        tomo_bin = np.repeat(-1, n)

        data = {**pz_data, **shear_data}

        R = self.compute_per_object_response(data)

        for i in range(nbin):
            sel_00 = calculators[i].add_data(data, i)
            tomo_bin[sel_00] = i

        # and calibrate the 2D sample.
        # This calibrator refers to select_weak_lensing_sample
        calculators[-1].add_data(data, self.config)

        return tomo_bin, R

    def compute_per_object_response(self, data):
        # The default implementation has no per-object response
        # Some subclasses supply it.
        return None

    def setup_output(self, nbin_source):
        """
        Set up the output data file.

        Creates the data sets and groups to put module output
        in the shear_tomography_catalog output file.
        """

        cat_type = read_shear_catalog_type(self)

        with self.open_input("shear_catalog", wrapper=True) as f:
            n = f.get_size()


        output = self.open_output("shear_tomography_catalog", parallel=True, wrapper=True)
        outfile = output.file
        group = outfile.create_group("tomography")
        group.attrs["catalog_type"] = cat_type


        if self.config['do_tomography']:
           zbins = self.config["source_zbin_edges"]
        else:
            # Use mock z limits here in the output
            # to indicate everything is selected.
            zbins = [0.0, 999.]
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
        # hard link to bin column. needed for RAIL masked summarizer stages
        outfile["tomography/class_id"] = outfile["tomography/bin"]

        for i, calculator in enumerate(calculators):
            stats = calculator.collect(self.comm, allgather=True)
            if self.rank == 0:
                stats.write_to(outfile, i if i < len(calculators) - 1 else "2d")


