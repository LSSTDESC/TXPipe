from .base_stage import PipelineStage
from .data_types import PhotozPDFFile, HDFFile, PickleFile
import numpy as np


class PZRailTrain(PipelineStage):
    name = "PZRailTrain"

    inputs = [
        ("photoz_training", HDFFile),
        ("photoz_testing", HDFFile),
    ]

    outputs = [("photoz_trained_model", PickleFile)]

    config_options = {
        "class_name": str,
        "zmin": 0.0,
        "zmax": 3.0,
        "nzbins": 301,
        # other estimator-specific config
        # can be put in the config file and
        # will end up in self.config
    }

    def run(self):
        from rail.estimation.estimator import Estimator

        # General config information that RAIL wants
        base_config = {
            "trainfile": self.get_input("photoz_training"),
            "testfile": self.get_input("photoz_testing"),
            "hdf5_groupname": "photometry",
            "chunk_size": self.config["chunk_rows"],
            "outpath": None,  # should not be used here
        }

        # Additional confguration specific to this algorithm
        # can be included in the config
        run_dict = {
            "run_params": self.config.copy(),
        }

        # create an instance of the specific estimation
        # method
        cls = Estimator._find_subclass(self.config["class_name"])
        estimator = cls(base_config, run_dict)

        # Run the main training phase
        estimator.inform()

        # If there is any kind of testing/validation to be run we could
        # do so here.

        # Otherwise, save the model.  This assumes that estimator
        # classes can be pickled, which is true for now but see the
        # issue opened on th RAIL repo
        with self.open_output("photoz_trained_model", wrapper=True) as output:
            output.write(estimator)


class PZRailEstimate(PipelineStage):
    name = "PZRailEstimate"

    inputs = [
        ("photometry_catalog", HDFFile),
        ("photoz_trained_model", PickleFile),
    ]

    outputs = [
        ("photoz_pdfs", PhotozPDFFile),
    ]

    config_options = {
        "chunk_rows": 10000,
    }

    def run(self):
        # Importing this means that we can unpickle the relevant class
        import rail.estimation

        with self.open_input("photoz_trained_model", wrapper=True) as f:
            estimator = f.read()

        with self.open_input("photometry_catalog") as f:
            nobj = f["photometry/ra"].size

        output = self.setup_output_file(estimator, nobj)

        cols = [f"mag_{b}" for b in "ugrizy"] + [f"mag_err_{b}" for b in "ugrizy"]
        chunk_rows = self.config["chunk_rows"]
        for s, e, data in self.iterate_hdf(
            "photometry_catalog", "photometry", cols, chunk_rows
        ):
            # Rename things so col names match what RAIL expects
            self.rename_columns(data)

            # Run the pre-trained estimator
            pz_data = estimator.estimate(data)

            # Save the results
            self.write_output_chunk(output, s, e, pz_data)

    def rename_columns(self, data):
        # RAIL expects the magnitudes and errors to have
        # the suffix _lsst, which we add here
        for band in "ugrizy":
            data[f"mag_{band}_lsst"] = data[f"mag_{band}"]
            data[f"mag_err_{band}_lsst"] = data[f"mag_err_{band}"]

    def setup_output_file(self, estimator, nobj):
        f = self.open_output("photoz_pdfs", parallel=True)
        # copied from RAIL as it doesn't seem to get saved at least
        # in the flexzboost version.  Need to understand z edges vs z mid
        z = np.linspace(estimator.zmin, estimator.zmax, estimator.nzbins)
        nz = z.size

        pdfs = f.create_group("pdf")
        pdfs.create_dataset("zgrid", (nz,))
        pdfs.create_dataset("pdf", (nobj, nz), dtype="f4")

        modes = f.create_group("point_estimates")
        modes.create_dataset("z_mode", (nobj,), dtype="f4")

        # One processor writes the redshift axis to output.
        if self.rank == 0:
            pdfs["zgrid"][:] = z

        return f

    def write_output_chunk(self, output_file, start, end, pz_data):
        """
        Write out a chunk of the computed PZ data.

        Parameters
        ----------

        output_file: h5py.File
            The object we are writing out to

        start: int
            The index into the full range of data that this chunk starts at

        end: int
            The index into the full range of data that this chunk ends at

        pz_data: dict
            As returned by rail, containing zmode and pz_pdf
        """
        output_file["pdf/pdf"][start:end] = pz_data["pz_pdf"]
        output_file["point_estimates/z_mode"][start:end] = pz_data["zmode"]
