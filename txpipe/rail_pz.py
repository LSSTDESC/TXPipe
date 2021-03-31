from .base_stage import PipelineStage
from .data_types import PhotozPDFFile, HDFFile, PickleFile, DataFile
import numpy as np


class PZRailTrain(PipelineStage):
    """Train a photo-z model using RAIL.

    The Redshift Assessment Infrastructure Layers (RAIL) library provides a uniform
    interface to DESC photo-z code.

    TXPipe uses RAIL across several different pipeline stages.
    This stage, which would normally be run first, uses a training set (e.g. of
    spectroscopic redshifts) to train and save an Estimator object that a later
    stage can use to measure redshifts of the survey sample.
    """
    name = "PZRailTrain"

    inputs = [
        ("photoz_training", DataFile),
        ("photoz_testing", DataFile),
    ]

    outputs = [("photoz_trained_model", PickleFile)]

    config_options = {
        "class_name": str,
        "chunk_rows": 10000,
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

        # Afterwards, save the model.  This assumes that estimator
        # classes can be pickled, which is true for now but see the
        # issue opened on th RAIL repo
        with self.open_output("photoz_trained_model", wrapper=True) as output:
            output.write(estimator)


class PZRailEstimate(PipelineStage):
    """Run a trained RAIL estimator to estimate PDFs and best-fit redshifts

    We load a redshift Estimator model, typically saved by the PZRailTrain stage,
    and then load chunks of photometry and run the estimator on it, and save the
    result.

    RAIL currently returns the PDF and then only the modal z as a point estimate,
    so we save that.  Previous stages have returned mean or median methods too.
    We could ask for this in RAIL, or calculate the mean or median ourselves.

    There's currently a slight ambiguity about bin edges vs bin centers that I
    will follow up on with the RAIL team.

    The training stage is (currently all) serial, but applying the trained
    model can be done in parallel, so we split into two stages to avoid
    many processors sitting idle or repeating the same training process.

    """
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

        # Load the estimator trained in PZRailTrain
        with self.open_input("photoz_trained_model", wrapper=True) as f:
            estimator = f.read()

        # prepare the output data - we will save things to this
        # as we go along.  We also need the z grid becauwe we use
        # it to get the mean z from the PDF
        output, z = self.setup_output_file(estimator)

        # Create the iterator the reads chunks of photometry
        # The method we use here automatically splits up data when we run in parallel
        cols = [f"mag_{b}" for b in "ugrizy"] + [f"mag_err_{b}" for b in "ugrizy"]
        chunk_rows = self.config["chunk_rows"]
        it = self.iterate_hdf("photometry_catalog", "photometry", cols, chunk_rows)

        # Loop through the chunks of data
        for s, e, data in it:
            print(f"Process {self.rank} estimating PZ PDF for rows {s:,} - {e:,}")
            # Rename things so col names match what RAIL expects
            self.rename_columns(data)

            # Run the pre-trained estimator
            pz_data = estimator.estimate(data)

            # Save the results
            self.write_output_chunk(output, s, e, z, pz_data)

    def rename_columns(self, data):
        # RAIL expects the magnitudes and errors to have
        # the suffix _lsst, which we add here
        for band in "ugrizy":
            data[f"mag_{band}_lsst"] = data[f"mag_{band}"]
            data[f"mag_err_{band}_lsst"] = data[f"mag_err_{band}"]

    def setup_output_file(self, estimator):
        # Briefly check the size of the catalog so we know how much
        # space to reserve in the output
        with self.open_input("photometry_catalog") as f:
            nobj = f["photometry/ra"].size

        # open the output file
        f = self.open_output("photoz_pdfs", parallel=True)
        # copied from RAIL as it doesn't seem to get saved at least
        # in the flexzboost version.  Need to understand z edges vs z mid
        z = np.linspace(estimator.zmin, estimator.zmax, estimator.nzbins)
        nz = z.size

        # create the spaces in the output
        pdfs = f.create_group("pdf")
        pdfs.create_dataset("zgrid", (nz,))
        pdfs.create_dataset("pdf", (nobj, nz), dtype="f4")

        modes = f.create_group("point_estimates")
        modes.create_dataset("z_mode", (nobj,), dtype="f4")
        modes.create_dataset("z_mean", (nobj,), dtype="f4")

        # One processor writes the redshift axis to output.
        if self.rank == 0:
            pdfs["zgrid"][:] = z

        return f, z

    def write_output_chunk(self, output_file, start, end, z, pz_data):
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
        # RAIL does not currently output the mean z by default, so
        # we compute it here
        p = pz_data['pz_pdf']
        mu = (p @ z) / p.sum(axis=1)

        output_file["pdf/pdf"][start:end] = p
        output_file["point_estimates/z_mode"][start:end] = pz_data["zmode"]
        output_file["point_estimates/z_mean"][start:end] = mu
