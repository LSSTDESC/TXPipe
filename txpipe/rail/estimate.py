from ..base_stage import PipelineStage
from ..data_types import PhotozPDFFile, HDFFile, PickleFile
from ..utils import rename_iterated
import numpy as np
import shutil


class PZRailEstimateSource(PipelineStage):
    """
    Run a trained RAIL estimator to estimate PDFs and best-fit redshifts

    We load a redshift Estimator model, typically saved by the PZRailTrainSource stage,
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

    name = "PZRailEstimateSource"
    model_input = "photoz_source_model"
    pdf_output = "source_photoz_pdfs"

    inputs = [
        ("shear_catalog", HDFFile),
        ("photoz_source_model", PickleFile),
    ]

    outputs = [
        ("source_photoz_pdfs", PhotozPDFFile),
    ]

    config_options = {
        "chunk_rows": 10000,
        "bands": "riz",
        "mag_prefix": "mcal_",
    }

    def run(self):
        # Importing this means that we can unpickle the relevant class
        import rail.estimation

        # Load the estimator trained in PZRailTrain
        estimator = self.load_model()

        # prepare the output data - we will save things to this
        # as we go along.  We also need the z grid becauwe we use
        # it to get the mean z from the PDF
        output, z = self.setup_output_file(estimator)

        # Loop through the chunks of data
        for s, e, data in self.data_iterator():
            print(f"Process {self.rank} estimating PZ PDF for rows {s:,} - {e:,}")

            # Run the pre-trained estimator
            pz_data = estimator.estimate(data)

            # Save the results
            self.write_output_chunk(output, s, e, z, pz_data)

    def load_model(self):
        with self.open_input(self.model_input, wrapper=True) as f:
            estimator = f.read()
        return estimator

    def data_iterator(self):
        bands = self.config["bands"]
        chunk_rows = self.config["chunk_rows"]
        prefix = self.config["mag_prefix"]

        # Columns we will load, and the new names we will give them
        renames = {}
        for band in bands:
            renames[f"{prefix}mag_{band}"] = f"mag_{band}_lsst"
            renames[f"{prefix}mag_err_{band}"] = f"mag_err_{band}_lsst"

        # old names, as loaded from file
        cols = list(renames.keys())

        # Create the iterator the reads chunks of photometry
        # The method we use here automatically splits up data when we run in parallel
        it = self.iterate_hdf("shear_catalog", "shear", cols, chunk_rows)

        # Also rename all the columns as they are loaded
        return rename_iterated(it, renames)

    def get_catalog_size(self):
        with self.open_input("shear_catalog") as f:
            nobj = f["shear/ra"].size
        return nobj

    def setup_output_file(self, estimator):
        # Briefly check the size of the catalog so we know how much
        # space to reserve in the output
        nobj = self.get_catalog_size()

        # open the output file
        f = self.open_output(self.pdf_output, parallel=True)
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
        p = pz_data["pz_pdf"]
        mu = (p @ z) / p.sum(axis=1)

        output_file["pdf/pdf"][start:end] = p
        output_file["point_estimates/z_mode"][start:end] = pz_data["zmode"]
        output_file["point_estimates/z_mean"][start:end] = mu


class PZRailEstimateLens(PZRailEstimateSource):
    name = "PZRailEstimateLens"
    """
    Use RAIL to estimate the PDFs of the source sample.

    This is implemented as a subclass of PZRailEstimateSource
    but with the loaded input data changed - see that class
    for more details.

    """

    model_input = "photoz_lens_model"
    pdf_output = "lens_photoz_pdfs"

    inputs = [
        ("photometry_catalog", HDFFile),
        ("photoz_lens_model", PickleFile),
    ]

    outputs = [
        ("lens_photoz_pdfs", PhotozPDFFile),
    ]

    config_options = {
        "chunk_rows": 10000,
        "bands": "ugrizy",
    }

    def data_iterator(self):
        bands = self.config["bands"]
        chunk_rows = self.config["chunk_rows"]

        # Columns we will load, and the new names we will give them
        renames = {}
        for band in bands:
            renames[f"mag_{band}"] = f"mag_{band}_lsst"
            renames[f"mag_err_{band}"] = f"mag_err_{band}_lsst"

        # old names, as loaded from file
        cols = list(renames.keys())

        # Create the iterator that reads chunks of photometry
        # The method we use here automatically splits up data when we run in parallel
        it = self.iterate_hdf("photometry_catalog", "photometry", cols, chunk_rows)

        # Also rename all the columns as they are loaded
        # using a new function.
        return rename_iterated(it, renames)

    def get_catalog_size(self):
        with self.open_input("photometry_catalog") as f:
            nobj = f["photometry/ra"].size
        return nobj


class PZRailEstimateSourceFromLens(PipelineStage):
    """
    In cases where source and lens come from the same base sample
    we can simply copy the computed PDFs from lens to source.
    """

    name = "PZRailEstimateSourceFromLens"

    inputs = [("lens_photoz_pdfs", PhotozPDFFile)]
    outputs = [("source_photoz_pdfs", PhotozPDFFile)]

    def run(self):
        shutil.copy(
            self.get_input("lens_photoz_pdfs"),
            self.get_output("source_photoz_pdfs"),
        )


class PZRailEstimateLensFromSource(PipelineStage):
    """
    In cases where source and lens come from the same base sample
    we can simply copy the computed PDFs from lens to source.
    """

    name = "PZRailEstimateLensFromSource"

    inputs = [("source_photoz_pdfs", PhotozPDFFile)]
    outputs = [("lens_photoz_pdfs", PhotozPDFFile)]

    def run(self):
        shutil.copy(
            self.get_input("source_photoz_pdfs"),
            self.get_output("lens_photoz_pdfs"),
        )
