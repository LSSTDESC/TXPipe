from .base import PipelineStage
from ceci.config import StageParameter
from ..data_types import HDFFile, PickleFile
from ..binning import read_training_data, build_tomographic_classifier

class TXSourceTomography(PipelineStage):
    """Train a random forest classifier to assign source galaxies to tomographic bins based on their photometry.

    The trained classified is saved as a pickle file to be used by the selector stage(s).
    This was originally merged in with the selector stage, but it can become slow in situations where the spectroscopic 
    training sample is large, so it has been split out into a separate stage to assist with debugging and re-running
    the selection stages.

    At some point this might be replaced by RAIL stages.
    """
    name = "TXSourceTomography"
    inputs = [
        ("spectroscopic_catalog", HDFFile),
    ]
    outputs = [
        ("shear_tomography_classifier", PickleFile),
    ]
    config_options = {
        "bands": StageParameter(list, ["r", "i", "z"], msg="Bands from the catalog to use for selection"),
        "spec_mag_column_format": StageParameter(str, "photometry/{band}", msg="Format string for spectroscopic magnitude columns"),
        "spec_redshift_column": StageParameter(str, "photometry/redshift", msg="Column name for spectroscopic redshifts"),
        "source_zbin_edges": StageParameter(list, required=True, msg="Redshift bin edges for source tomography"),
        "random_seed": StageParameter(int, 42, msg="Random seed for reproducibility"),
    }

    def run(self):
        bands = self.config['bands']

        # Load the data that we need
        with self.open_input("spectroscopic_catalog") as spec_file:
            training_data = read_training_data(
                spec_file,
                bands,
                self.config["spec_mag_column_format"],
                self.config["spec_redshift_column"],
            )

        # Build the actual classifier.
        classifier, features = build_tomographic_classifier(
            bands,
            training_data,
            self.config["source_zbin_edges"],
            self.config["random_seed"],
            self.comm,
        )

        # Save the classifier.
        with self.open_output("shear_tomography_classifier", wrapper=True) as outfile:
            pickle_data = {
                "classifier": classifier,
                "features": features,
                "bands": bands,
                "source_zbin_edges": self.config["source_zbin_edges"],
            }
            outfile.write(pickle_data)
