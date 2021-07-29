from ..base_stage import PipelineStage
from ..data_types import PickleFile, DataFile
import numpy as np
import shutil


class PZRailTrainSource(PipelineStage):
    """Train a photo-z model using RAIL.

    The Redshift Assessment Infrastructure Layers (RAIL) library provides a uniform
    interface to DESC photo-z code.

    TXPipe uses RAIL across several different pipeline stages.
    This stage, which would normally be run first, uses a training set (e.g. of
    spectroscopic redshifts) to train and save an Estimator object that a later
    stage can use to measure redshifts of the survey sample.
    """
    name = "PZRailTrainSource"
    training_tag = "photoz_source_training"
    testing_tag = "photoz_source_testing"
    model_tag = "photoz_source_model"

    inputs = [
        (training_tag, DataFile),
        (testing_tag, DataFile),
    ]

    outputs = [(model_tag, PickleFile)]

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
        from rail.fileIO import load_training_data

        # General config information that RAIL wants
        # TODO: Some of these may not be needed any more
        base_config = {
            "trainfile": self.get_input(self.training_tag),
            "testfile": self.get_input(self.testing_tag),
            "hdf5_groupname": "photometry",
            "chunk_size": self.config["chunk_rows"],
            "outpath": None,  # should not be used here
            "output_format": "old",
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

        training_data = load_training_data(
            self.get_input(self.training_tag),
            "hdf5",
            "photometry"
        )

        # Run the main training phase
        estimator.inform(training_data)

        # If there is any kind of testing/validation to be run we could
        # do so here.

        # Afterwards, save the model.  This assumes that estimator
        # classes can be pickled, which is true for now but see the
        # issue opened on th RAIL repo
        with self.open_output(self.model_tag, wrapper=True) as output:
            output.write(estimator)


class PZRailTrainLens(PZRailTrainSource):
    """Train a photo-z model using RAIL.

    The Redshift Assessment Infrastructure Layers (RAIL) library provides a uniform
    interface to DESC photo-z code.

    TXPipe uses RAIL across several different pipeline stages.
    This stage, which would normally be run first, uses a training set (e.g. of
    spectroscopic redshifts) to train and save an Estimator object that a later
    stage can use to measure redshifts of the survey sample.
    """
    name = "PZRailTrainLens"
    training_tag = "photoz_lens_training"
    testing_tag = "photoz_lens_testing"
    model_tag = "photoz_lens_model"

    inputs = [
        (training_tag, DataFile),
        (testing_tag, DataFile),
    ]

    outputs = [(model_tag, PickleFile)]

class PZRailTrainLensFromSource(PipelineStage):
    """
    Where the same underlying training data is used for
    both source and lens samples, copy the PZ trained model
    for the sources to the one for the lenses.
    """
    name = "PZRailTrainLensFromSource"
    inputs = [("photoz_source_model", PickleFile)]
    outputs = [("photoz_lens_model", PickleFile)]

    def run(self):
        shutil.copy(
            self.get_input("photoz_source_model"),
            self.get_output("photoz_lens_model"),
            )

class PZRailTrainSourceFromLens(PipelineStage):
    """
    Where the same underlying training data is used for
    both source and lens samples, copy the PZ trained model
    for the sources to the one for the lenses.
    """
    name = "PZRailTrainSourceFromLens"
    inputs = [("photoz_lens_model", PickleFile)]
    outputs = [("photoz_source_model", PickleFile)]

    def run(self):
        shutil.copy(
            self.get_input("photoz_lens_model"),
            self.get_output("photoz_source_model"),
            )

