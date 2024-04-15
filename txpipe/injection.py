from .base_stage import PipelineStage
from .data_types import (
    YamlFile,
    HDFFile,
    FitsFile,
)
from .utils import LensNumberDensityStats, Splitter, rename_iterated
from .binning import build_tomographic_classifier, apply_classifier
import numpy as np
import warnings


class TXSSIMagnification(PipelineStage):
    """
    class for
    """

    name = "TXSSIMagnification"

    inputs = [
        ("binned_lens_catalog_nomag", HDFFile),
        ("binned_lens_catalog_mag", HDFFile),
    ]

    outputs = [
        ("magnification", HDFFile),
    ]

    config_options = {
        "chunk_rows": 10000,
    }

    def run(self):
        """
        Run the analysis for this stage.
        """

        # load the catalogs

        # get/estimate the magnification applied to each catalog
        # could put this as optional config item in TXSSIIngest

        # compute number of objects in each bin in each catalog
        # + number of shared objects

        # compute magnification coeff (b_mag, C_sample, alpha, etc)





