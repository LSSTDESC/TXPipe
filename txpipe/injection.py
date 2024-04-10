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


class TXSSIIngest(PipelineStage):
    """
    class for
    """

    name = "TXSSIIngest"

    inputs = [
        ("injection_catalog", HDFFile),
        ("ssi_photometry_catalog", HDFFile),
    ]

    outputs = [
        ("matched_ssi_photometry_catalog", HDFFile),
    ]

    config_options = {
        "chunk_rows": 10000,
        "match_radius": 0.1,
    }

    def run(self):
        """
        Run the analysis for this stage.
        """
        # prep the catalogs for reading

        # loop through chunks of the ssi photometry
        # catalog and match to the injections

        # output the matched catalog with as much SSI metadata
        # stored as possible 


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





