from .base_stage import PipelineStage
from .utils import choose_pixelization
from .data_types import (
    HDFFile,
    ShearCatalog,
    TextFile,
    MapsFile,
    FileCollection,
    FiducialCosmology,
    TomographyCatalog,
)
import glob
import time

class TXLogNormalGlass(PipelineStage):
    """
    Uses GLASS to generate a simulated catalog from lognormal fields
    GLASS citation: 
    https://ui.adsabs.harvard.edu/abs/2023OJAp....6E..11T
    """

    name = "TXLogNormalGlass"
    parallel = False
    inputs = [ 
        ("mask", MapsFile),
        ("lens_photoz_stack", HDFFile),
        ("fiducial_cosmology", FiducialCosmology),
    ]

    outputs = [
        ("photometry_catalog", HDFFile),
        ("lens_tomography_catalog_unweighted", TomographyCatalog), 
        #TO DO: add shear maps to output
    ]

    config_options = {
        "num_dens": None,
    }

    def run(self):
        import glass.shells
        import glass.fields
        import glass.points
        import glass.galaxies


