"""
Pipeline modules for the 3x2pt (TX) project.

"""

# Make sure any stages you want to use in a pipeline
# are imported here.
from .base_stage import PipelineStage
from .source_selector import (
    TXSourceSelectorMetacal,
    TXSourceSelectorLensfit,
    TXSourceSelectorMetadetect,
)
from .lens_selector import TXMeanLensSelector
from .photoz_stack import TXPhotozStack, TXPhotozPlot, TXTruePhotozStack
from .random_cats import TXRandomCat
from .twopoint_fourier import TXTwoPointFourier
from .twopoint import TXTwoPoint
from .blinding import TXBlinding
from .covariance import TXFourierGaussianCovariance, TXRealGaussianCovariance
from .diagnostics import TXSourceDiagnosticPlots, TXLensDiagnosticPlots
from .exposure_info import TXExposureInfo
from .psf_diagnostics import TXPSFDiagnostics, TXRoweStatistics
from .noise_maps import TXSourceNoiseMaps, TXLensNoiseMaps, TXNoiseMapsJax
from .maps import TXSourceMaps, TXLensMaps
from .auxiliary_maps import TXAuxiliarySourceMaps, TXAuxiliaryLensMaps
from .map_plots import TXMapPlots
from .masks import TXSimpleMask, TXSimpleMaskFrac
from .metadata import TXTracerMetadata
from .convergence import TXConvergenceMaps
from .map_correlations import TXMapCorrelations
from .rail import PZRailSummarize, PZRealizationsPlot, TXParqetToHDF
from .theory import TXTwoPointTheoryReal, TXTwoPointTheoryFourier
from .jackknife import TXJackknifeCenters
from .twopoint_null_tests import TXGammaTFieldCenters
from .twopoint_plots import TXTwoPointPlots, TXTwoPointPlotsFourier, TXTwoPointPlotsTheory
from .calibrate import TXShearCalibration
from .ingest import *
from .spatial_diagnostics import TXFocalPlanePlot
from .lssweights import TXLSSWeights
from .simulation import TXLogNormalGlass
from .magnification import TXSSIMagnification
from .covariance_nmt import TXFourierNamasterCovariance, TXRealNamasterCovariance
from .delta_sigma import TXDeltaSigma
# We no longer import all the extensions automatically here to avoid
# some dependency problems when running under the LSST environment on NERSC.
# You can still import them explicitly in your pipeline scripts by doing:
# import txpipe.extensions
# or you can add txpipe.extensions to the "modules" section in the a pipeline YML file.
