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
from .photoz_stack import TXPhotozSourceStack, TXPhotozLensStack
from .random_cats import TXRandomCat
from .twopoint_fourier import TXTwoPointFourier
from .twopoint import TXTwoPoint
from .blinding import TXBlinding
from .input_cats import TXCosmoDC2Mock
from .covariance import TXFourierGaussianCovariance, TXRealGaussianCovariance
from .metacal_gcr_input import TXMetacalGCRInput
from .diagnostics import TXSourceDiagnosticPlots, TXLensDiagnosticPlots
from .exposure_info import TXExposureInfo
from .psf_diagnostics import TXPSFDiagnostics, TXRoweStatistics
from .noise_maps import TXSourceNoiseMaps, TXLensNoiseMaps, TXNoiseMapsJax
from .ingest_redmagic import TXIngestRedmagic
from .maps import TXMainMaps
from .auxiliary_maps import TXAuxiliarySourceMaps, TXAuxiliaryLensMaps
from .map_plots import TXMapPlots
from .masks import TXSimpleMask
from .metadata import TXTracerMetadata
from .convergence import TXConvergenceMaps
from .map_correlations import TXMapCorrelations
from .rail import PZRailSummarize, PZRealizationsPlot, TXParqetToHDF
from .theory import TXTwoPointTheoryReal, TXTwoPointTheoryFourier
from .jackknife import TXJackknifeCenters
from .twopoint_null_tests import TXGammaTFieldCenters
from .twopoint_plots import TXTwoPointPlots, TXTwoPointPlotsFourier, TXTwoPointPlotsTheory
from .calibrate import TXShearCalibration
from .lssweights import TXLSSweights

# Here are the stages that mostly will be used for other projects
# such as the self-calibration of Intrinsic alignment.
from .extensions.twopoint_scia import TXSelfCalibrationIA
from .extensions.clmm import TXTwoPointRLens
from .covariance_nmt import TXFourierNamasterCovariance, TXRealNamasterCovariance
