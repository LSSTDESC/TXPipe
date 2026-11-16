"""
Pipeline modules for the 3x2pt (TX) project.

"""

# Make sure any stages you want to use in a pipeline
# are imported here.
from .base_stage import PipelineStage
from .selection import (
    TXMeanLensSelector,
    TXDESISelector,
    TXDESIMockSelector,
    TXSimpleSelection,
    TXMetaDetectSelection,
    TXScalarMetaDetectSelection,
    TXMetacalSelection,
    TXLensfitSelection,
    TXHSCSelection,
)
from .calibration import (
    TXShearCalibration,
    TXComputeShearCalibrationSimple,
    TXComputeShearCalibrationHSC,
    TXComputeShearCalibrationLensfit,
    TXComputeShearCalibrationMetacal,
    TXComputeShearCalibrationMetadetect,
    TXComputeShearCalibrationScalarMetadetect,
)
from .photoz import (
    TXPhotozStack,
    TXPhotozPlot,
    TXTruePhotozStack,
    TXTrainRandomForestTomography,
    TXComputeSourceTomography,
    TXApplyRandomForestSourceTomography,
    TXSimpleRedshiftSourceTomography,
    PZRailSummarize,
    PZRealizationsPlot,
    TXParqetToHDF,
)
from .maps import (
    TXSourceMaps,
    TXLensMaps,
    TXPSFMaps,
    TXFlagMaps,
    TXDepthMaps,
    TXBrightObjectMaps,
    TXSourceNoiseMaps,
    TXLensNoiseMaps,
    TXNoiseMapsJax,
    TXMapPlots,
    TXSimpleMask,
    TXSimpleMaskFrac,
    TXConvergenceMaps,
    TXMapCorrelations,
)
from .randoms import TXRandomCat
from .twopoint import (
    TXTwoPoint,
    TXTwoPointFourier,
    TXGammaTFieldCenters,
    TXTwoPointPlots,
    TXTwoPointPlotsFourier,
    TXTwoPointPlotsTheory,
    TXDeltaSigma,
    TXJackknifeCenters,
    TXSSIMagnification,
)
from .covariance import (
    TXFourierGaussianCovariance,
    TXRealGaussianCovariance,
    TXFourierNamasterCovariance,
    TXRealNamasterCovariance,
)
from .theory import TXTwoPointTheoryReal, TXTwoPointTheoryFourier
from .blinding import TXNullBlinding, TXSmokescreenBlinding
from .diagnostics import (
    TXSourceDiagnosticPlots,
    TXLensDiagnosticPlots,
    TXExposureInfo,
    TXPSFDiagnostics,
    TXRoweStatistics,
    TXFocalPlanePlot,
)
from .systematics import TXLSSWeights
from .metadata import TXTracerMetadata
from .ingest import *
from .systematics_twopoint_fourier import TXSystematicsTwoPointFourier

# We no longer import all the extensions automatically here to avoid
# some dependency problems when running under the LSST environment on NERSC.
# You can still import them explicitly in your pipeline scripts by doing:
# import txpipe.extensions
# or you can add txpipe.extensions to the "modules" section in the a pipeline YML file.
