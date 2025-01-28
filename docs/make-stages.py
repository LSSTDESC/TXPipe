import sys
sys.path.append('..')
import txpipe
import yaml
import collections
import ceci

sections = yaml.safe_load("""
Ingestion:
    blurb: |
        These stages import data into TXPipe input formats, or generate mock data from
        simpler input catalogs.
    stages:
        - TXCosmoDC2Mock
        - TXBuzzardMock
        - TXGaussianSimsMock
        - TXIngestRedmagic
        - TXExposureInfo
        - TXIngestStars
        - TXMetacalGCRInput
        - TXIngestDataPreview02
        - TXSimpleMock
        - TXMockTruthPZ
        - TXLogNormalGlass

Photo-z:
    blurb: |
        These stages deal with photo-z PDF training and estimation
    stages:
        - TXPhotozStack
        - TXTruePhotozStack
        - PZRailTrainSource
        - PZRailTrainLens
        - PZRailTrainLensFromSource
        - PZRailTrainSourceFromLens
        - PZRailEstimateSource
        - PZRailEstimateLens
        - PZRailEstimateSourceFromLens
        - PZRailEstimateLensFromSource


Selection:
    blurb: |
        These stages deal with selection objects and assigning them to tomographic
        bins.
    stages:
        - TXSourceSelector
        - TXSourceSelectorSimple
        - TXSourceSelectorMetacal
        - TXSourceSelectorMetadetect
        - TXSourceSelectorLensfit
        - TXSourceSelectorHSC
        - TXBaseLensSelector
        - TXTruthLensSelector
        - TXMeanLensSelector
        - TXModeLensSelector
        - TXRandomForestLensSelector


Calibration and Splitting:
    blurb: |
        These stages deal with calibrating shear and splitting up catalogs into
        sub-catalogs.
    stages:
        - TXShearCalibration
        - TXStarCatalogSplitter
        - TXLensCatalogSplitter
        - TXExternalLensCatalogSplitter
        - TXTruthLensCatalogSplitter
        - TXTruthLensCatalogSplitterWeighted
        
Weights:
    blurb: |
        These stages deal with weighting the lens sample
    stages:
        - TXLSSWeights
        - TXLSSWeightsLinBinned
        - TXLSSWeightsLinPix
        - TXLSSWeightsUnit
    

Maps:
    blurb: |
        These stages deal with making different kinds of maps for analysis and
        plotting.
    stages:
        - TXBaseMaps
        - TXSourceMaps
        - TXLensMaps
        - TXExternalLensMaps
        - TXDensityMaps
        - TXSourceNoiseMaps
        - TXLensNoiseMaps
        - TXExternalLensNoiseMaps
        - TXNoiseMapsJax
        - TXConvergenceMaps
        - TXMapCorrelations
        - TXSimpleMask
        - TXSimpleMaskSource
        - TXSimpleMaskFrac
        - TXAuxiliarySourceMaps
        - TXAuxiliaryLensMaps
        - TXUniformDepthMap

Ensemble Photo-z:
    blurb: |
        These stages compute ensemble redshift histograms n(z) for bins
    stages:
        - TXPhotozSourceStack
        - TXPhotozLensStack
        - TXSourceTrueNumberDensity
        - TXLensTrueNumberDensity
        - PZRailSummarize
        


Two-Point:
    blurb: |
        These stages deal with measuring or predicting two-point statistics.
    stages:
        - TXJackknifeCenters
        - TXJackknifeCentersSource
        - TXTwoPointFourier
        - TXTwoPoint
        - TXRandomCat
        - TXSubsampleRandoms
        - TXTwoPointTheoryReal
        - TXTwoPointTheoryFourier
        - TXTwoPointPixel
        - TXTwoPointPixelExtCross
        - TXTwoPointRLens


Covariance:
    blurb: |
        These stages compute covariances of measurements
    stages:
        - TXFourierGaussianCovariance
        - TXRealGaussianCovariance
        - TXFourierTJPCovariance
        - TXFourierNamasterCovariance
        - TXRealNamasterCovariance


Blinding:
    blurb: |
        These stages deal with blinding measurements
    stages:
        - TXBlinding
        - TXNullBlinding


Plots:
    blurb: |
        These stages make plots out TXPipe output data
    stages:
        - TXTwoPointPlots
        - TXTwoPointPlotsFourier
        - TXConvergenceMapPlots
        - TXMapPlots
        - PZRealizationsPlot
        - TXTwoPointPlotsTheory
        - TXPhotozPlot


Diagnostics:
    blurb: |
        These stages compute and/or plot diagnostics of catalogs or other data
    stages:
        - TXDiagnosticQuantiles
        - TXSourceDiagnosticPlots
        - TXLensDiagnosticPlots
        - TXPSFDiagnostics
        - TXPSFMomentCorr
        - TXRoweStatistics
        - TXTauStatistics
        - TXGalaxyStarShear
        - TXGalaxyStarDensity
        - TXBrighterFatterPlot
        - TXGammaTFieldCenters
        - TXGammaTStars
        - TXGammaTRandoms
        - TXApertureMass
        - TXFocalPlanePlot

Source Injection:
    blurb: |
        These stages ingest and use synthetic source injection information
    stages:
        - TXIngestSSIGCR
        - TXMatchSSI
        - TXIngestSSIMatched
        - TXIngestSSIMatchedDESBalrog
        - TXSSIMagnification
        - TXAuxiliarySSIMaps
        - TXMapPlotsSSI
        - TXIngestSSI
        - TXIngestSSIDESBalrog
        - TXIngestSSIDetectionDESBalrog


Extensions:
    blurb: |
        These stages are written for TXPipe extension projects.
    stages:
        - TXSelfCalibrationIA
        - CLClusterBinningRedshiftRichness
        - CLClusterShearCatalogs
        - CLClusterEnsembleProfiles

New and Miscellaneous:
    blurb: |
        These stages either don't fit into a category above or have not yet been
        assigned to one.
    stages:
        - TXTracerMetadata
        - TXParqetToHDF

""")

stages = {}
files = {}

for name, section_data in sections.items():
    stages_in_section = section_data["stages"]
    for stage in stages_in_section:
        stages[stage] = name

def get_doc_line(stage):
    # find the first non-empty doc line, if there is one.
    try:
        doc_lines = [s.strip() for s in stage.__doc__.split("\n")]
        doc_lines = [d for d in doc_lines if d]
        doc = doc_lines[0]
    except (AttributeError, IndexError):
        doc = ""
    if len(doc) > 100:
        doc = doc[:96] + " ..."
    return doc

def get_name(cls):
    module = cls.__module__
    if module is None or module == str.__class__.__module__:
        return cls.__name__
    return module + '.' + cls.__name__

qual_names = collections.defaultdict(list)

def describe_config(cls):
    lines = ["            <UL>"]
    for name, val in cls.config_options.items():
        if isinstance(val, ceci.config.StageParameter):
            if val.required:
                req = True
            if val.dtype is None:
                dt = "any"
            else:
                dt = val.dtype.__name__
                default = val.default
            help = val._help
        else:
            help = ""
            if isinstance(val, type):
                dt = val.__name__
                req = True
            else:
                dt = type(val).__name__
                req = False
                default = val
        if req:
            lines.append(f"            <LI><strong>{name}</strong>: ({dt}) Required. {help}</LI>")
        else:
            lines.append(f"            <LI><strong>{name}</strong>: ({dt}) Default={default}. {help}</LI>")
    lines.append("            </UL>")
    return "\n".join(lines)
        


for class_name, (cls, path) in txpipe.PipelineStage.pipeline_stages.items():
    if class_name == "BaseStageDoNotRunDirectly":
        continue
    if class_name not in stages:
        print("Warning - update the section list for ", class_name)

    qual_name = get_name(cls)
    section = stages.get(class_name, "New and Miscellaneous")
    doc = get_doc_line(cls)
    qual_names[section].append((qual_name, doc, cls))


for name, section_data in sections.items():
    with open(f"src/stages/{name}.rst", "w") as f:
        f.write(name + "\n")
        f.write(("=" * len(name)) + "\n\n")
        f.write(section_data["blurb"] + "\n")
        files[name] = f

        for qual_name, doc, _ in qual_names[name]:
            f.write(f"* :py:class:`~{qual_name}` - {doc}\n\n")
        f.write("\n")

        for qual_name, _, cls in qual_names[name]:
            if cls.dask_parallel:
                par_text = "Yes - Dask"
            elif cls.parallel:
                par_text = "Yes - MPI"
            else:
                par_text = "No - Serial"

            config_text = describe_config(cls)



            f.write(f"""
.. autotxclass:: {qual_name}
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

{config_text}


""")



with open ("src/stages.rst", "w") as f:
    f.write(""".. Autogenerated by make-stages.py

TXPipe Stage Listing
====================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

""")
    f.write("   stages/base\n")
    for s, sd in sections.items():
        f.write("   stages/" + s + "\n")
    f.write("\n")