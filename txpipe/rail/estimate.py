from ..base_stage import PipelineStage
from ..data_types import PhotozPDFFile
import shutil


class PZRailEstimateSourceFromLens(PipelineStage):
    """
    Make a source redshifts file by copying lens redshifts

    In cases where source and lens come from the same base sample
    we can simply copy the computed PDFs.
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
    Make a lens  redshifts file by copying source redshifts

    In cases where source and lens come from the same base sample
    we can simply copy the computed PDFs.
    """

    name = "PZRailEstimateLensFromSource"

    inputs = [("source_photoz_pdfs", PhotozPDFFile)]
    outputs = [("lens_photoz_pdfs", PhotozPDFFile)]

    def run(self):
        shutil.copy(
            self.get_input("source_photoz_pdfs"),
            self.get_output("lens_photoz_pdfs"),
        )
