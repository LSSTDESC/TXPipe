from .base_stage import PipelineStage
from .data_types import PhotozPDFFile, ShearCatalog, YamlFile, HDFFile
import numpy as np

class TXSelfCalbrationQCalc(PipelineStage):
    name = 'TXSCQ'
    inputs = [
        ('shear_photoz_stack', HDFFile),
        ('shear_tomography_catalog', TomographyCatalog),
        ('photoz_pdfs', PhotozPDFFile),
    ]
    outputs = [
        ('Q_SCIA', TextFile)
    ]
    config_options = {}

    def run(self):
        











    # Shamelessly stolen from twopoint! removed any reference to lenses,
    # since we don't need them
    def _read_nbin_from_tomography(self):
        tomo = self.open_input('shear_tomography_catalog')
        d = dict(tomo['tomography'].attrs)
        tomo.close()
        nbin_source = d['nbin_source']
        source_list = range(nbin_source)
        return source_list