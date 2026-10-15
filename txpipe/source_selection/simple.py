from .base import TXSourceSelectorBase
from ..shear_calibration import band_variants, MockCalculator
import numpy as np


class TXSourceSelectorSimple(TXSourceSelectorBase):
    """
    Source selection and tomography for mock catalogs that do not
    require any calibration.
    """

    name = "TXSourceSelectorSimple"
    config_options = TXSourceSelectorBase.config_options.copy()

    def setup_response_calculators(self, nbin_source):
        calculators = [MockCalculator(self.select) for i in range(nbin_source)]
        calculators.append(MockCalculator(self.select_any_bin))
        return calculators
