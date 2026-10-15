from .base import TXSourceSelectorBase
from .metadetect import TXSourceSelectorMetadetect
from ..calibration_utils import (
    scalar_metadetect_variants,
    ScalarMetaDetectCalculator,
    band_variants,
    SCALAR_META_VARIANTS,
)
import numpy as np
from ceci.config import StageParameter


class TXSourceSelectorScalarMetadetect(TXSourceSelectorMetadetect):
    """
    Source selection and tomography for scalar metadetect catalogs.

    This subclass selects for scalar MetaDetect catalogs, which keep only the
    ns, 1p, and 1m variants and use a scalar response model.
    """

    name = "TXSourceSelectorScalarMetadetect"
    variants = SCALAR_META_VARIANTS

    def setup_response_calculators(self, nbin_source):
        delta_gamma = self.config["delta_gamma"]
        calculators = [
            ScalarMetaDetectCalculator(self.select, delta_gamma)
            for i in range(nbin_source)
        ]
        calculators.append(ScalarMetaDetectCalculator(self.select_any_bin, delta_gamma))
        return calculators


    def setup_response_output(self, outfile, nbin_source):
        group = outfile.create_group("response")
        group.create_dataset("R", (nbin_source,), dtype="f")
        group.create_dataset("R_2d", (1,), dtype="f")
