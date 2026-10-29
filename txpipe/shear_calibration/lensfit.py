from .base import TXComputeShearCalibrationBase
from ..calibration_utils import LensfitCalculator, band_variants
import numpy as np
from ceci.config import StageParameter

class TXComputeShearCalibrationLensfit(TXComputeShearCalibrationBase):
    """
    Source selection and tomography for lensfit catalogs

    This selector class is for Lensfit catalogs like those used in KIDS.

    It is a simpler calibration scheme than the above two, and does not involve
    variant catalogs, just taking the mean of a value for one catalog.
    """

    name = "TXComputeShearCalibrationLensfit"

    # add one option to the base class configuration
    config_options = {
        **TXComputeShearCalibrationBase.config_options,
        "input_m_is_weighted": StageParameter(
            bool, required=True, msg="Whether the input m values are already weighted"
        ),
        "dec_cut": StageParameter(bool, True, msg="Whether to apply a declination cut"),
    }

    def setup_response_calculators(self, nbin_source):
        calculators = [
            LensfitCalculator(self.select, input_m_is_weighted=self.config["input_m_is_weighted"])
            for i in range(nbin_source)
        ]
        calculators.append(LensfitCalculator(self.select_any_bin, input_m_is_weighted=self.config["input_m_is_weighted"]))
        return calculators

    def setup_output(self, nbin_source):
        # This call to the super-class method defined above sets up most of the output
        # here, so the rest of this method only does things specific to this
        # calibration scheme
        outfile = super().setup_output(nbin_source)
        n = outfile["tomography/bin"].size
        group = outfile.create_group("response")
        group.create_dataset("K", (nbin_source,), dtype="f")
        group.create_dataset("C_N", (nbin_source, 2), dtype="f")
        group.create_dataset("C_S", (nbin_source, 2), dtype="f")
        group.create_dataset("K_2d", (1,), dtype="f")
        group.create_dataset("C_2d_N", (2,), dtype="f")
        group.create_dataset("C_2d_S", (2,), dtype="f")

        return outfile

