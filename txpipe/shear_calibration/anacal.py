from .base import TXComputeShearCalibrationBase
from ..calibration_utils import metacal_variants, AnaCalCalculator
from ..data_types import HDFFile
import numpy as np
from ceci.config import StageParameter


class TXComputeShearCalibrationAnacal(TXComputeShearCalibrationBase):
    """
    Source selection and tomography for anacal catalogs
    
    """

    name = "TXComputeShearCalibrationAnacal"

    config_options = {
        **TXComputeShearCalibrationBase.config_options,
        "delta_gamma": StageParameter(
            float,
            required=True,
            msg="Delta gamma value for metacal response calculation",
        ),
    }

    def column_groups(self):
        return [
            {
                # e1, e2 are e_meas = wsel * e_raw; the rest feed R_shape and R_detect
                "shear_catalog": [
                    "e1", "e2", "weight",
                    "wsel", "e1_raw", "e2_raw",
                    "de1_dg1", "de2_dg2",
                    "weight_dg1", "weight_dg2",
                ],
                "shear_bin_assignments": metacal_variants("bin"),
                "shear_selection": metacal_variants("selection"),
            }
        ]

    def setup_output(self, nbin_source):
        """
        AnaCal has a single scalar response per bin, plus one for the 2D sample.
        """
        outfile = super().setup_output(nbin_source)
        group = outfile.create_group("response")
        group.create_dataset("R", (nbin_source,), dtype="f")
        group.create_dataset("R_2d", (1,), dtype="f")
        return outfile

    def setup_response_calculators(self, nbin_source):
        delta_gamma = self.config["delta_gamma"]
        calculators = [AnaCalCalculator(self.select, delta_gamma) for i in range(nbin_source)]
        calculators.append(AnaCalCalculator(self.select_any_bin, delta_gamma))
        return calculators
