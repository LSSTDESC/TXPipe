from .base import TXSourceSelectorBase
from .base import select_weak_lensing_sample, select_tomographic_weak_lensing_sample
from ..utils.calibration_tools import band_variants, MockCalculator
from ..utils.calibrators import NullCalibrator
import numpy as np
from .base import BinStats


class TXSourceSelectorSimple(TXSourceSelectorBase):
    """
    Source selection and tomography for mock catalogs that do not
    require any calibration.
    """

    name = "TXSourceSelectorSimple"
    config_options = TXSourceSelectorBase.config_options.copy()

    def data_iterator(self):
        chunk_rows = self.config["chunk_rows"]
        bands = self.config["bands"]

        # Select columns we need.
        shear_cols = [
            "psf_T_mean",
            "weight",
            "flags",
            "T",
            "s2n",
            "g1",
            "g2",
            "weight",
        ]
        if self.config["input_pz"]:
            shear_cols += ["mean_z"]
        elif self.config["true_z"]:
            shear_cols += ["redshift_true"]
        else:
            shear_cols += band_variants(bands, "mag", "mag_err", shear_catalog_type="hsc")

        # Iterate using parent class method
        return self.iterate_hdf("shear_catalog", "shear", shear_cols, chunk_rows)

    def setup_response_calculators(self, nbin_source):
        calculators = [MockCalculator(select_tomographic_weak_lensing_sample) for i in range(nbin_source)]
        calculators.append(MockCalculator(select_weak_lensing_sample))
        return calculators

    def compute_output_stats(self, calculator, mean, variance):
        # Collate calibration values
        N, Neff = calculator.collect(self.comm, allgather=True)
        calibrator = NullCalibrator(mean)
        mean_e = mean.copy()
        sigma_e = np.sqrt(0.5 * (variance[0] + variance[1]))
        return BinStats(N, Neff, mean_e, sigma_e, calibrator)
