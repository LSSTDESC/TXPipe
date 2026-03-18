
from .base import TXSourceSelectorBase, BinStats
from ..utils.calibrators import LensfitCalibrator
from ..utils.calibration_tools import LensfitCalculator, band_variants
import numpy as np
from ceci.config import StageParameter




class TXSourceSelectorLensfit(TXSourceSelectorBase):
    """
    Source selection and tomography for lensfit catalogs

    This selector class is for Lensfit catalogs like those used in KIDS.

    It is a simpler calibration scheme than the above two, and does not involve
    variant catalogs, just taking the mean of a value for one catalog.
    """

    name = "TXSourceSelectorLensfit"

    # add one option to the base class configuration
    config_options = {
        **TXSourceSelectorBase.config_options,
        "input_m_is_weighted": StageParameter(
            bool, required=True, msg="Whether the input m values are already weighted"
        ),
        "dec_cut": StageParameter(bool, True, msg="Whether to apply a declination cut"),
    }

    def data_iterator(self):
        chunk_rows = self.config["chunk_rows"]
        bands = self.config["bands"]
        shear_cols = [
            "dec",
            "psf_T_mean",
            "weight",
            "flags",
            "T",
            "s2n",
            "g1",
            "g2",
            "weight",
            "m",
        ]
        shear_cols += band_variants(bands, "mag", "mag_err", shear_catalog_type="lensfit")
        if self.config["input_pz"]:
            shear_cols += ["mean_z"]
        elif self.config["true_z"]:
            shear_cols += ["redshift_true"]
        return self.iterate_hdf("shear_catalog", "shear", shear_cols, chunk_rows)

    def setup_response_calculators(self, nbin_source):
        calculators = [
            LensfitCalculator(self.select, input_m_is_weighted=self.config["input_m_is_weighted"])
            for i in range(nbin_source)
        ]
        calculators.append(LensfitCalculator(self.select_2d, input_m_is_weighted=self.config["input_m_is_weighted"]))
        return calculators

    def setup_output(self):
        # This call to the super-class method defined above sets up most of the output
        # here, so the rest of this method only does things specific to this
        # calibration scheme
        outfile = super().setup_output()
        n = outfile["tomography/bin"].size
        nbin_source = outfile["counts/counts"].size
        group = outfile.create_group("response")
        group.create_dataset("K", (nbin_source,), dtype="f")
        group.create_dataset("C_N", (nbin_source, 2), dtype="f")
        group.create_dataset("C_S", (nbin_source, 2), dtype="f")
        group.create_dataset("K_2d", (1,), dtype="f")
        group.create_dataset("C_2d_N", (2), dtype="f")
        group.create_dataset("C_2d_S", (2), dtype="f")

        return outfile

    def compute_output_stats(self, calculator, mean, variance):
        K, C_N, C_S, N, Neff = calculator.collect(self.comm, allgather=True)
        calibrator = LensfitCalibrator(K, C_N, C_S)
        mean_e = (C_N + C_S) / 2
        sigma_e = np.sqrt((0.5 * (variance[0] + variance[1]))) / (1 + K)

        return BinStats(N, Neff, mean_e, sigma_e, calibrator)
