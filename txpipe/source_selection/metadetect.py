from .base import TXSourceSelectorBase, BinStats
from ..utils.calibrators import MetaDetectCalibrator
from ..utils.calibration_tools import (
    metadetect_variants,
    MetaDetectCalculator,
    band_variants,
)
import numpy as np
from ceci.config import StageParameter
from ..utils import rename_iterated


class TXSourceSelectorMetadetect(TXSourceSelectorBase):
    """
    Source selection and tomography for metadetect catalogs

    This subclass selects for MetaDetect catalogs, which is expected to be used for
    Rubin data. It computes the selection bias due to object detection by repeating
    the detection process under different applied shears.

    As a consequence the different calibration columns have different lengths, since
    different objects are detected in each case.
    """

    name = "TXSourceSelectorMetadetect"

    # add one option to the base class configuration
    config_options = {
        **TXSourceSelectorBase.config_options,
        "delta_gamma": StageParameter(
            float,
            required=True,
            msg="Delta gamma value for metadetect response calculation",
        ),
    }

    def data_iterator(self):
        # As above, this is where we work out which columns we need.
        chunk_rows = self.config["chunk_rows"]
        bands = self.config["bands"]

        # Core quantities we need
        shear_cols = metadetect_variants("T", "s2n", "g1", "g2", "ra", "dec", "mcal_psf_T_mean", "weight", "flags")

        # Magnitudes and errors
        shear_cols += band_variants(bands, "mag", "mag_err", shear_catalog_type="metadetect")
        renames = {}

        # We need truth shears and/or PZ point-estimates for each shear too
        if self.config["input_pz"]:
            shear_cols += metadetect_variants("mean_z")
        elif self.config["true_z"]:
            shear_cols += ["00/redshift_true"]
            renames["00/redshift_true"] = "redshift_true"

        for prefix in ["00", "1p", "1m", "2p", "2m"]:
            renames[f"{prefix}/mcal_psf_T_mean"] = f"{prefix}/psf_T_mean"

        # This is a parent ceci.PipelineStage method.
        # It returns an iterator we loop through
        it = self.iterate_hdf("shear_catalog", "shear", shear_cols, chunk_rows, longest=True)
        return rename_iterated(it, renames)

    def setup_response_calculators(self, nbin_source):
        delta_gamma = self.config["delta_gamma"]
        calculators = [MetaDetectCalculator(self.select, delta_gamma) for i in range(nbin_source)]
        calculators.append(MetaDetectCalculator(self.select_2d, delta_gamma))
        return calculators

    def apply_simple_redshift_cut(self, data):
        # If we have the truth pz then we just need to do the binning once,
        # as in the parent class
        if self.config["true_z"]:
            return super().apply_simple_redshift_cut(data)

        # Otherwise we have to do it once for each variant
        pz_data = {}
        variants = ["00/", "1p/", "2p/", "1m/", "2m/"]
        for v in variants:
            zz = data[f"{v}mean_z"]

            pz_data_v = np.zeros(len(zz), dtype=int) - 1
            for zi in range(len(self.config["source_zbin_edges"]) - 1):
                mask_zbin = (zz >= self.config["source_zbin_edges"][zi]) & (
                    zz < self.config["source_zbin_edges"][zi + 1]
                )
                pz_data_v[mask_zbin] = zi

            pz_data[f"{v}zbin"] = pz_data_v

        return pz_data

    def setup_output(self):
        """
        MetaDetect outputs do not include per-object calibration values,
        only the per-bin values.
        """
        # This call to the super-class method defined above sets up most of the output
        # here, so the rest of this method only does things specific to this
        # calibration scheme
        outfile = super().setup_output()
        n = outfile["tomography/bin"].size
        nbin_source = outfile["counts/counts"].size
        group = outfile.create_group("response")

        # Per-bin 2x2 calibration matrix
        group.create_dataset("R", (nbin_source, 2, 2), dtype="f")
        # Global calibration matrix
        group.create_dataset("R_2d", (2, 2), dtype="f")
        return outfile

    def compute_output_stats(self, calculator, mean, variance):
        # Collate calibration values
        R, N, Neff = calculator.collect(self.comm, allgather=True)
        calibrator = MetaDetectCalibrator(R, mean, mu_is_calibrated=False)
        mean_e = calibrator.mu.copy()

        # Apply to the variances to get sigma_e
        P = np.diag(np.linalg.inv(R @ R))
        sigma_e = np.sqrt(0.5 * P @ variance)

        # Like metacal, N_eff = N for metadetect
        return BinStats(N, Neff, mean_e, sigma_e, calibrator)
