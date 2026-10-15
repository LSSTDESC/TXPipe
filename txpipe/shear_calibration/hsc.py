from .base import TXSourceSelectorBase
from ..calibration_utils import band_variants, HSCCalculator
import numpy as np
from ceci.config import StageParameter


class TXSourceSelectorHSC(TXSourceSelectorBase):
    """
    Source selection and tomography for HSC catalogs

    This subclass is for selecting objects on catalogs of the form made by HSC.

    This scheme is quite similar to the one used by lensfit. The main difference is in
    the per-object response.

    TODO: The HSC calibrator is currently broken, and will crash when it gets
    to compute_output_stats
    """

    name = "TXSourceSelectorHSC"
    config_options = TXSourceSelectorBase.config_options.copy()
    config_options["max_shear_cut"] = StageParameter(float, 0.0, msg="Maximum shear value for object selection")


    def setup_output(self, nbin_source):
        # This call to the super-class method defined above sets up most of the output
        # here, so the rest of this method only does things specific to this
        # calibration scheme
        outfile = super().setup_output(nbin_source)
        n = outfile["tomography/bin"].size
        group = outfile.create_group("response")

        # There is a single scalar per-object value for this scheme
        group.create_dataset("R", (n,), dtype="f")

        # and a set of additive and multiplicative factors.
        # The K and R values are degenerate.
        group.create_dataset("K", (nbin_source,), dtype="f")
        group.create_dataset("C", (nbin_source, 2), dtype="f")
        group.create_dataset("R_mean", (nbin_source,), dtype="f")
        group.create_dataset("K_2d", (1,), dtype="f")
        group.create_dataset("C_2d", (2), dtype="f")
        group.create_dataset("R_mean_2d", (1,), dtype="f")
        return outfile

    def write_tomography(self, outfile, start, end, source_bin, R):
        super().write_tomography(outfile, start, end, source_bin, R)
        group = outfile["response"]
        group["R"][start:end] = R

    def compute_per_object_response(self, data):
        w_tot = np.sum(data["weight"])
        R = np.array([1.0 - np.sum(data["weight"] * data["sigma_e"]) / w_tot] * len(data["weight"]))
        return R

    def setup_response_calculators(self, nbin_source):
        calculators = [HSCCalculator(self.select) for i in range(nbin_source)]
        calculators.append(HSCCalculator(self.select_any_bin))
        return calculators
    