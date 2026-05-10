from .base import TXSourceSelectorBase
from ..utils.calibration_tools import band_variants, HSCCalculator
from ..utils.calibrators import HSCCalibrator
import numpy as np
from .base import BinStats
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
            "m",
            "c1",
            "c2",
            "sigma_e",
        ]
        shear_cols += band_variants(bands, "mag", "mag_err", shear_catalog_type="hsc")
        if self.config["input_pz"]:
            shear_cols += ["mean_z"]
        elif self.config["true_z"]:
            shear_cols += ["redshift_true"]

        # Iterate using parent class method
        return self.iterate_hdf("shear_catalog", "shear", shear_cols, chunk_rows)

    def setup_output(self):
        # This call to the super-class method defined above sets up most of the output
        # here, so the rest of this method only does things specific to this
        # calibration scheme
        outfile = super().setup_output()
        n = outfile["tomography/bin"].size
        nbin_source = outfile["counts/counts"].size
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

    def compute_output_stats(self, calculator, mean, variance):
        R, K, N, Neff = calculator.collect(self.comm, allgather=True)
        calibrator = HSCCalibrator(R, K)
        sigma_e = np.sqrt((0.5 * (variance[0] + variance[1]))) / (1 + K)
        return BinStats(N, Neff, mean, sigma_e, calibrator)

    def setup_response_calculators(self, nbin_source):
        calculators = [HSCCalculator(self.select) for i in range(nbin_source)]
        calculators.append(HSCCalculator(self.select_2d))
        return calculators

    def select_2d(self, data, calling_from_select=False):
        """
        Add an additional cut to the parent class, if specified, on the max shear.
        HSM DP0.2 catalogs seem to contain occasional very large shears that skew peaks.
        This removes those. This is only really for testing.
        """
        sel = super().select_2d(data, calling_from_select=calling_from_select)
        shear_cut = self.config["max_shear_cut"]
        if shear_cut:
            g = np.sqrt(data["g1"] ** 2 + data["g2"] ** 2)
            cut = g < shear_cut
            p = 100 * (1 - (cut.sum() / cut.size))
            print(f" shear cut removes {p:.2f}% of objects")
            sel &= cut
            p = sel.sum() / sel.size * 100
            print(f" after shear cut retain {p:.2f}% of objects")
        return sel
