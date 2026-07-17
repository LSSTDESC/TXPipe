from .base import TXSourceSelectorBase
from .base import select_weak_lensing_sample, select_tomographic_weak_lensing_sample
from ..shear_calibration import AnaCalCalculator, band_variants
import numpy as np
from ceci.config import StageParameter


class TXSourceSelectorAnacal(TXSourceSelectorBase):
    """
    Source selection and tomography for AnaCal catalogs.

    This selector subclass is designed for anacal-type catalogs like those
    that DESC plans to produce for Rubin data.
    """

    name = "TXSourceSelectorAnaCal"

    config_options = {
        **TXSourceSelectorBase.config_options,
        "delta_gamma": StageParameter(
            float,
            required=True,
            msg="Delta gamma value for the AnaCal response calculation",
        ),
        "mask_threshold": StageParameter(
            float, 40,
            msg="mask threshold, for when to mask objects.",
        ),
        # AnaCal-specific defaults for the two base cuts.  These override
        # the base class's required-with-no-default declarations so a
        # minimal yaml block still runs; users can still set them
        # explicitly in the yaml if they want tighter/looser cuts.
        "s2n_cut": StageParameter(
            float, 5.0,
            msg="AnaCal S/N cut (i-band flux / flux_err).",
        ),
        "T_cut": StageParameter(
            float, 0.1,
            msg="AnaCal size cut: (m00 + m20) / m00 > T_cut.",
        ),
    }

    def data_iterator(self):
        """
        This iterator returns chunks of data in dictionaries one by one.

        We call to a parent class method to do the main iteration; the work here is
        just choosing which columns to read.
        """

        bands = self.config["bands"]
        shear_cols = ["ra",
                      "dec",
                      "weight",
                      "wsel",       # raw wsel — needed by R_shape numerator
                      "mask_value",
                      "e1",         # e_meas ≡ wsel · e_raw (pre-multiplied)
                      "e2",
                      "e1_raw",     # raw e — needed by R_detect numerator
                      "e2_raw",
                      "m00",
                      "m20",
                      "de1_dg1",
                      "de2_dg2",
                      "dm00_dg1",
                      "dm00_dg2",
                      "dm20_dg1",
                      "dm20_dg2",
                      "s2n",
                      "ds2n_dg1",
                      "ds2n_dg2",
                      "weight_dg1",
                      "weight_dg2"
                      ]
        shear_cols += band_variants(bands, "mag", "mag_err", shear_catalog_type="anacal")

        if self.config["input_pz"]:
            # Baseline photo-z + the four shifted variants used to derive
            # zbin_{1p,1m,2p,2m} for the selection-response calculation.
            shear_cols += ["mean_z", "mean_z_1p", "mean_z_1m", "mean_z_2p", "mean_z_2m"]
        elif self.config["true_z"]:
            shear_cols += ["redshift_true"]

        chunk_rows = self.config["chunk_rows"]
        return self.iterate_hdf("shear_catalog", "shear", shear_cols, chunk_rows)

    def apply_simple_redshift_cut(self, shear_data):
        """Override of the base hook that TXSourceSelectorBase.run() calls
        under ``input_pz`` / ``true_z``.  We keep the base name so
        polymorphism fires, but the real work — injecting ±γ shifted
        variants of every quantity the AnaCal selector cuts on — lives in
        ``add_sheared_variant_columns`` for a more descriptive name.
        """
        pz_data = super().apply_simple_redshift_cut(shear_data)
        self.add_sheared_variant_columns(pz_data, shear_data)
        return pz_data

    def add_sheared_variant_columns(self, pz_data, shear_data):
        """Inject ±γ shifted variants of every quantity the AnaCal selector
        cuts on: zbin (from mean_z), s2n (from ds2n_dg), and the AnaCal
        size moments m00, m20 (from dm00_dg, dm20_dg).

        The additions land in ``pz_data`` — the base
        ``calculate_tomography`` will merge that dict into ``data`` before
        the calculator sees it, so ``_DataWrapper(data, "_1p")["s2n"]``
        transparently routes to the injected ``s2n_1p`` column.  This is
        why the calculator needs no per-cut special casing.

        zbin variants are only produced under ``input_pz`` (truth-z has no
        shear response); s2n and moment variants are always emitted since
        the selector applies those cuts in every mode.
        """
        dg = self.config["delta_gamma"]

        # zbin_{1p,1m,2p,2m}: only under input_pz (from shifted photo-z).
        if self.config["input_pz"]:
            edges = self.config["source_zbin_edges"]
            for suf in ("1p", "1m", "2p", "2m"):
                zz = shear_data[f"mean_z_{suf}"]
                b = np.full(len(zz), -1, dtype=int)
                for zi in range(len(edges) - 1):
                    m = (zz >= edges[zi]) & (zz < edges[zi + 1])
                    b[m] = zi
                pz_data[f"zbin_{suf}"] = b

        # s2n_{1p,1m,2p,2m}: shifted S/N via ds2n_dg{1,2}.
        s2n = shear_data["s2n"]
        pz_data["s2n_1p"] = s2n + dg * shear_data["ds2n_dg1"]
        pz_data["s2n_1m"] = s2n - dg * shear_data["ds2n_dg1"]
        pz_data["s2n_2p"] = s2n + dg * shear_data["ds2n_dg2"]
        pz_data["s2n_2m"] = s2n - dg * shear_data["ds2n_dg2"]

        # m00 and m20 variants: needed for the size cut (m00+m20)/m00 > T_cut
        # under ±γ.  Kept identical in form to the metacal variant scheme.
        m00 = shear_data["m00"]
        m20 = shear_data["m20"]
        pz_data["m00_1p"] = m00 + dg * shear_data["dm00_dg1"]
        pz_data["m00_1m"] = m00 - dg * shear_data["dm00_dg1"]
        pz_data["m00_2p"] = m00 + dg * shear_data["dm00_dg2"]
        pz_data["m00_2m"] = m00 - dg * shear_data["dm00_dg2"]
        pz_data["m20_1p"] = m20 + dg * shear_data["dm20_dg1"]
        pz_data["m20_1m"] = m20 - dg * shear_data["dm20_dg1"]
        pz_data["m20_2p"] = m20 + dg * shear_data["dm20_dg2"]
        pz_data["m20_2m"] = m20 - dg * shear_data["dm20_dg2"]

    def setup_output(self):
        """
        Prepare the output columns for the response values generated by Anacal
        """
        outfile = super().setup_output()
        n = outfile["tomography/bin"].size
        group = outfile.create_group("response")
        group.create_dataset("R", (n, 1, 1), dtype="f")
        group.create_dataset("R_2d", (1,), dtype="f")
        return outfile

    def setup_response_calculators(self, nbin_source):
        delta_gamma = self.config["delta_gamma"]
        calculators = [
            AnaCalCalculator(select_anacal_tomographic_weak_lensing_sample, delta_gamma)
            for i in range(nbin_source)
        ]
        calculators.append(
            AnaCalCalculator(select_anacal_weak_lensing_sample, delta_gamma)
        )
        return calculators

    def write_tomography(self, outfile, start, end, source_bin, R):
        super().write_tomography(outfile, start, end, source_bin, R)
        group = outfile["response"]
        group["R"][start:end] = R

def select_anacal_weak_lensing_sample(data, config, calling_from_select=False):
    """Baseline AnaCal weak-lensing selection.

    Applies four cuts, each looked up through the ``_DataWrapper`` so the
    ±γ shifted variants (m00_1p, s2n_1p, zbin_1p, ...) are used
    automatically when the caller wraps ``data`` with a suffix:

    1. mask_value < mask_threshold (shear-independent, no variants).
    2. s2n > s2n_cut (variants from ds2n_dg{1,2}).
    3. AnaCal size cut (m00 + m20) / m00 > T_cut
       (variants from dm00_dg, dm20_dg).
    4. zbin >= 0 (variants from shifted mean_z).
    """
    s2n_cut = config["s2n_cut"]
    T_cut = config["T_cut"]
    verbose = config["verbose"]

    flag = data["mask_value"] < config["mask_threshold"]
    n0 = len(flag)
    sel = flag
    f1 = sel.sum() / n0

    sel &= data["s2n"] > s2n_cut
    f2 = sel.sum() / n0

    # AnaCal size cut, was previously duplicated in AnaCalCalculator.get_submask.
    m00 = data["m00"]
    m20 = data["m20"]
    sel &= (m00 + m20) / m00 > T_cut
    f3 = sel.sum() / n0

    sel &= data["zbin"] >= 0
    f4 = sel.sum() / n0

    if verbose and calling_from_select:
        print(f"Tomo selection {f1:.2%} flag, {f2:.2%} SNR, {f3:.2%} size, ", end="")
    elif verbose:
        print(f"2D selection {f1:.2%} flag, {f2:.2%} SNR, {f3:.2%} size, {f4:.2%} any z bin")
        print("total 2D", sel.sum())
    return sel

def select_anacal_tomographic_weak_lensing_sample(data, config, bin_index):
    zbin = data["zbin"]
    sel = select_anacal_weak_lensing_sample(data, config, calling_from_select=True)
    sel &= zbin == bin_index
    return sel
