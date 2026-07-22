from .base import TXSourceSelectorBase
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
            msg="Band-combined size cut (m00 + m20) / m00 > T_cut on the "
                "fpfs1 moments emitted by xlens.MergePipe.",
        ),
        # Band-combined shape magnitude cut |e| < emax on the ``esq``
        # column emitted by xlens' MergePipe (esq = e1**2 + e2**2 on the
        # WCS-corrected fpfs1 shape). The selector cuts on esq < emax**2
        # internally so this stays a familiar |e| threshold.
        "emax": StageParameter(
            float, 0.5,
            msg="Band-combined shape magnitude cut |e| < emax "
                "(applied as esq < emax**2 on the merge-stage esq column).",
        ),
        # Per-band AB mag upper bounds — one StageParameter per band,
        # matching the ``r_hi_cut`` / ``i_hi_cut`` convention in
        # lens_selector.py. Default 50 is above the xlens
        # smooth-truncation cap (MAG_CAP=40), so the default effectively
        # applies no cut; tighten in the yaml to gate on brightness.
        "g_hi_cut": StageParameter(float, 50.0, msg="Upper g-mag cut."),
        "r_hi_cut": StageParameter(float, 50.0, msg="Upper r-mag cut."),
        "i_hi_cut": StageParameter(float, 50.0, msg="Upper i-mag cut."),
        "z_hi_cut": StageParameter(float, 50.0, msg="Upper z-mag cut."),
    }

    def data_iterator(self):
        """
        This iterator returns chunks of data in dictionaries one by one.

        We call to a parent class method to do the main iteration; the work
        here is just choosing which columns to read.
        """

        bands = self.config["bands"]
        shear_cols = [
            "ra",
            "dec",
            "weight",
            "wsel",          # raw wsel — needed by R_shape numerator
            "mask_value",
            "e1",            # e_meas ≡ wsel · e_raw (pre-multiplied)
            "e2",
            "e1_raw",        # raw e — needed by R_detect numerator
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
            "weight_dg2",
        ]
        shear_cols += band_variants(
            bands, "mag", "mag_err", shear_catalog_type="anacal",
        )
        # Per-band shear-response derivatives — ``dmag_{b}_dg{c}`` is
        # what ``add_sheared_variant_columns`` uses to build the ±γ
        # variants of the per-band mag cut.  ``dmag_err_{b}_dg{c}`` is
        # carried alongside for downstream code that wants a shear
        # response on the magnitude error.
        for b in bands:
            shear_cols += [
                f"dmag_{b}_dg1", f"dmag_{b}_dg2",
                f"dmag_err_{b}_dg1", f"dmag_err_{b}_dg2",
            ]
        # Band-combined shape magnitude and its shear derivatives,
        # emitted by xlens.MergePipe — feeds the |e|<emax cut and its
        # ±γ variants.
        shear_cols += ["esq", "desq_dg1", "desq_dg2"]

        if self.config["input_pz"]:
            # Baseline photo-z + the four shifted variants used to derive
            # zbin_{1p,1m,2p,2m} for the selection-response calculation.
            shear_cols += [
                "mean_z", "mean_z_1p", "mean_z_1m",
                "mean_z_2p", "mean_z_2m",
            ]
        elif self.config["true_z"]:
            shear_cols += ["redshift_true"]

        chunk_rows = self.config["chunk_rows"]
        return self.iterate_hdf(
            "shear_catalog", "shear", shear_cols, chunk_rows,
        )

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
        cuts on: zbin (from mean_z), s2n (from ds2n_dg), size moments
        m00/m20 (from dm00_dg/dm20_dg), band-combined shape magnitude
        esq (from desq_dg{1,2}), and per-band mag_{b} (from
        mag_{b}_dg{1,2}).

        The additions land in ``pz_data`` — the base
        ``calculate_tomography`` will merge that dict into ``data`` before
        the calculator sees it, so ``_DataWrapper(data, "_1p")["s2n"]``
        transparently routes to the injected ``s2n_1p`` column.  This is
        why the calculator needs no per-cut special casing.

        zbin variants are only produced under ``input_pz`` (truth-z has no
        shear response); s2n / moment / esq / mag_{b} variants are always
        emitted since the selector applies those cuts in every mode.
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

        # esq variants: shape-magnitude cut |e|<emax at ±γ via desq_dg{1,2}
        # (xlens.MergePipe emits both esq and its shear derivatives).
        esq = shear_data["esq"]
        pz_data["esq_1p"] = esq + dg * shear_data["desq_dg1"]
        pz_data["esq_1m"] = esq - dg * shear_data["desq_dg1"]
        pz_data["esq_2p"] = esq + dg * shear_data["desq_dg2"]
        pz_data["esq_2m"] = esq - dg * shear_data["desq_dg2"]

        # Per-band mag variants for the mag_{b} < {b}_hi_cut cut at ±γ.
        for b in self.config["bands"]:
            mag = shear_data[f"mag_{b}"]
            pz_data[f"mag_{b}_1p"] = mag + dg * shear_data[f"dmag_{b}_dg1"]
            pz_data[f"mag_{b}_1m"] = mag - dg * shear_data[f"dmag_{b}_dg1"]
            pz_data[f"mag_{b}_2p"] = mag + dg * shear_data[f"dmag_{b}_dg2"]
            pz_data[f"mag_{b}_2m"] = mag - dg * shear_data[f"dmag_{b}_dg2"]

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
            AnaCalCalculator(
                select_anacal_tomographic_weak_lensing_sample, delta_gamma,
            )
            for i in range(nbin_source)
        ]
        calculators.append(
            AnaCalCalculator(
                select_anacal_weak_lensing_sample, delta_gamma,
            )
        )
        return calculators

    def write_tomography(self, outfile, start, end, source_bin, r):
        super().write_tomography(outfile, start, end, source_bin, r)
        group = outfile["response"]
        group["R"][start:end] = r


def select_anacal_weak_lensing_sample(
        data, config, calling_from_select=False):
    """Baseline AnaCal weak-lensing selection.

    Applies six cuts, each looked up through the ``_DataWrapper`` so the
    ±γ shifted variants (esq_1p, mag_g_1p, m00_1p, s2n_1p, zbin_1p, ...)
    are used automatically when the caller wraps ``data`` with a suffix:

    1. mask_value < mask_threshold (shear-independent, no variants).
    2. s2n > s2n_cut (variants from ds2n_dg{1,2}).
    3. Band-combined size cut (m00 + m20) / m00 > T_cut on the fpfs1
       moments (variants automatically use m00_{1p,...}/m20_{1p,...} =
       m + dg * dm_dg{c}).
    4. Band-combined shape magnitude esq < emax**2
       (variants from desq_dg{1,2}, both derived from the WCS-corrected
       fpfs1 shape by xlens.MergePipe).
    5. Per-band mag_{b} < {b}_hi_cut for every band in config.bands
       (variants from mag_{b}_dg{1,2}).
    6. zbin >= 0 (variants from shifted mean_z).
    """
    s2n_cut = config["s2n_cut"]
    t_cut = config["T_cut"]
    esq_max = config["emax"] ** 2
    verbose = config["verbose"]

    flag = data["mask_value"] < config["mask_threshold"]
    n0 = len(flag)
    sel = flag
    f1 = sel.sum() / n0

    sel &= data["s2n"] > s2n_cut
    f2 = sel.sum() / n0

    # Band-combined size cut on fpfs1 moments.
    m00 = data["m00"]
    m20 = data["m20"]
    sel &= (m00 + m20) / m00 > t_cut
    f3 = sel.sum() / n0

    sel &= data["esq"] < esq_max
    f4 = sel.sum() / n0

    for b in config["bands"]:
        sel &= data[f"mag_{b}"] < config[f"{b}_hi_cut"]
    f5 = sel.sum() / n0

    sel &= data["zbin"] >= 0
    f6 = sel.sum() / n0

    if verbose and calling_from_select:
        print(
            f"Tomo selection {f1:.2%} flag, {f2:.2%} SNR, {f3:.2%} size, "
            f"{f4:.2%} |e|, {f5:.2%} mag, ",
            end="",
        )
    elif verbose:
        print(
            f"2D selection {f1:.2%} flag, {f2:.2%} SNR, {f3:.2%} size, "
            f"{f4:.2%} |e|, {f5:.2%} mag, {f6:.2%} any z bin"
        )
        print("total 2D", sel.sum())
    return sel


def select_anacal_tomographic_weak_lensing_sample(data, config, bin_index):
    zbin = data["zbin"]
    sel = select_anacal_weak_lensing_sample(
        data, config, calling_from_select=True,
    )
    sel &= zbin == bin_index
    return sel
