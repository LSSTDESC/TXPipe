import numpy as np
from .calibration_utils.names import META_VARIANTS, SCALAR_META_VARIANTS, metacal_variants, band_variants
from .data_types import ShearCatalog, CatalogSelection
from .base_stage import PipelineStage
from .calibration_utils.calibration_calculators import _DataWrapper
from ceci.config import StageParameter

class Cutter:
    def __init__(self, n, verbose=False):
        self.sel = np.ones(n, dtype=bool)
        self.verbose = verbose
        self.size = n

    def keep(self, sel, name):
        f1 = sel.sum() / self.size
        self.sel &= sel
        final_size = self.sel.sum()
        f2 = final_size / sel.size
        if self.verbose:
            print(f" - {name} retains {f1:.1%} and leaves {f2:.1%} cumulatively [ = {final_size:,}] afterwards")


class TXSimpleSelection(PipelineStage):
    """Compute non-redshift related shear catalog cuts.
    
    This can be used directly for simple shear catalogs, or
    subclasses can be used for any other type.
    """
    name = "TXSimpleSelection"
    inputs = [
        ("shear_catalog", ShearCatalog),
    ]
    outputs = [
        ("shear_selection", CatalogSelection)
    ]
    groups = ["shear"]
    columns = ["T", "s2n", "psf_T_mean", "flags"]
    config_options = {
        "chunk_rows": StageParameter(int, 10_000, msg="Number of rows to process at once"),
        "s2n_min": StageParameter(float, 10.0, msg="Minimum s/n for selection"),
        "T_min": StageParameter(float, 0.5, msg="Minimum T/Tpsf for selection"),
        "verbose": StageParameter(bool, False, msg="Whether to print out lots of information"),
    }

    def run(self):
        # make basic selections on all relevant groups
        # save to a CatalogSelection
        fin = self.open_input("shear_catalog")
        fout = self.open_output("shear_selection", parallel=True)

        for group in self.groups:
            g_in = fin[group]
            g_out = fout.create_group(group)
            n = g_in["ra"].size
            # maybe add compression here since can be very effective on this?
            n_sel = 0
            sel_out = g_out.create_dataset("selection", shape=(n,), dtype=bool) 
            for (s, e, data) in self.data_iterator(group):
                print(f"Rank {self.rank} processing rows {s:,} - {e:,}")
                sel = self.compute_selection(s, e, data)
                sel_out[s: e] = sel
                n_sel += sel.sum()
            if self.comm is not None:
                n_sel = self.comm.allreduce(n_sel)
            g_out.attrs["n_selected"] = n_sel

        fin.close()
        fout.close()


    def data_iterator(self, group):
        chunk_rows = self.config["chunk_rows"]
        return self.iterate_hdf(
            "shear_catalog",
            group,
            self.columns,
            chunk_rows,
        )

        

    def compute_selection(self, start, end, data, return_cutter=False):
        """
        This baseline method just does simpler S/N, flag, and size cuts.

        
        """
        s2n_cut = self.config["s2n_min"]
        T_cut = self.config["T_min"]
        verbose = self.config["verbose"]
        n = data["s2n"].size
        cutter = Cutter(n, verbose=verbose)

        cutter.keep(data["flags"] == 0, "Flag")
        cutter.keep(data["s2n"] > s2n_cut, "S/N")
        cutter.keep(data["T"] / data["psf_T_mean"] > T_cut, "T/Tpsf")

        if return_cutter:
            return cutter
        else:
            return cutter.sel


class TXMetaDetectSelection(TXSimpleSelection):
    name = "TXMetaDetectSelection"
    groups = ["shear/" + v for v in META_VARIANTS]


class TXScalarMetaDetectSelection(TXSimpleSelection):
    name = "TXScalarMetaDetectSelection"
    groups = ["shear/" + v for v in SCALAR_META_VARIANTS]


class TXMetacalSelection(TXSimpleSelection):
    name = "TXMetacalSelection"

    def run(self):
        # Shadow the class variable TXSimpleSelection.columns with one
        # specific to this instance
        self.columns = metacal_variants("T", "s2n") + ["psf_T_mean"]
        with self.open_input("shear_catalog", wrapper=True) as f:
            if "flags_1p" in f.file["shear"]:
                self.columns += metacal_variants("flags")
            else:
                self.columns += ["flags"]

        # make basic selections on all relevant groups
        # save to a CatalogSelection
        fin = self.open_input("shear_catalog")
        fout = self.open_output("shear_selection", parallel=True)
        g_out = fout.create_group("shear")

        n = fin["shear/ra"].size
        g_out.create_dataset("selection", shape=(n,), dtype=bool)
        for variant in META_VARIANTS[1:]:
            g_out.create_dataset(f"selection_{variant}", shape=(n,), dtype=bool) 

        for (s, e, data) in self.data_iterator("shear"):
            for variant in META_VARIANTS:
                n_sel = 0
                if variant == "ns":
                    sel_out = g_out["selection"]
                else:
                    sel_out = g_out[f"selection_{variant}"]
                # maybe add compression here since can be very effective on this?
                print(f"Rank {self.rank} processing rows {s:,} - {e:,}")
                data_v = _DataWrapper(data, suffix="_"+variant)
                sel = self.compute_selection(s, e, data_v)
                sel_out[s: e] = sel

            if self.comm is not None:
                    n_sel = self.comm.allreduce(n_sel)
                    g_out.attrs["n_selected"] = n_sel

        fin.close()
        fout.close()

class TXAnacalSelection(TXSimpleSelection):
    """Compute non-redshift shear catalog cuts for AnaCal catalogs.

    The cuts are evaluated on the baseline catalog and on each of the four
    ±γ shifted variants (columns with suffix _1p, _1m, _2p, _2m precomputed
    by TXIngestAnacal), so that the calibration stage can compute the
    selection response. Cuts on shear-independent quantities (mask fraction,
    PSF, flux error) have no variants, and _DataWrapper falls back to the
    baseline column for them.
    """
    name = "TXAnacalSelection"
    config_options = TXSimpleSelection.config_options | {
        "bands": StageParameter(list, ["r", "i", "z"], msg="Bands to apply magnitude, colour, PSF and flux-error cuts in"),
        "s2n_min": StageParameter(float, 5.0, msg="Minimum S/N (i-band flux / flux_err)"),
        "T_min": StageParameter(float, 0.1, msg="Minimum band-combined size (m00 + m20) / m00 on the fpfs1 moments"),
        "mask_threshold": StageParameter(float, 0.04, msg="Keep objects with n_mask_base < mask_threshold (values >= 1 apply no cut)"),
        "emax": StageParameter(float, 0.5, msg="Band-combined shape cut |e| < emax, applied as esq < emax**2"),
        "g_hi_cut": StageParameter(float, 50.0, msg="Upper g-mag cut"),
        "r_hi_cut": StageParameter(float, 50.0, msg="Upper r-mag cut"),
        "i_hi_cut": StageParameter(float, 50.0, msg="Upper i-mag cut"),
        "z_hi_cut": StageParameter(float, 50.0, msg="Upper z-mag cut"),
        "gr_cut": StageParameter(float, 50.0, msg="|g - r| colour cut"),
        "ri_cut": StageParameter(float, 50.0, msg="|r - i| colour cut"),
        "iz_cut": StageParameter(float, 50.0, msg="|i - z| colour cut"),
        "psf_g_cut": StageParameter(float, 1.0, msg="Per-band PSF ellipticity cut |psf_g1|,|psf_g2| < psf_g_cut (1 = no cut)"),
        "flux_err_column": StageParameter(str, "lsst_{band}_flux_gauss2_err", msg="Per-band flux-error column used by {b}_flux_err_cut"),
        "g_flux_err_cut": StageParameter(float, 1e30, msg="Upper g flux-error cut [nJy]"),
        "r_flux_err_cut": StageParameter(float, 1e30, msg="Upper r flux-error cut [nJy]"),
        "i_flux_err_cut": StageParameter(float, 1e30, msg="Upper i flux-error cut [nJy]"),
        "z_flux_err_cut": StageParameter(float, 1e30, msg="Upper z flux-error cut [nJy]"),
    }

    def run(self):
        # Shadow the class variable TXSimpleSelection.columns with one
        # specific to this instance, since it depends on the configuration
        bands = self.config["bands"]
        self.columns = metacal_variants("s2n", "m00", "m20", "esq", *[f"mag_{b}" for b in bands])
        self.columns += ["n_mask_base"]
        # PSF and flux-error columns are only read when their cuts are
        # switched on, so catalogs without them still run with the defaults
        for b in bands:
            if self.config["psf_g_cut"] < 1.0:
                self.columns += [f"psf_g1_{b}", f"psf_g2_{b}"]
            if self.config[f"{b}_flux_err_cut"] < 1e30:
                self.columns.append(self.config["flux_err_column"].format(band=b))

        fin = self.open_input("shear_catalog")
        n = fin["shear/ra"].size
        fin.close()

        fout = self.open_output("shear_selection", parallel=True)
        g_out = fout.create_group("shear")

        # "selection" is the baseline; the others are the ±γ variants
        suffixes = ["", "_1p", "_1m", "_2p", "_2m"]
        for suffix in suffixes:
            g_out.create_dataset(f"selection{suffix}", shape=(n,), dtype=bool)
        n_sel = np.zeros(len(suffixes), dtype=np.int64)

        for (s, e, data) in self.data_iterator("shear"):
            print(f"Rank {self.rank} processing rows {s:,} - {e:,}")
            for i, suffix in enumerate(suffixes):
                sel = self.compute_selection(s, e, _DataWrapper(data, suffix=suffix))
                g_out[f"selection{suffix}"][s:e] = sel
                n_sel[i] += sel.sum()

        if self.comm is not None:
            n_sel = self.comm.allreduce(n_sel)
        for suffix, count in zip(suffixes, n_sel):
            g_out.attrs[f"n_selected{suffix}"] = count

        fout.close()

    def compute_selection(self, start, end, data):
        config = self.config
        bands = config["bands"]
        cutter = Cutter(data["s2n"].size, verbose=config["verbose"])

        cutter.keep(data["n_mask_base"] < config["mask_threshold"], "mask fraction")
        cutter.keep(data["s2n"] > config["s2n_min"], "S/N")

        # Band-combined size cut on fpfs1 moments
        m00 = data["m00"]
        cutter.keep((m00 + data["m20"]) / m00 > config["T_min"], "size")
        cutter.keep(data["esq"] < config["emax"] ** 2, "|e|")

        for b in bands:
            cutter.keep(data[f"mag_{b}"] < config[f"{b}_hi_cut"], f"mag_{b}")

        for b1, b2 in (("g", "r"), ("r", "i"), ("i", "z")):
            if b1 in bands and b2 in bands:
                colour = np.abs(data[f"mag_{b1}"] - data[f"mag_{b2}"])
                cutter.keep(colour < config[f"{b1}{b2}_cut"], f"{b1} - {b2}")

        psf_g_cut = config["psf_g_cut"]
        if psf_g_cut < 1.0:
            for b in bands:
                # NaN PSF (failed HSM fit) compares False, i.e. is rejected
                psf_ok = (np.abs(data[f"psf_g1_{b}"]) < psf_g_cut) & (np.abs(data[f"psf_g2_{b}"]) < psf_g_cut)
                cutter.keep(psf_ok, f"PSF g {b}")

        for b in bands:
            cut = config[f"{b}_flux_err_cut"]
            if cut < 1e30:
                cutter.keep(data[config["flux_err_column"].format(band=b)] < cut, f"flux err {b}")

        return cutter.sel

class TXLensfitSelection(TXSimpleSelection):
    name = "TXLensfitSelection"


class TXHSCSelection(TXSimpleSelection):
    name = "TXHSCSelection"
    config_options = TXSimpleSelection.config_options | {
        "max_shear_cut": StageParameter(float, 0.0, msg="Maximum shear value for object selection")
    }
    columns = TXSimpleSelection.columns + ["g1", "g2"]

    def compute_selection(self, start, end, data):
        """
        Add an additional cut to the parent class, if specified, on the max shear.
        HSM DP0.2 catalogs seem to contain occasional very large shears that skew peaks.
        This removes those. This is only really for testing.
        """
        cutter = super().compute_selection(start, end, data, return_cutter=True)

        shear_cut = self.config["max_shear_cut"]
        if shear_cut:
            g = np.sqrt(data["g1"] ** 2 + data["g2"] ** 2)
            cutter.keep(g < shear_cut, "max_shear")
        return cutter.sel


class TXDP2RubinMetadetectSelection(TXMetaDetectSelection):
    name = "TXDP2RubinMetadetectSelection"

    columns = [
        "ra", "is_primary", "mfrac", "gauss_snr", "gauss_T", "gauss_psfReconvolved_T", "g1", "g2", "weight",
    ] + band_variants("griz", "mag", shear_catalog_type="simple")

    config_options = TXMetaDetectSelection.config_options | {
        "mfrac_max": StageParameter(float, default=0.1, msg="Maximum mask fraction for object selection"),
        "s2n_min": StageParameter(float, default=10.0, msg="Minimum gauss_snr for object selection"),
        "T_max": StageParameter(float, default=20.0, msg="Maximum gauss_T for object selection"),
        "Tratio_min": StageParameter(float, default=0.5, msg="Minimum gauss_T / gauss_psfReconvolved_T for object selection"),
        "i_min": StageParameter(float, default=18.2, msg="Minimum i-band magnitude"),
        "i_max": StageParameter(float, default=23.5, msg="Maximum i-band magnitude"),
        "r_min": StageParameter(float, default=15.2, msg="Minimum r-band magnitude"),
        "g_min": StageParameter(float, default=15.2, msg="Minimum g-band magnitude"),
        "rmi_min": StageParameter(float, default=-1.5, msg="Minimum r-i color for object selection"),
        "rmi_max": StageParameter(float, default=4.0, msg="Maximum r-i color for object selection"),
        "imz_min": StageParameter(float, default=-1.5, msg="Minimum i-z color for object selection"),
        "imz_max": StageParameter(float, default=4.0, msg="Maximum i-z color for object selection"),
    }

    def compute_selection(self, start, end, data):
        # shortcut if no data
        if data["gauss_T"].size == 0:
            return np.zeros(0, dtype=bool)

        config = self.config

        mfrac_max = config["mfrac_max"]
        s2n_min = config["s2n_min"]
        T_max = config["T_max"]
        Tratio_min = config["Tratio_min"]
        i_min = config["i_min"]
        i_max = config["i_max"]
        r_min = config["r_min"]
        g_min = config["g_min"]
        rmi_min = config["rmi_min"]
        rmi_max = config["rmi_max"]
        imz_min = config["imz_min"]
        imz_max = config["imz_max"]

        T_ratio = data["gauss_T"] / data["gauss_psfReconvolved_T"]
        rmi = data["mag_r"] - data["mag_i"]
        imz = data["mag_i"] - data["mag_z"]

        cutter = Cutter(data["ra"].size, verbose=config["verbose"])

        # Basic cuts
        cutter.keep(data["is_primary"] == True, "is_primary")
        cutter.keep(data["mfrac"] < mfrac_max, f"mfrac < {mfrac_max}")
        cutter.keep(np.isfinite(data["g1"]), "g1 finite" )
        cutter.keep(np.isfinite(data["g2"]), "g2 finite" )
        cutter.keep(np.isfinite(data["weight"]), "weight finite" )
        
        # Galaxy cuts
        cutter.keep(data["gauss_snr"] > s2n_min, f"S/N > {s2n_min}")
        cutter.keep(data["gauss_T"] < T_max, f"T < {T_max}")
        cutter.keep(T_ratio > Tratio_min, f"T / T psf > {Tratio_min}")

        # Magnitude cuts
        cutter.keep((data["mag_i"] > i_min) & (data["mag_i"] < i_max), f"i in [{i_min}, {i_max}]")
        cutter.keep(data["mag_r"] > r_min, f"r > {r_min}")
        cutter.keep(data["mag_g"] > g_min, f"g > {g_min}")

        # Sanity cuts on color
        cutter.keep((rmi > rmi_min) & (rmi < rmi_max), f"r-i color in [{rmi_min}, {rmi_max}]")
        cutter.keep((imz > imz_min) & (imz < imz_max), f"i-z color in [{imz_min}, {imz_max}]")

        return cutter.sel


class TXDP2DESCSelection(TXScalarMetaDetectSelection):
    name = "TXDP2DESCSelection"
    columns = ["T", "s2n", "g1", "g2", "ra", "dec", "weight", "psf_T_mean", "flags",  "is_primary", "mfrac", "psfrec_g1", "psfrec_g2", "rmi", "imz", "g_flags"] + band_variants("riz", "mag", shear_catalog_type="simple")

    config_options =TXScalarMetaDetectSelection.config_options | {
        "max_psf_g": StageParameter(float, default=0.05, msg="Maximum PSF ellipticity for object selection"),
        "rmi_min": StageParameter(float, default=-2.0, msg="Minimum r-i color for object selection"),
        "rmi_max": StageParameter(float, default=3.0, msg="Maximum r-i color for object selection"),
        "imz_min": StageParameter(float, default=-2.0, msg="Minimum i-z color for object selection"),
        "imz_max": StageParameter(float, default=3.0, msg="Maximum i-z color for object selection"),
        "T_max": StageParameter(float, default=20.0, msg="T threshold for object selection"),
        "Tratio_min": StageParameter(float, default=0.5, msg="Minimum T ratio for object selection"),
        "Tratio_max": StageParameter(float, default=20.0, msg="Maximum T ratio for object selection"),
        "s2n_min": StageParameter(float, default=10.0, msg="Minimum signal-to-noise ratio for object selection"),
        "s2n_max": StageParameter(float, default=100.0, msg="Maximum signal-to-noise ratio for object selection"),
        "mfrac_max": StageParameter(float, default=0.1, msg="Maximum mask fraction for object selection"),
        "i_min":StageParameter(float, default=18.2, msg="Minimum i-band magnitude"),
        "i_max":StageParameter(float, default=23.5, msg="Maximum i-band magnitude"),
        "r_min":StageParameter(float, default=15.2, msg="Minimum g-band magnitude"),
        "r_max":StageParameter(float, default=30., msg="Maximum g-band magnitude"),
    }
    def compute_selection(self, start, end, data):
        # shortcut if no data
        if data["T"].size == 0:
            return np.zeros(0, dtype=bool)

        config = self.config

        max_psf_g = config["max_psf_g"]
        rmi_min = config["rmi_min"]
        rmi_max = config["rmi_max"]
        imz_min = config["imz_min"]
        imz_max = config["imz_max"]
        T_max = config["T_max"]
        Tratio_min = config["Tratio_min"]
        Tratio_max = config["Tratio_max"]
        s2n_min = config["s2n_min"]
        s2n_max = config["s2n_max"]
        mfrac_max = config["mfrac_max"]
        i_min = config["i_min"]
        i_max = config["i_max"]
        r_min = config["r_min"]
        r_max = config["r_max"]

        T_ratio = data["T"] / data["psf_T_mean"]

        cutter = Cutter(data["ra"].size, verbose=config["verbose"])
        cutter.keep(data["is_primary"] == True, "is_primary")
        # Basic cuts
        cutter.keep(data["flags"] == 0, "flags == 0")
        cutter.keep(data["mfrac"] < mfrac_max, f"mfrac < {mfrac_max}")
        
        # Image quality cuts
        psfrec_gmax = np.maximum(data['psfrec_g1'], data['psfrec_g2'])
        cutter.keep(np.abs(psfrec_gmax) < max_psf_g, f"psf |ellipticity| max < {max_psf_g}")

        # Sanity cuts on color and size; somewhat arbitrary at this stage
        cutter.keep((data["rmi"] > rmi_min) & (data["rmi"] < rmi_max), f"r-i color in [{rmi_min}, {rmi_max}]")
        cutter.keep((data["imz"] > imz_min) & (data["imz"] < imz_max), f"i-z color in [{imz_min}, {imz_max}]")
        cutter.keep(data["T"] < T_max, f"T < {T_max}")
        cutter.keep(T_ratio < Tratio_max, f"T / T psf < {Tratio_max}")
        cutter.keep(data["s2n"] < s2n_max, f"S/N < {s2n_max}")

        # Magnitude cuts
        cutter.keep(data["mag_i"] > i_min, f"i > {i_min}")
        cutter.keep(data["mag_r"] > r_min, f"r > {r_min}")
        cutter.keep(data["mag_i"] < i_max, f"i < {i_max}")
        cutter.keep(data["mag_r"] < r_max, f"r < {r_max}")


        # Good ellipticities
        cutter.keep(data["g_flags"] == 0, "ellipticity flags = 0")

        # Galaxy cuts
        cutter.keep(T_ratio > Tratio_min, f"T / T psf > {Tratio_min}")
        cutter.keep(data["s2n"] > s2n_min, f"S/N > {s2n_min}")

        return cutter.sel


