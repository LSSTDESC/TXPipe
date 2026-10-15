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


class TXBasicSelection(PipelineStage):
    """Compute non-redshift related shear catalog cuts.
    
    This can be used directly for simple shear catalogs, or
    subclasses can be used for any other type.
    """
    name = "TXBasicSelection"
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
            sel_out = g_out.create_dataset("selection", shape=(n,), dtype=bool) 
            for (s, e, data) in self.data_iterator(group):
                print(f"Rank {self.rank} processing rows {s:,} - {e:,}")
                sel = self.compute_selection(s, e, data)
                sel_out[s: e] = sel

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
        cutter.keep(data["T"] / data[f"psf_T_mean"] > T_cut, "T/Tpsf")

        if return_cutter:
            return cutter
        else:
            return cutter.sel


class TXMetaDetectSelection(TXBasicSelection):
    name = "TXMetaDetectSelection"
    groups = ["shear/" + v for v in META_VARIANTS]


class TXScalarMetaDetectSelection(TXBasicSelection):
    name = "TXScalarMetaDetectSelection"
    groups = ["shear/" + v for v in SCALAR_META_VARIANTS]


class TXMetacalSelection(TXBasicSelection):
    name = "TXMetacalSelection"

    def run(self):
        # Shadow the class variable TXBasicSelection.columns with one
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
        g_out.create_dataset(f"selection", shape=(n,), dtype=bool) 
        for variant in META_VARIANTS[1:]:
            g_out.create_dataset(f"selection_{variant}", shape=(n,), dtype=bool) 

        for (s, e, data) in self.data_iterator("shear"):
            for variant in META_VARIANTS:
                if variant == "ns":
                    sel_out = g_out[f"selection"]
                else:
                    sel_out = g_out[f"selection_{variant}"]
                # maybe add compression here since can be very effective on this?
                print(f"Rank {self.rank} processing rows {s:,} - {e:,}")
                data_v = _DataWrapper(data, suffix="_"+variant)
                sel = self.compute_selection(s, e, data_v)
                sel_out[s: e] = sel

        fin.close()
        fout.close()



class TXLensfitSelection(TXBasicSelection):
    name = "TXLensfitSelection"


class TXHSCSelection(TXBasicSelection):
    name = "TXHSCSelection"
    config_options = TXBasicSelection.config_options | {
        "max_shear_cut": StageParameter(float, 0.0, msg="Maximum shear value for object selection")
    }
    columns = TXBasicSelection.columns + ["g1", "g2"]

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
            cutter.keep(g < shear_cut)
        return cutter.sel


class TXDP2Selection(TXMetaDetectSelection):
    name = "TXDP2Selection"

    config_options = {
        "chunk_rows": StageParameter(int, 10_000, msg="Number of rows to process at once"),
        "s2n_min": StageParameter(float, 10.0, msg="Minimum s/n for selection"),
        "T_min": StageParameter(float, 0.5, msg="Minimum T/Tpsf for selection"),
        "verbose": StageParameter(bool, False, msg="Whether to print out lots of information"),
    }

    columns = ["T", "s2n", "psf_T_mean", "flags", "mag_g", "mag_r", "mag_i", "mag_z", "gauss_T", "mfrac", "g1", "g2", "weight", "is_primary"]

    def compute_selection(self, start, end, data):

        # These are the same cuts as the base class but repeated here
        # for clarity.
        s2n_cut = self.config["s2n_min"]
        T_cut = self.config["T_min"]
        verbose = self.config["verbose"]
        n = data["s2n"].size
        cutter = Cutter(n, verbose=verbose)

        cutter.keep(data["flags"] == 0, "Flag")
        cutter.keep(data["s2n"] > s2n_cut, "S/N")
        cutter.keep(data["T"] / data[f"psf_T_mean"] > T_cut, "T/Tpsf")


        # --- metadetect-specific cuts go here ---
        mag_g_cut = config["mag_g_cut"]
        mag_r_cut = config["mag_r_cut"]
        mag_i_cut = config["mag_i_cut"]
        mag_z_cut = config["mag_z_cut"]
        gmr_cut = config["gr_cut"]
        rmi_cut = config["ri_cut"]
        imz_cut = config["iz_cut"]
        T_max = config['T_max']
        mfrac_cut = config['mfrac_cut']
        bands = config['bands']

        cutter.keep(data[f"mag_g"] < config[f"mag_g_cut"], "mag_g")
        cutter.keep(data[f"mag_r"] < config[f"mag_r_cut"], "mag_r")
        cutter.keep(data[f"mag_i"] < config[f"mag_i_cut"], "mag_i")
        cutter.keep(data[f"mag_z"] < config[f"mag_z_cut"], "mag_z")

        if "g" in bands and "r" in bands:
            cutter.keep(np.abs(data["mag_g"] - data["mag_r"]) < gmr_cut, "g - r")
        if "r" in bands and "i" in bands:
            cutter.keep(np.abs(data["mag_r"] - data["mag_i"]) < r-i_cut, "r - i")
        if "i" in bands and "z" in bands:
            cutter.keep(np.abs(data["mag_i"] - data["mag_z"]) < imz_cut, "i - z")
        # We should also have some crazy color cuts and magnitude cuts which should come from PZ group
        cutter.keep((data["gauss_T"] < T_max), "gauss_T" )
        cutter.keep((data['mfrac'] < mfrac_cut), "mfrac" )
        cutter.keep(np.isfinite(data["g1"]), "g1 finite" )
        cutter.keep(np.isfinite(data["g2"]), "g2 finite" )
        cutter.keep(np.isfinite(data["weight"]), "weight finite" )
        cutter.keep(data["is_primary"], "is_primary" )

        return cutter.sel


class TXDP2DESCSelection(TXScalarMetaDetectSelection):
    name = "TXDP2DESCSelection"
    columns = ["T", "s2n", "g1", "g2", "ra", "dec", "weight", "psf_T_mean", "flags",  "is_primary", "mfrac", "psfrec_g1", "psfrec_g2", "rmi", "imz", "g_flags"] + band_variants("riz", "mag")

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
    }
    def compute_selection(self, start, end, data):
        # shortcut if no data
        if data["T"].size == 0:
            return np.zeros(0, dtype=bool)

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

        T_ratio = data["T"] / data["psf_T_mean"]

        cutter = Cutter(data["ra"].size, verbose=config["verbose"])
        cutter.keep(data["is_primary"] == True, "is_primary")
        # Basic cuts
        cutter.keep(data["flags"] == 0, "flags")
        cutter.keep(data["mfrac"] < mfrac_max, "mfrac")

        # Image quality cuts
        psfrec_gmax = np.maximum(data['psfrec_g1'], data['psfrec_g2'])
        cutter.keep(np.abs(psfrec_gmax) < max_psf_g, "psf ellipticity")

        # Sanity cuts on color and size; somewhat arbitrary at this stage
        cutter.keep((data["rmi"] > rmi_min) & (data["rmi"] < rmi_max), "r-i color")
        cutter.keep((data["imz"] > imz_min) & (data["imz"] < imz_max), "i-z color")
        cutter.keep(data["T"] < T_max, "T maximum")
        cutter.keep(T_ratio < Tratio_max, "T ratio maximum")
        cutter.keep(data["s2n"] < s2n_max, "S/N maximum")

        # Good ellipticities
        cutter.keep(data["g_flags"] == 0, "ellipticity flags")

        # Galaxy cuts
        cutter.keep(T_ratio > Tratio_min, "T ratio minimum")
        cutter.keep(data["s2n"] > s2n_min, "S/N minimum")

        # It's not obvious whether, when we are using an external selection to choose
        # tomographic bins, our "non-tomographic" sample should include objects that
        # are not selected into any valid tomographic bin. But right now our design
        # requires that, because the designation for non-tomographic objects is just
        # any selected tomographic bin. If we want a separate sample where non-tomographic
        # also means we ignore the RAIL SOM selection then that should be a separate run
        # of TXPipe right now.
        cutter.keep(data["zbin"] >= 0, "Any z bin")
        return cutter.sel


