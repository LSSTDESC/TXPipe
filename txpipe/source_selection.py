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
