from .scalar_metadetect import TXSourceSelectorScalarMetadetect
from .base import select_weak_lensing_sample, TXSourceSelectorBase
from ..shear_calibration import metadetect_variants, MetaDetectCalculator, band_variants, SCALAR_META_VARIANTS, scalar_metadetect_variants, ScalarMetaDetectCalculator
from ..data_types import HDFFile
from ceci.config import StageParameter
import numpy as np


# max_psf_g: 0.05
# rmi_min: -2.0
# rmi_max: 3.0
# imz_min: -2.0
# imz_max: 3.0
# T_max: 20.0
# Tratio_min: 0.5
# Tratio_max: 20.
# s2n_min: 10.0
# s2n_max: 100.0
# mfrac_max: 0.1

dp2_cut_options = {
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



class TXSourceSelectorScalarMetadetectDP2(TXSourceSelectorScalarMetadetect):
    """
    Source selection and tomography for metadetect catalogs, with extra
    DP2-specific selection cuts.

    This is kept separate from TXSourceSelectorMetadetect so that we can
    iterate on the DP2-specific cuts here as more data comes in and we
    find out what new selections we need, without affecting the generic
    metadetect selector.
    """

    name = "TXSourceSelectorScalarMetadetectDP2"
    # It would be nice to do all these in a single RAIL run
    inputs = TXSourceSelectorScalarMetadetect.inputs + [
        ("tomography_assignments_ns", HDFFile),
        ("tomography_assignments_1m", HDFFile),
        ("tomography_assignments_1p", HDFFile),
    ]

    config_options = TXSourceSelectorScalarMetadetect.config_options | dp2_cut_options

    def make_tomographic_bin_chooser_function(self):
        # In this case the tomography has already been done by an external
        # selector, so we can just copy the redshift bin across to the new dict.
        def classifier(start, end, shear_data):
            output = {}
            for v in SCALAR_META_VARIANTS:
                print("copying ",v, shear_data[f"{v}/zbin"].size)
                output[f"{v}/zbin"] = shear_data[f"{v}/zbin"]
            return output
        return classifier

    def write_tomography(self, outfile, start, end, source_bin, per_object_response):
        # Write out each of the individual variants.
        # The basic "bin" column was set up to be the same as the 00 variant,
        # so we can just write to all of them.
        for i, v in enumerate(SCALAR_META_VARIANTS):
            col = source_bin[i]
            outfile[f"tomography/bin_{v}"][start:start+col.size] = col

        assert per_object_response is None, "MetaDetect does not produce per-object response values, only per-bin values, so this should be None"


    def data_iterator(self):
        # As above, this is where we work out which columns we need.
        chunk_rows = self.config["chunk_rows"]
        bands = self.config["bands"]
        cat_type = "scalar_metadetect"
        self.config["T_col"] = "T"
        with self.open_input("shear_catalog", wrapper=True) as f:
            self.config["bands"] = f.get_bands()

        # Core quantities we need
        shear_cols = ["T", "s2n", "g1", "g2", "ra", "dec", "weight", "psf_T_mean", "flags",  "is_primary", "mfrac", "psfrec_g1", "psfrec_g2", "rmi", "imz", "g_flags"]


        # Magnitudes and errors - we are going to deal with the variant in a minute so
        # right now we just want mag_r, mag_i, etc.
        shear_cols += band_variants(bands, "mag", "mag_err", shear_catalog_type="simple")


        # We need truth shears and/or PZ point-estimates for each shear too
        if self.config["input_pz"]:
            shear_cols.append("mean_z")
        elif self.config["true_z"]:
            shear_cols.append("redshift_true")
        
        # Think this is wrong - check
        tomo_cols = ["bhat_for_wide_data"]

        variant_shear_cols = scalar_metadetect_variants(*shear_cols)


        # The slightly odd way that this is all set up, which is descended from metacal,
        # means that the calibration calculators are going to expect dictionaries containing
        # all the different variants at once for a block of data. Even if in this case
        # most of them are empty.
        def generate_empty():
            out = {col: np.zeros(0, dtype=np.float64) for col in variant_shear_cols}
            for v in SCALAR_META_VARIANTS:
                out[f'{v}/zbin'] = np.zeros(0, dtype=np.int64)
            return out

        # Now we have all the shear columns for a single variant. We are going
        # to loop through the variants here one by one. This is a little different
        # for how we have done this before which was inherited from metacal where all
        # the lengths were the same
        for variant in SCALAR_META_VARIANTS:
            v_shear_cols = [f"{variant}/{col}" for col in shear_cols]
            it1 = self.iterate_hdf("shear_catalog", "shear", v_shear_cols, chunk_rows)
            it2 = self.iterate_hdf(f"tomography_assignments_{variant}", "/", tomo_cols, chunk_rows)
            for ((s, e, shear_data), (s1, e1, tomo_data)) in zip(it1, it2):
                if s != s1 or e != e1:
                    raise ValueError("Error in tomo/shear column relationship")
                data = generate_empty()
                data.update(shear_data)
                data[f'{variant}/zbin'] = tomo_data['bhat_for_wide_data']
                yield s, e, data

    def setup_response_calculators(self, nbin_source):
        delta_gamma = self.config["delta_gamma"]
        calculator_class = ScalarMetaDetectCalculator
        calculators = [
            ScalarMetaDetectCalculator(select_tomographic_weak_lensing_sample_metadetect_desc_dp2, delta_gamma)
            for i in range(nbin_source)
        ]
        calculators.append(ScalarMetaDetectCalculator(select_weak_lensing_sample_metadetect_desc_dp2, delta_gamma))
        return calculators


def select_weak_lensing_sample_metadetect_desc_dp2(data, config, calling_from_select=False):
    """
    Select weak lensing sample objects for metadetect catalogs.

    This starts from the general cuts in select_weak_lensing_sample (flags,
    size, S/N, mask fraction, tomographic bin) and then applies extra cuts
    that only make sense for metadetect catalogs. Add / remove cuts below
    and re-run to iterate.
    """
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
            print(f" - {name} cuts {f1:.1%} and leaves {f2:.1%} cumulatively cut [ = {final_size:,}] afterwards")


def select_tomographic_weak_lensing_sample_metadetect_desc_dp2(data, config, bin_index):
    """
    Tomographic counterpart to select_weak_lensing_sample_metadetect_dp2, in the
    same way that select_tomographic_weak_lensing_sample relates to
    select_weak_lensing_sample.
    """
    # shortcut if no data
    if data["zbin"].size == 0:
        return np.zeros(0, dtype=bool)
    zbin = data["zbin"]
    verbose = config["verbose"]

    sel = select_weak_lensing_sample_metadetect_desc_dp2(data, config, calling_from_select=True)
    sel &= (zbin == bin_index)
    f4 = sel.sum() / sel.size

    if verbose:
        print(f"{f4:.2%} z for bin {bin_index}, total tomo", sel.sum())

    return sel


