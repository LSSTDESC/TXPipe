from .scalar_metadetect import TXSourceSelectorScalarMetadetect
from .base import select_weak_lensing_sample, TXSourceSelectorBase
from ..shear_calibration import metadetect_variants, MetaDetectCalculator, band_variants, META_VARIANTS, scalar_metadetect_variants, ScalarMetaDetectCalculator
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

    config_options = TXSourceSelectorScalarMetadetect.config_options | dp2_cut_options

    def data_iterator(self):
        # As above, this is where we work out which columns we need.
        chunk_rows = self.config["chunk_rows"]
        bands = self.config["bands"]
        cat_type = "scalar_metadetect"
        self.config["T_col"] = "T"
        with self.open_input("shear_catalog", wrapper=True) as f:
            self.config["bands"] = f.get_bands()

        # Core quantities we need
        shear_cols = scalar_metadetect_variants("T", "s2n", "g1", "g2", "ra", "dec", "weight", "psf_T_mean", "flags",  "is_primary", "mfrac", "psfrec_g1", "psfrec_g2", "rmi", "imz", "g_flags")

        # Magnitudes and errors
        shear_cols += band_variants(bands, "mag", "mag_err", shear_catalog_type=cat_type)

        # We need truth shears and/or PZ point-estimates for each shear too
        if self.config["input_pz"]:
            shear_cols += scalar_metadetect_variants("mean_z")
        elif self.config["true_z"]:
            shear_cols += scalar_metadetect_variants("redshift_true")

        # This is a parent ceci.PipelineStage method.
        # It returns an iterator we loop through.
        # The "longest=True" option means that the iterator will
        # continue looping even when some of the columns have been exhausted, which is 
        # what we want here since the different shear variants have different lengths.
        # The calibration calculation needs to deal with this.
        it = self.iterate_hdf("shear_catalog", "shear", shear_cols, chunk_rows, longest=True)
        return it

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
    
    # Basic cuts
    sel = np.ones(data["ra"].size, dtype=bool)
    sel &= (data["is_primary"] == True)
    sel &= (data["flags"] == 0)
    sel &= data["mfrac"] < mfrac_max

    # Image quality cuts
    psfrec_gmax = np.maximum(data['psfrec_g1'], data['psfrec_g2'])
    sel &= np.abs(psfrec_gmax) < max_psf_g

    # Sanity cuts on color and size; somewhat arbitrary at this stage
    sel &= (data["rmi"] > rmi_min) & (data["rmi"] < rmi_max)
    sel &= (data["imz"] > imz_min) & (data["imz"] < imz_max)
    sel &= (data["T"] < T_max)
    sel &= (T_ratio < Tratio_max)
    sel &= (data["s2n"] < s2n_max)

    # Good ellipticities
    sel &= (data["g_flags"] == 0)

    # Galaxy cuts
    sel &= (T_ratio > Tratio_min)
    sel &= (data["s2n"] > s2n_min)

    return sel


def select_tomographic_weak_lensing_sample_metadetect_desc_dp2(data, config, bin_index):
    """
    Tomographic counterpart to select_weak_lensing_sample_metadetect_dp2, in the
    same way that select_tomographic_weak_lensing_sample relates to
    select_weak_lensing_sample.
    """
    zbin = data["zbin"]
    verbose = config["verbose"]

    sel = select_weak_lensing_sample_metadetect_desc_dp2(data, config, calling_from_select=True)
    sel &= (zbin == bin_index)
    f4 = sel.sum() / sel.size

    if verbose:
        print(f"{f4:.2%} z for bin {bin_index}")
        print("total tomo", sel.sum())

    return sel


