from .metadetect import TXSourceSelectorMetadetect
from .base import select_weak_lensing_sample, TXSourceSelectorBase
from ..shear_calibration import metadetect_variants, MetaDetectCalculator, band_variants, META_VARIANTS
from ceci.config import StageParameter
import numpy as np

class TXSourceSelectorMetadetectDP2(TXSourceSelectorMetadetect):
    """
    Source selection and tomography for metadetect catalogs, with extra
    DP2-specific selection cuts.

    This is kept separate from TXSourceSelectorMetadetect so that we can
    iterate on the DP2-specific cuts here as more data comes in and we
    find out what new selections we need, without affecting the generic
    metadetect selector.
    """

    name = "TXSourceSelectorMetadetectDP2"

    config_options = {
        **TXSourceSelectorMetadetect.config_options,
        "mag_g_cut": StageParameter(float, required=True, msg="Magnitude cut threshold for object selection"),
        "mag_r_cut": StageParameter(float, required=True, msg="Magnitude cut threshold for object selection"),
        "mag_i_cut": StageParameter(float, required=True, msg="Magnitude cut threshold for object selection"),
        "mag_z_cut": StageParameter(float, required=True, msg="Magnitude cut threshold for object selection"),
        "gr_cut": StageParameter(float, required=True, msg="Color cut threshold for object selection"),
        "ri_cut": StageParameter(float, required=True, msg="Color cut threshold for object selection"),
        "iz_cut": StageParameter(float, required=True, msg="Color cut threshold for object selection"),
        "mfrac_cut": StageParameter(float, required=True, msg="mfrac threshold for object selection"),
        "gauss_T_cut": StageParameter(float, required=True, msg="gauss_T threshold for object selection"),
    }

    def data_iterator(self):
        # As above, this is where we work out which columns we need.
        chunk_rows = self.config["chunk_rows"]
        bands = self.config["bands"]

        # Core quantities we need
        shear_cols = metadetect_variants("T", "s2n", "g1", "g2", "ra", "dec", "weight", "psf_T_mean", "flags",  "is_primary", "gauss_T", "mfrac")

        # Magnitudes and errors
        shear_cols += band_variants(bands, "mag", "mag_err", shear_catalog_type="metadetect")

        # We need truth shears and/or PZ point-estimates for each shear too
        if self.config["input_pz"]:
            shear_cols += metadetect_variants("mean_z")
        elif self.config["true_z"]:
            shear_cols += metadetect_variants("redshift_true")

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
        calculators = [
            MetaDetectCalculator(select_tomographic_weak_lensing_sample_metadetect_dp2, delta_gamma)
            for i in range(nbin_source)
        ]
        calculators.append(MetaDetectCalculator(select_weak_lensing_sample_metadetect_dp2, delta_gamma))
        return calculators


def select_weak_lensing_sample_metadetect_dp2(data, config, calling_from_select=False):
    """
    Select weak lensing sample objects for metadetect catalogs.

    This starts from the general cuts in select_weak_lensing_sample (flags,
    size, S/N, mask fraction, tomographic bin) and then applies extra cuts
    that only make sense for metadetect catalogs. Add / remove cuts below
    and re-run to iterate.
    """

    sel = select_weak_lensing_sample(data, config, calling_from_select=calling_from_select)
    n0 = sel.size

    # --- metadetect-specific cuts go here ---
    mag_g_cut = config["mag_g_cut"]
    mag_r_cut = config["mag_r_cut"]
    mag_i_cut = config["mag_i_cut"]
    mag_z_cut = config["mag_z_cut"]
    gmr_cut = config["gr_cut"]
    rmi_cut = config["ri_cut"]
    imz_cut = config["iz_cut"]
    gauss_T_cut = config['gauss_T_cut']
    mfrac_cut = config['mfrac_cut']

    # We should also have some crazy color cuts and magnitude cuts which should come from PZ group
    sel &= (data["mag_g"] < mag_g_cut) & \
        (data["mag_r"] < mag_r_cut) & \
        (data["mag_i"] < mag_i_cut) & \
        (data["mag_z"] < mag_z_cut) & \
        (np.abs(data["mag_g"] - data["mag_r"]) < gmr_cut) & \
        (np.abs(data["mag_r"] - data["mag_i"]) < rmi_cut) & \
        (np.abs(data["mag_i"] - data["mag_z"]) < imz_cut) & \
        (data['gauss_T'] < gauss_T_cut) & \
        (data['mfrac'] < mfrac_cut)

    # Adding all the flags cut to make sure we are not using any objects with flags set.
    # The flags was made from all the ohter ones.
    # is_primary should actually automatically be true
    sel &= (data["flags"] == 0) & (data["is_primary"] == True)

    return sel


def select_tomographic_weak_lensing_sample_metadetect_dp2(data, config, bin_index):
    """
    Tomographic counterpart to select_weak_lensing_sample_metadetect_dp2, in the
    same way that select_tomographic_weak_lensing_sample relates to
    select_weak_lensing_sample.
    """
    zbin = data["zbin"]
    verbose = config["verbose"]

    sel = select_weak_lensing_sample_metadetect_dp2(data, config, calling_from_select=True)
    sel &= zbin == bin_index
    f4 = sel.sum() / sel.size

    if verbose:
        print(f"{f4:.2%} z for bin {bin_index}")
        print("total tomo", sel.sum())

    return sel
