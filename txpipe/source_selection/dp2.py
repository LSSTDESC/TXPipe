from .metadetect import TXSourceSelectorMetadetect
from .base import select_weak_lensing_sample
from ..shear_calibration import MetaDetectCalculator


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

    verbose = config["verbose"]
    variant = data.suffix

    sel = select_weak_lensing_sample(data, config, calling_from_select=calling_from_select)
    n0 = sel.size

    # --- metadetect-specific cuts go here ---
    # Follow the same pattern as select_weak_lensing_sample, but add extra cuts for metadetect catalogs.:
    mfrac_cut = config["mfrac_cut"]
    mfrac = data["object_mask_fraction"]
    sel &= mfrac < mfrac_cut
    f_new = sel.sum() / n0

    # Adding all the flags cut to make sure we are not using any objects with flags set.
    sel &= (data["gauss_flags"] == 0) & \
            (data["pgauss_flags"] == 0) & \
            (data["gauss_shape_flags"] == 0) & \
            (data["gauss_object_flags"] == 0) & \
            (data["pgauss_object_flags"] == 0) & \
            (data["psfOriginal_flags"] == 0) & \
            (data["gauss_psfReconvolved_flags"] == 0) &\
            (data["is_primary"] == True)

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
