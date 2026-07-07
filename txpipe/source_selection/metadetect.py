from .base import TXSourceSelectorBase
from .base import select_weak_lensing_sample, select_tomographic_weak_lensing_sample
from ..shear_calibration import (
    metadetect_variants,
    MetaDetectCalculator,
    band_variants,
    META_VARIANTS
)
import numpy as np
from ceci.config import StageParameter
from ..utils import rename_iterated


class TXSourceSelectorMetadetect(TXSourceSelectorBase):
    """
    Source selection and tomography for metadetect catalogs

    This subclass selects for MetaDetect catalogs, which is expected to be used for
    Rubin data. It computes the selection bias due to object detection by repeating
    the detection process under different applied shears.

    As a consequence the different calibration columns have different lengths, since
    different objects are detected in each case.
    """

    name = "TXSourceSelectorMetadetect"

    # add one option to the base class configuration
    config_options = {
        **TXSourceSelectorBase.config_options,
        "delta_gamma": StageParameter(
            float,
            required=True,
            msg="Delta gamma value for metadetect response calculation",
        ),
        "dp2_selection": StageParameter(
            bool,
            False,
            msg="Whether to apply the metadetect DP2-specific extra selection cuts",
        ),
    }

    def data_iterator(self):
        # As above, this is where we work out which columns we need.
        chunk_rows = self.config["chunk_rows"]
        bands = self.config["bands"]

        # Core quantities we need
        shear_cols = metadetect_variants("T", "s2n", "g1", "g2", "ra", "dec", "weight", "psf_T_mean", "flags", "object_mask_fraction", "pgauss_T", "pgauss_TErr", "gauss_flags", "pgauss_flags", "gauss_shape_flags", "is_primary", "gauss_object_flags", "pgauss_object_flags", "psfOriginal_flags", "gauss_psfReconvolved_flags", "g_gaussFlux_flags", "g_pgaussFlux_flags", "r_gaussFlux_flags", "r_pgaussFlux_flags", "i_gaussFlux_flags", "i_pgaussFlux_flags", "z_gaussFlux_flags", "z_pgaussFlux_flags")

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
        if self.config["dp2_selection"]:
            tomo_selector = select_tomographic_weak_lensing_sample_metadetect_dp2
            selector = select_weak_lensing_sample_metadetect_dp2
        else:
            tomo_selector = select_tomographic_weak_lensing_sample
            selector = select_weak_lensing_sample
        calculators = [MetaDetectCalculator(tomo_selector, delta_gamma) for i in range(nbin_source)]
        calculators.append(MetaDetectCalculator(selector, delta_gamma))
        return calculators

    def write_tomography(self, outfile, start, end, source_bin, per_object_response):
        # Write out each of the individual variants.
        # The basic "bin" column was set up to be the same as the 00 variant,
        # so we can just write to all of them.
        for i, v in enumerate(META_VARIANTS):
            outfile[f"tomography/bin_{v}"][start:end] = source_bin[i]

        assert per_object_response is None, "MetaDetect does not produce per-object response values, only per-bin values, so this should be None"


    def apply_simple_redshift_cut(self, data):

        # Otherwise we have to do it once for each variant
        pz_data = {}
        variants = ["ns/", "1p/", "2p/", "1m/", "2m/"]
        for v in variants:
            if self.config["true_z"]:
                zz = data[f"{v}redshift_true"]
            else:
                zz = data[f"{v}mean_z"]

            pz_data_v = np.zeros(len(zz), dtype=int) - 1
            for zi in range(len(self.config["source_zbin_edges"]) - 1):
                mask_zbin = (zz >= self.config["source_zbin_edges"][zi]) & (
                    zz < self.config["source_zbin_edges"][zi + 1]
                )
                pz_data_v[mask_zbin] = zi

            pz_data[f"{v}zbin"] = pz_data_v

        return pz_data

    def setup_output(self):
        """
        MetaDetect outputs do not include per-object calibration values,
        only the per-bin values.
        """
        # This call to the super-class method defined above sets up most of the output
        # here, so the rest of this method only does things specific to this
        # calibration scheme
        outfile = super().setup_output()

        # For the metadetect we also want to save the selected bin for every variant.
        # We will need this later on in the pipeline for diagnostics.
        # We do the 00 variant separately because it's just the same as the base
        # case that was set up in the parent class call above, so we just link to that
        # dataset rather than creating a new one.
        with self.open_input("shear_catalog") as infile:
            for v in META_VARIANTS[1:]:
                n = infile[f"shear/{v}/ra"].size
                outfile["tomography"].create_dataset(f"bin_{v}", (n,), dtype=np.int32)
        # Link the 00 variant to the base tomography/bin dataset 
        outfile["tomography/bin_ns"] = outfile["tomography/bin"]

        # There is only global calibration information for metadetect, nothing
        # per-bin.
        nbin_source = outfile["counts/counts"].size
        group = outfile.create_group("response")
        # Per-bin 2x2 calibration matrix
        group.create_dataset("R", (nbin_source, 2, 2), dtype="f")
        # Global calibration matrix
        group.create_dataset("R_2d", (2, 2), dtype="f")
        return outfile


    def calculate_tomography(self, pz_data, shear_data, calculators):
        """
        Select objects to go in each tomographic bin and their calibration.

        Parameters
        ----------

        pz_data: table or dict of arrays
            A chunk of input photo-z data containing mean values for each object
        shear_data: table or dict of arrays
            A chunk of input shear data with metacalibration variants.
        """
        nbin = len(self.config["source_zbin_edges"]) - 1
        n = len(list(shear_data.values())[0])

        tomo_bins = []
        for v in META_VARIANTS:
            n = len(shear_data[f"{v}/g1"])
            tomo_bin = np.repeat(-1, n)
            tomo_bins.append(tomo_bin)

        data = {**pz_data, **shear_data}

        R = self.compute_per_object_response(data)

        for i in range(nbin):
            selections = calculators[i].add_data(data, self.config, i)
            for j, v in enumerate(META_VARIANTS):
                tomo_bins[j][selections[j]] = i

        # and calibrate the 2D sample.
        # This calibrator refers to select_weak_lensing_sample
        calculators[-1].add_data(data, self.config)

        return tomo_bins, R


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
