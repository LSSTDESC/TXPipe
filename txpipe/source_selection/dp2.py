from .metadetect import TXSourceSelectorMetadetect
from .base import select_weak_lensing_sample, TXSourceSelectorBase
from ..data_types import ShearCatalog, HDFFile
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
        calculators = [
            MetaDetectCalculator(select_tomographic_weak_lensing_sample_metadetect_dp2, delta_gamma)
            for i in range(nbin_source)
        ]
        calculators.append(MetaDetectCalculator(select_weak_lensing_sample_metadetect_dp2, delta_gamma))
        return calculators



class TXDP2BasicSelectionMetadetect(PipelineStage):
    """Make a cut catalog applying the underlying DP2 selection without any tomograpy.

    RAIL's SOMPZ approach has a slightly different sequence of calculations to what we
    have so far been assuming in TXPipe. In that model, RAIL's SOMPZ stages want to combine
    tomography and n(z) calculation, so we need to supply it a catalog that has already
    been cut down to just the core (SNR, size, etc) selection used for the WL sample.

    So our strategy here is to generate a stripped-down catalog to pass to SOMPZ, but also
    to save the selection so that later we can correctly do the shear selection bias calibration.

    Hopefully in DR1 we can improve the integration.
    """

    inputs = [
        ("shear_catalog", ShearCatalog),
    ]
    outputs = [
        ("cut_shear_catalog", ShearCatalog),
        ("base_shear_selection_catalog", HDFFile),
    ]

    config_options = TXSourceSelectorMetadetectDP2.config_options.copy()

    def run(self):
        
        input_file = self.open_input("shear_catalog")
        output_file = self.open_output("cut_shear_catalog", parallel=True)
        selection_file = self.open_output("base_shear_selection_catalog", parallel=True)
        output_group, selection_group = self.setup_output(input_file, output_file, selection_file)
        

        # We made full-length catalogs for the output file above, but of course
        # they will be smaller really because we will throw away some data.
        # But we don't know how small until we do it, so for now we will just
        # keep track of the sizes of all the outputs for each variant        
        final_sizes = np.zeros(len(META_VARIANTS), dtype=int)
        # Each process selects and writes a different variant, so at least
        # we can use up to 5 parallel processes
        for variant in self.split_tasks_by_rank(META_VARIANTS):
            index = 0
            output_subgroup = output_group[variant]
            # Read a chunk of data
            for (s, e, data) in self.data_iterator(input_file, variant):

                # Run the standard selection function
                sel = select_tomographic_weak_lensing_sample_metadetect_dp2(data, self.config)

                # Save the selection mask
                selection_group[f"select_{variant}"][s:e] = sel
                
                # output just the selected part to the new catalog.
                # It's not clear we actually want to save all the columns here
                # the content that is needed is what RAIL needs to do PZ.
                # I think that's just magnitudes, weight, ra, dec (for the clustering PZ)
                # so really I should cut that down here, especially when
                # we have the larger DP2 catalogs. That's a TODO.
                n = sel.sum()
                for name, col in data.items():
                    output_subgroup[name][index: index + n] = col
                index += n
            # Record the size of the selected data so we can resize later
            final_sizes[META_VARIANTS.index(variant)] = index
        
        # resize all the columns
        self.resize_all_columns(output_group, final_sizes)

    def setup_output(self, input_file, output_file, selection_file):
        output_group = output_file.create_group("shear")
        g = f.create_group("shear")

        # Create a group for each variant (1p, 1m, etc.)
        sizes = {}
        for variant in META_VARIANTS:
            input_subgroup = input_file[f"shear/{variant}"]
            output_subgroup = output_group.create_group(variant)
            # and in each of those groups create a copy of all the columns.
            # We make it the same size as the input column and cut it down at the end.
            size = input_subgroup['ra'].size
            sizes[variant] = size
            for col in input_subgroup.keys():
                if not col.size == size:
                    raise ValueError("Columns in shear cat are different sizes")
                output_subgroup.create_dataset(name=name, dtype=col.dtype, shape=col.shape, maxshape=(None,))

        # We will also save the boolean selection function so that later on we can combine
        # this with the tomographic bin selection that RAIL will return
        selection_group = selection_file.create_group("selection")
        for variant in META_VARIANTS:
            size = sizes[varant]
            selection_group.create_dataset(f"select_{variant}", dtype=bool, shape=(size,))
        
        return output_group, selection_group

        
    # resize each one
    def resize_all_columns(self, output_group, final_sizes):
        # each of the proceses does a different variant, and the array of final
        # sizes is by variant, so it should be zero for the entries corresponding
        # to the other processes, and so summing in-place should work fine.
        in_place_reduce(final_sizes, self.comm, allreduce=True)
        # The reason we have to share the final sizes with all the processes is that
        # operations that change the structure of the HDF5 file must be done collectively
        # by all processes, and I think that includes resizing.
        for v, variant in enumerate(META_VARIANTS):
            output_subgroup = output_group.create_group(variant)
            size = final_sizes[v]
            for col in output_subgroup.keys():
                output_subgroup[col].resize(size)


    def data_iterator(self, input_file, variant):
        chunk_size = self.config["chunk_rows"]
        group = input_file["shear/" + variant]
        index = 0
        size = group["ra"].size
        while index < size:
            s = index
            e = index + chunk_size
            data = {name: col[s:e] for name, col in group.items()}
            s = e

class TXDP2CombineBasicAndRailSelections(TXSourceSelectorMetadetect):
    """
    Combine the selection mask from TXDP2BasicSelectionMetadetect and
    the tomographic selection from RAIL into a single metadetect selection
    process. We will need an anacal version of this too.
    """
    inputs = [
        ("rail_bhat_thing_rename_me", HDFFile),
        ("base_shear_selection_catalog", HDFFile),
    ]
    outputs = [("shear_tomography_catalog", TomographyCatalog)]

    def setup_response_calculators(self, nbin_source):
        delta_gamma = self.config["delta_gamma"]
        calculators = [
            MetaDetectCalculator(self.select, delta_gamma)
            for i in range(nbin_source)
        ]
        calculators.append(MetaDetectCalculator(self.select_2d, delta_gamma))
        return calculators


    def data_iterator(self):
        # As above, this is where we work out which columns we need.
        chunk_rows = self.config["chunk_rows"]
        tomo_file = self.open_input('rail_bhat_thing_rename_me')

        for variant in META_VARIANTS[:]:
            # we need to iterate both the mask we saved in TXDP2BasicSelectionMetadetect
            # and the tomographic bin choices that RAIL made for us.
            # they are different lengths! we should read a chunk of the base selection
            # mask, then count how many objects are selected and read that many rows
            # of the tomo bin data.
            select_col = "select_" + variant
            chunk_rows = self.config["chunk_rows"]
            tomo_index = 0
            for mask_data in self.iterate_hdf("base_shear_selection_catalog", "selection", [select_col], chunk_rows):
                sel1 = mask_data[select_col]
                nsel1 = sel1.sum()
                # now read that much data from the RAIL output
                # assume we have collated the variant bhat values I think
                tomo_bin = tomo_file[variant]["bhat_for_wide_data"][tomo_index : tomo_index + nsel1]
                full_tomo_bin = np.full(sel1.size, -1)
                full_tomo_bin[np.where(sel1)] = tomo_bin
                # empty selections for all the other variants.
                # this should be less messy. It comes from how the metadetect calculator
                # was designed assuming that all the variants are mixed up together.
                data = {}
                for v in META_VARIANTS:
                    data["bin_" + v] = np.zeros([], dtype=int)
                    data["base_selection_" + v] = np.zeros([], dtype=bool)
                data["bin_" + variant] = tomo_bin
                data["base_selection_" + variant] = sel1
                yield data

    
    def select(self, data, config, bin_index):
        sel = (data["bin"] == bin_index) & (data["base_selection"] == True)

    def select_2d(self, data, config):
        sel = (data["bin"] >= 0) &  (data["base_selection"] == True)

    



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

    # We should also have some crazy color cuts and magnitude cuts which should come from PZ group
    sel &= (data["mag_g"] < mag_g_cut) & \
        (data["mag_r"] < mag_r_cut) & \
        (data["mag_i"] < mag_i_cut) & \
        (data["mag_z"] < mag_z_cut) & \
        (np.abs(data["mag_g"] - data["mag_r"]) < gmr_cut) & \
        (np.abs(data["mag_r"] - data["mag_i"]) < rmi_cut) & \
        (np.abs(data["mag_i"] - data["mag_z"]) < imz_cut)

    # Follow the same pattern as select_weak_lensing_sample, but add extra cuts for metadetect catalogs.:
    mfrac_cut = config["mfrac_cut"]
    mfrac = data["object_mask_fraction"]
    sel &= mfrac < mfrac_cut

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
