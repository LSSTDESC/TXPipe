import numpy as np
from .utils import choose_pixelization
from .base_stage import PipelineStage
from .data_types import MapsFile
from ceci.config import StageParameter


class TXBaseMask(PipelineStage):
    """
    Base class for generating binary survey masks using auxiliary input maps.
    Subclasses should implement `make_binary_mask`, which defines the logic for mask construction.

    The base class handles writing the mask to output and computing metadata such as area and f_sky.
    """

    name = "TXBaseMask"
    parallel = False
    # make a mask from the auxiliary maps
    outputs = [("mask", MapsFile)]

    def run(self):
        """
        Run the pipeline stage: generate a binary mask, finalize it,
        and write to output.
        """
        mask, pixel_scheme, metadata = self.make_binary_mask()
        mask, metadata = self.finalize_mask(mask, pixel_scheme, metadata)

        with self.open_output("mask", wrapper=True) as f:
            f.file.create_group("maps")
            f.write_map("mask", mask, metadata)

    def finalize_mask(self, mask, pixel_scheme, metadata):
        """
        Finalize the binary mask by flattening, cleaning NaNs/negatives,
        and extracting unmasked pixel indices. Computes area and f_sky.

        Parameters
        ----------
        mask : hsp.HealSparseMap
            binary mask map.
        pixel_scheme : PixelScheme
            Pixelization scheme used for area calculations.
        metadata : dict
            Metadata dictionary to update with area and f_sky.

        Returns
        -------
        mask : hsp.HealSparseMap or np.ndarray
            Final mask 
        metadata : dict
            Updated metadata.
        """

        # Total survey area calculation. This is simplistic:
        # TODO: account for weights / hit fractions here, and allow
        # for different lens and shear survey areas
        num_hit = mask.n_valid * 1.0
        area = pixel_scheme.pixel_area(degrees=True) * num_hit
        f_sky = area / 41252.96125
        print(f"f_sky = {f_sky}")
        print(f"area = {area:.2f} sq deg")
        metadata["area"] = area
        metadata["f_sky"] = f_sky

        #Note: leaving these here in case we want to keep the gnomonic maps?
        if pixel_scheme == "gnomonic":
            mask[np.isnan(mask)] = 0.0
            mask[mask < 0] = 0
            
            # The flatten required for gnomonic maps
            mask = mask.flatten()
            pix = np.where(mask)[0]
            mask = mask[pix].astype(float)

        return mask, metadata

    def compute_fracdet_from_hsp(self, metadata):
        """
        Computes detection fraction from an input healsparse map (higher resolution, binary)

        Parameters
        ----------
        metadata : dict
            Metadata with target resolution (e.g. 'nside').

        Returns
        -------
        fracdet : hsp.HealSparseMap
            Fractional detection array at the config nside.
        """
        import healsparse

        # load healsparse map
        supreme_map_file = self.config["supreme_map_file"]
        print(f"Generating fracdet map from {supreme_map_file}")
        spmap = healsparse.HealSparseMap.read(supreme_map_file)

        # fracdet it
        nside = metadata["nside"]
        fracdet = spmap.fracdet_map(nside)

        return fracdet

    def compute_fracdet_from_hsp_list(self, metadata):
        """
        Computes detection fraction from a list of input healsparse maps (higher resolution, binary)

        Returns a map of the fractional of valid pixels that are present in *all* the input maps

        Parameters
        ----------
        metadata : dict
            Metadata with target resolution (e.g. 'nside').

        Returns
        -------
        fracdet : hsp.HealSparseMap
            Fractional detection array at the config nside.
        """
        import healsparse

        # load healsparse map
        supreme_map_file_list = self.config["supreme_map_files"]
        print(f"Generating fracdet map from {supreme_map_file_list}")
        spmap_list = [healsparse.HealSparseMap.read(supreme_map_file) for supreme_map_file in supreme_map_file_list]
        spmap = healsparse.operations.max_intersection(spmap_list)

        # fracdet it
        nside = metadata["nside"]
        fracdet = spmap.fracdet_map(nside)

        return fracdet


class TXSimpleMask(TXBaseMask):
    """
    Generate a simple binary mask using cuts on depth and bright object maps.
    """

    name = "TXSimpleMask"
    inputs = [("aux_lens_maps", MapsFile)]
    config_options = {
        "depth_cut": StageParameter(float, 23.5, msg="Depth cut for mask creation."),
        "bright_object_max": StageParameter(float, 10.0, msg="Maximum allowed bright object count."),
    }

    def make_binary_mask(self):
        """
        Create a binary mask by applying cuts to depth and bright object count maps.

        Returns
        -------
        mask : hsp.HealSparseMap
            Boolean mask array.
        pixel_scheme : PixelScheme
            Pixelization object.
        metadata : dict
            Metadata from input file.
        """
        import healsparse as hsp
        from functools import reduce
        from operator import and_

        with self.open_input("aux_lens_maps", wrapper=True) as f:
            metadata = dict(f.file["maps"].attrs)
            bright_obj = f.read_map("bright_objects/count")
            depth = f.read_map("depth/depth")
            pixel_scheme = choose_pixelization(**metadata)
        
        #TODO: this can be chunked if we dont want to access all valid pixels at the same time
        valid_pix = depth.valid_pixels
        hit = hsp.HealSparseMap.make_empty(depth.nside_coverage, depth.nside_sparse, dtype=np.bool )
        hit.update_values_pix(valid_pix, 1)

        masks = {
            "depth": hsp.HealSparseMap.make_empty_like(hit),
            "bright_obj": hsp.HealSparseMap.make_empty_like(hit),
        }
        masks["depth"].update_values_pix(valid_pix, depth[valid_pix] > self.config["depth_cut"])
        masks["bright_obj"].update_values_pix(valid_pix, ~(bright_obj[valid_pix] > self.config["bright_object_max"]) )

        for name, hsp_map in masks.items():
            frac = 1 - (hsp_map & hit).n_valid/hit.n_valid
            print(f"Mask '{name}' removes fraction {frac:.3f} of hit pixels")

        # Overall mask
        # Note healpsarse only knows about python _and, not numpy's logical_and
        #mask = reduce(and_, [mask for _, mask in masks.items()])
        mask = hsp.operations.and_intersection([mask for _, mask in masks.items()])

        return mask, pixel_scheme, metadata


class TXSimpleMaskSource(TXBaseMask):
    """
    Generate a binary mask for source galaxies using positive lensing weights
    across source bins.
    """

    name = "TXSimpleMaskSource"
    # make a mask from the source maps
    inputs = [("source_maps", MapsFile)]
    config_options = {}

    def make_binary_mask(self):
        """
        Mask all pixels where lensing weights are zero for any source bin.

        Returns
        -------
        mask : hsp.HealSparseMap
            Boolean mask array.
        pixel_scheme : PixelScheme
            Pixelization object.
        metadata : dict
            Metadata from input file.
        """
        lensing_weights = []
        with self.open_input("source_maps", wrapper=True) as f:
            metadata = dict(f.file["maps"].attrs)
            pixel_scheme = choose_pixelization(**metadata)
            for i in range(metadata["nbin_source"]):
                lw = f.read_map(f"lensing_weight_{i}")
                lensing_weights.append(lw)

        lensing_weights = np.array(lensing_weights)
        mask = np.logical_and.reduce(lensing_weights > 0.0, axis=0)

        return mask, pixel_scheme, metadata


class TXSimpleMaskFrac(TXSimpleMask):
    """
    Make a simple mask using a depth cut and bright object cut
    Include the fractional coverage of each pixel using a high-res survey property map (e.g. N-exposures for given band(s) )

    #NOTE: This assumes all cuts to the mask are being done within TXpipe at the config Nside
    #if we want to apply cuts on the survey property map nside (higher resolution), we will need to add this feature
    """

    name = "TXSimpleMaskFrac"
    parallel = False
    # make a mask from the auxiliary maps
    inputs = [
        ("aux_lens_maps", MapsFile),
    ]
    outputs = [("mask", MapsFile)]
    config_options = {
        "depth_cut": StageParameter(float, 23.5, msg="Depth cut for mask creation."),
        "bright_object_max": StageParameter(float, 10.0, msg="Maximum allowed bright object count."),
        "supreme_map_file": StageParameter(str, "none", msg="Path to supreme map file for fracdet computation."),
        "supreme_map_files": StageParameter(list, [], msg="List of supreme map files for fracdet computation."),
        "frac_cut": StageParameter(float, 0.0, msg="Minimum fractional coverage to keep pixel in mask."),
    }

    def run(self):
        """
        Apply mask logic and replace selected pixels with fractional coverage values.
        """
        import healsparse as hsp

        mask, pixel_scheme, metadata = self.make_binary_mask()
        mask, metadata = self.finalize_mask(mask, pixel_scheme, metadata)

        assert self.config["supreme_map_file"] == "none" or self.config["supreme_map_files"] == [], (
            "You have specified both map_file and map_files, pick one"
        )

        if self.config["supreme_map_file"] is not "none":
            fracdet = self.compute_fracdet_from_hsp(metadata)
        else:
            fracdet = self.compute_fracdet_from_hsp_list(metadata)
        
        #boolean mask of pixels with high fracdet
        frac_cut_mask = hsp.HealSparseMap.make_empty_like(mask)
        frac_cut_mask[fracdet.valid_pixels] = (fracdet[fracdet.valid_pixels] > self.config["frac_cut"])

        #find the intersection of the mask cuts and the fractional coverage map
        frac_det_mask = hsp.operations.product_intersection([fracdet, mask, frac_cut_mask])

        with self.open_output("mask", wrapper=True) as f:
            f.file.create_group("maps")
            f.write_map("mask", frac_det_mask, metadata)


class TXCustomMask(TXSimpleMaskFrac):
    """
    Make a mask from a custom list of cuts to aux maps (e.g depth cut and bright object cuts)

    Fracdet currently taken from aux_lens_maps TODO: add option to compute this from hsp map
    """

    name = "TXCustomMask"
    # make a mask from the auxiliary maps
    inputs = [
        ("aux_lens_maps", MapsFile),
    ]
    outputs = [("mask", MapsFile)]
    config_options = {
        "fracdet_name": StageParameter(str, "footprint/fracdet_griz", msg="Fracdet map name."),
        "cuts": StageParameter(list, ["footprint/fracdet_griz > 0"], msg="List of mask cuts to apply."),
        "degrade": StageParameter(bool, False, msg="Degrade resolution if input map Nside differs from config nside."),
    }

    def run(self):
        """
        Apply custom mask logic, assign fracdet values, and optionally degrade resolution.
        """
        import healsparse as hsp

        mask, pixel_scheme, metadata = self.make_binary_mask()
        mask, metadata = self.finalize_mask(mask, pixel_scheme, metadata)

        fracdet = self.get_fracdet()

        # assign fracdec for all selected pixels
        valid_pix = mask.valid_pixels
        masked_frac_map = hsp.HealSparseMap.make_empty_like(fracdet)
        masked_frac_map[valid_pix] = fracdet[valid_pix]

        if self.config["degrade"]:
            if self.config["nside"] == metadata["nside"]:
                print("Nsides match, no degrading necessary")
            else:
                print(f"Input Nside={metadata['nside']}, degrading to {self.config['nside']}")
                masked_frac_map, metadata = self.degrade(masked_frac_map, metadata, self.config["nside"])

        with self.open_output("mask", wrapper=True) as f:
            f.file.create_group("maps")
            f.write_map("mask", masked_frac_map, metadata)

    def make_binary_mask(self):
        """
        Create a binary mask from arbitrary user-defined cuts on auxiliary maps.

        Returns
        -------
        mask : hsp.HealSparseMap
            Boolean mask array.
        pixel_scheme : PixelScheme
            Pixelization object.
        metadata : dict
            Metadata from input.
        """
        import healpy
        import healsparse as hsp
        import re
        import operator

        OP_MAP = {
            ">": operator.gt,
            ">=": operator.ge,
            "<": operator.lt,
            "<=": operator.le,
            "==": operator.eq,
            "!=": operator.ne,
        }

        # get a list of quantities to be cut
        cuts = self.config["cuts"]
        compare_pattern = r"(==|!=|>=|<=|>|<)"
        names = np.unique([re.split(compare_pattern, cut_string)[0].strip() for cut_string in cuts])

        masks = []

        for icut, cut_string in enumerate(cuts):
            print(cut_string)
            map_name, operator_string, value = [s.strip() for s in re.split(compare_pattern, cut_string)]
            operator = OP_MAP[operator_string]

            #load map
            with self.open_input("aux_lens_maps", wrapper=True) as f:
                metadata = dict(f.file["maps"].attrs)
                m = f.read_map(map_name)
            pixel_scheme = choose_pixelization(**metadata)
            valid_pix = m.valid_pixels

            #set value dtype to match the map 
            value = np.array(value, dtype=m.dtype).item()

            #make hit map
            # hit is only used to print fraction of pixels removed from each cut
            # Arbitrarily choose the first map specified
            if icut == 0:
                hit_name = map_name
                hit = hsp.HealSparseMap.make_empty(m.nside_coverage, m.nside_sparse, dtype=np.bool )
                hit.update_values_pix(valid_pix, 1)

            #make mask for this cut
            mask = hsp.HealSparseMap.make_empty(m.nside_coverage, m.nside_sparse, dtype=np.bool )
            mask.update_values_pix(valid_pix, operator(m[valid_pix], value) )

            masks.append(mask)

        for imask, m in enumerate(masks):
            frac = 1 - (m & hit).n_valid/hit.n_valid
            print(f"Cut '{cuts[imask]}' removes fraction {frac:.3f} of {hit.n_valid} {hit_name} pixels")

        # Overall mask
        mask = hsp.operations.and_intersection(masks)

        return mask, pixel_scheme, metadata

    def get_fracdet(self):
        """
        Load fractional detection map from auxiliary maps input.

        Returns
        -------
        fracdet : hsp.HealSparseMap
            Fracdet array.
        """
        with self.open_input("aux_lens_maps", wrapper=True) as f:
            fracdet = f.read_map(self.config["fracdet_name"])
        return fracdet

    def degrade(self, mask, metadata_in, nside_out):
        """
        Degrade a high-resolution fractional mask to lower resolution using healsparse

        Parameters
        ----------
        mask : hsp.HealSparseMap
            Input healsparse mask (either bool or a fractional coverage map)
        metadata_in : dict
            Input metadata.
        nside_out : int
            Desired lower-resolution nside.

        Returns
        -------
        mask_out : hsp.HealSparseMap
            Degraded mask.
        metadata_out : dict
            Updated metadata.
        """
        import healsparse as hsp
        import healpy as hp
        import copy

        # do a "sum" degrade of the frac mask
        map_degraded_sum = mask.degrade(nside_out, reduction="sum")

        # Divide the sum of the mask by the ratio of pixel areas
        degraded_pixels = np.unique(
            map_degraded_sum.valid_pixels
        )  # TODO: figure out why there are sometimes duplicates in valid_pixels
        mask_out = hsp.HealSparseMap.make_empty_like(map_degraded_sum)
        mask_out.update_values_pix(
            degraded_pixels, 
            map_degraded_sum[degraded_pixels] * (nside_out / metadata_in["nside"]) ** 2.0
            )

        metadata_out = copy.copy(metadata_in)
        metadata_out["nside"] = nside_out

        # sanity_check: make sure area in == area out
        area_in = np.sum(mask[mask.valid_pixels]) * hp.nside2pixarea(metadata_in["nside"], degrees=True)
        area_out = np.sum(mask_out[mask_out.valid_pixels]) * hp.nside2pixarea(metadata_out["nside"], degrees=True)
        assert np.round(area_in, 3) == np.round(area_out, 3)

        return mask_out, metadata_out
