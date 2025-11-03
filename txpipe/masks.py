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
        pix, mask, metadata = self.finalize_mask(mask, pixel_scheme, metadata)

        with self.open_output("mask", wrapper=True) as f:
            f.file.create_group("maps")
            f.write_map("mask", pix, mask, metadata)

    def finalize_mask(self, mask, pixel_scheme, metadata):
        """
        Finalize the binary mask by flattening, cleaning NaNs/negatives,
        and extracting unmasked pixel indices. Computes area and f_sky.

        Parameters
        ----------
        mask : np.ndarray
            Raw binary mask array.
        pixel_scheme : PixelScheme
            Pixelization scheme used for area calculations.
        metadata : dict
            Metadata dictionary to update with area and f_sky.

        Returns
        -------
        pix : np.ndarray
            Indices of unmasked pixels.
        mask : np.ndarray
            Final mask values corresponding to `pix`.
        metadata : dict
            Updated metadata.
        """

        # Total survey area calculation. This is simplistic:
        # TODO: account for weights / hit fractions here, and allow
        # for different lens and shear survey areas
        num_hit = mask.sum() * 1.0
        area = pixel_scheme.pixel_area(degrees=True) * num_hit
        f_sky = area / 41252.96125
        print(f"f_sky = {f_sky}")
        print(f"area = {area:.2f} sq deg")
        metadata["area"] = area
        metadata["f_sky"] = f_sky

        mask[np.isnan(mask)] = 0.0
        mask[mask < 0] = 0

        # Pull out unmasked pixels and their values.
        # The flatten only affects gnomonic maps; the
        # healpix maps are already flat
        mask = mask.flatten()
        pix = np.where(mask)[0]
        mask = mask[pix].astype(float)

        return pix, mask, metadata

    def compute_fracdet_from_hsp(self, metadata):
        """
        Computes detection fraction from an input healsparse map (higher resolution, binary)

        Parameters
        ----------
        metadata : dict
            Metadata with target resolution (e.g. 'nside').

        Returns
        -------
        fracdet : np.ndarray
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
        Computes detection fraction from a list of input healsparse map (higher resolution, binary)

        Returns a map of the fractional of valid pixels that are present in *all* the input maps

        Parameters
        ----------
        metadata : dict
            Metadata with target resolution (e.g. 'nside').

        Returns
        -------
        fracdet : np.ndarray
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
        mask : np.ndarray
            Boolean mask array.
        pixel_scheme : PixelScheme
            Pixelization object.
        metadata : dict
            Metadata from input file.
        """
        import healpy

        with self.open_input("aux_lens_maps", wrapper=True) as f:
            metadata = dict(f.file["maps"].attrs)
            bright_obj = f.read_map("bright_objects/count")
            depth = f.read_map("depth/depth")
            pixel_scheme = choose_pixelization(**metadata)
        hit = depth > healpy.UNSEEN
        masks = [
            ("depth", depth > self.config["depth_cut"]),
            ("bright_obj", ~(bright_obj > self.config["bright_object_max"])),
        ]

        for name, m in masks:
            frac = 1 - (m & hit).sum() / hit.sum()
            print(f"Mask '{name}' removes fraction {frac:.3f} of hit pixels")

        # Overall mask
        mask = np.logical_and.reduce([mask for _, mask in masks])

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
        mask : np.ndarray
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

        mask, pixel_scheme, metadata = self.make_binary_mask()
        pix, mask, metadata = self.finalize_mask(mask, pixel_scheme, metadata)

        assert self.config["supreme_map_file"] == "none" or self.config["supreme_map_files"] == [], (
            "You have specified both map_file and map_files, pick one"
        )

        if self.config["supreme_map_file"] is not "none":
            fracdet = self.compute_fracdet_from_hsp(metadata)
        else:
            fracdet = self.compute_fracdet_from_hsp_list(metadata)

        # assign fracdec for all selected pixels
        assert (mask == 1.0).all()
        if metadata["nest"]:
            mask = fracdet[pix]
        else:
            import healpy as hp

            mask = fracdet[hp.ring2nest(metadata["nside"], pix)]

        # cut pixels with low fracdet
        select_frac_cut = mask > self.config["frac_cut"]
        pix = pix[select_frac_cut]
        mask = mask[select_frac_cut]

        with self.open_output("mask", wrapper=True) as f:
            f.file.create_group("maps")
            f.write_map("mask", pix, mask, metadata)


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

        mask, pixel_scheme, metadata = self.make_binary_mask()
        pix, mask, metadata = self.finalize_mask(mask, pixel_scheme, metadata)

        fracdet = self.get_fracdet()

        # assign fracdec for all selected pixels
        assert (mask == 1.0).all()
        mask = fracdet[pix]

        if self.config["degrade"]:
            if self.config["nside"] == metadata["nside"]:
                print("Nsides match, no degrading necessary")
            else:
                print(f"Input Nside={metadata['nside']}, degrading to {self.config['nside']}")
                pix, mask, metadata = self.degrade(pix, mask, metadata, self.config["nside"])

        with self.open_output("mask", wrapper=True) as f:
            f.file.create_group("maps")
            f.write_map("mask", pix, mask, metadata)

    def make_binary_mask(self):
        """
        Create a binary mask from arbitrary user-defined cuts on auxiliary maps.

        Returns
        -------
        mask : np.ndarray
            Boolean mask array.
        pixel_scheme : PixelScheme
            Pixelization object.
        metadata : dict
            Metadata from input.
        """
        import healpy
        import re

        # get a list of quantities to be cut
        cuts = self.config["cuts"]
        compare_pattern = r"(==|!=|>=|<=|>|<)"
        names = np.unique([re.split(compare_pattern, cut_string)[0].strip() for cut_string in cuts])

        # load the values into a dict
        # We will assume these can all be held in memory, but note very high-res maps will not allow this
        values = {}
        with self.open_input("aux_lens_maps", wrapper=True) as f:
            metadata = dict(f.file["maps"].attrs)
            for name in names:
                values[name] = f.read_map(name)
            pixel_scheme = choose_pixelization(**metadata)

        # evaluate the mask for each cut
        masks = []
        for cut_string in cuts:
            name = re.split(compare_pattern, cut_string)[0].strip()
            cut_string_eval = cut_string.replace(name, f"values['{name}']")
            print(cut_string_eval)
            masks.append((name, eval(cut_string_eval)))

        # hit is only used to print fraction of pixels removed from each cut
        # Arbitrarily choose the first map specified
        hit = values[names[0]] > healpy.UNSEEN

        for name, m in masks:
            frac = 1 - (m & hit).sum() / hit.sum()
            print(f"Mask '{name}' removes fraction {frac:.3f} of {names[0]} pixels")

        # Overall mask
        mask = np.logical_and.reduce([mask for _, mask in masks])

        return mask, pixel_scheme, metadata

    def get_fracdet(self):
        """
        Load fractional detection map from auxiliary maps input.

        Returns
        -------
        fracdet : np.ndarray
            Fracdet array.
        """
        with self.open_input("aux_lens_maps", wrapper=True) as f:
            fracdet = f.read_map(self.config["fracdet_name"])
        return fracdet

    def degrade(self, pix, mask, metadata_in, nside_out):
        """
        Degrade a high-resolution fractional mask to lower resolution using healsparse

        Parameters
        ----------
        pix : np.ndarray
            Input pixel indices.
        mask : np.ndarray
            Input mask values.
        metadata_in : dict
            Input metadata.
        nside_out : int
            Desired lower-resolution nside.

        Returns
        -------
        pix_out : np.ndarray
            Output pixel indices at lower resolution.
        mask_out : np.ndarray
            Degraded mask values.
        metadata_out : dict
            Updated metadata.
        """
        import healsparse as hsp
        import healpy as hp
        import copy

        # convert our pixel, mask arrays into a healsparse map
        nside_coverage = 32
        map_hsp = hsp.HealSparseMap.make_empty(
            nside_coverage, metadata_in["nside"], dtype=type(mask[0]), sentinel=hp.UNSEEN
        )
        if not metadata_in["nest"]:
            pix = hp.ring2nest(metadata_in["nside"], pix)
        map_hsp.update_values_pix(pixels=pix, values=mask)

        # do a "sum" degrade of the frac mask
        map_degraded_sum = map_hsp.degrade(nside_out, reduction="sum")

        # Divide the sum of the mask by the ratio of pixel areas
        degraded_pixels = np.unique(
            map_degraded_sum.valid_pixels
        )  # TODO: figure out why there are sometimes duplicates in valid_pixels
        mask_out = map_degraded_sum[degraded_pixels] * (nside_out / metadata_in["nside"]) ** 2.0

        select_nonzero = mask_out != 0.0
        if not metadata_in["nest"]:
            pix_out = hp.nest2ring(nside_out, degraded_pixels[select_nonzero])
        else:
            pix_out = degraded_pixels[select_nonzero]
        mask_out = mask_out[select_nonzero]

        metadata_out = copy.copy(metadata_in)
        metadata_out["nside"] = nside_out

        # sanity_check: make sure area in == area out
        area_in = np.sum(mask) * hp.nside2pixarea(metadata_in["nside"], degrees=True)
        area_out = np.sum(mask_out) * hp.nside2pixarea(metadata_out["nside"], degrees=True)
        assert np.round(area_in, 3) == np.round(area_out, 3)

        return pix_out, mask_out, metadata_out
