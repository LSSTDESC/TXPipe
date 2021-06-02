import numpy as np
from .utils import choose_pixelization
from .base_stage import PipelineStage
from .data_types import MapsFile


class TXSimpleMask(PipelineStage):
    name = "TXSimpleMask"
    # make a mask from the auxiliary maps
    inputs = [("aux_maps", MapsFile)]
    outputs = [("mask", MapsFile)]
    config_options = {
        "depth_cut": 23.5,
        "bright_object_max": 10.0,
    }

    def run(self):
        import healpy

        with self.open_input("aux_maps", wrapper=True) as f:
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

        num_hit = (mask.sum() * 1.0)
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

        with self.open_output("mask", wrapper=True) as f:
            f.write_map("mask", pix, mask, metadata)


class TXHealsparseMask(PipelineStage):
    """
    Ingest a HealSparse map into TXPipe format.
    """
    name = "TXExternalMask"

    # We use the auxiliary map suite (which contains depth maps and related quantities)
    # just to choose get the Nside parameter.
    inputs = [("aux_maps", MapsFile)]

    outputs = [("mask", MapsFile)]

    config_options = {
        "mask_path": str,
    }

    def run(self):
        import healpy
        import healsparse


        # Get the pixelization scheme
        with self.open_input("aux_maps", wrapper=True) as f:
            metadata = dict(f.file["maps"].attrs)
            pixel_scheme = choose_pixelization(**metadata)

        # Load the map, in this case from a Healsparse file
        mask = self.read_map(pixel_scheme)

        # In-place clip to positive values only, and remove nan. This also removes
        # healpix UNSEEN values, which are negative
        np.nan_to_num(mask, copy=False, nan=0, neginf=0, posinf=0)
        np.clip(mask, 0, np.inf, out=mask)

        # Convert to float64 type everywhere
        pix = np.where(mask)[0]
        mask = mask[pix].astype(float)

        # Determinet the unmasked area, report and store it.
        area = pixel_scheme.pixel_area(degrees=True) * pix.size
        f_sky = area / 41252.96125
        print(f"f_sky = {f_sky}")
        print(f"area = {area:.2f} sq deg")
        metadata["area"] = area
        metadata["f_sky"] = f_sky

        # Write to a mask file
        with self.open_output("mask", wrapper=True) as f:
            f.write_map("mask", pix, mask, metadata)


    def read_map(self, pixel_scheme)
        import healpy
        import healsparse

        # Read the 
        hsp_mask = healsparse.HealSparseMap.read(self.config['mask_path'])
        nside = hsp_mask.nside_sparse

        # Complain if the external mask size is too small for what we
        # need here
        if pixel_scheme.nside > nside:
            raise ValueError("Nside for the external sparse mask must be "
                             "larger than nside for the object map")

        # If needed, degrade the mask to our target resolution
        if pixel_scheme.nside < nside:
            hsp_mask = hsp_mask.degrade(nside_out=pixel_scheme.nside)

        # Load the map.  The default Healsparse ordering is NEST, unlike many
        # other uses, so we probably have to re-order from nest to ring here.
        mask = hsp_mask.generate_healpix_map()
        if not pixel_scheme.nest:
            mask = healpy.reorder(mask, r2n=True)


        return mask
