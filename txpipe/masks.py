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

            f.file.create_group("maps")
            f.write_map("mask", pix, mask, metadata)

class TXExternalMask(PipelineStage):
    """
    This class reads an external mask
    in healsparse format
    """
    name = "TXExternalMask"
    # make a mask from the auxiliary maps
    inputs = [("aux_maps", MapsFile)]
    outputs = [("mask", MapsFile)]
    config_options = {
        "ext_mask": '',
    }

    def run(self):
        import healpy
        import healsparse
        metadata = dict()
        hsp_mask = healsparse.HealSparseMap.read(self.config['ext_mask'])
        nside = hsp_mask.nside_sparse
        nside_cov = hsp_mask.nside_coverage
        with self.open_input("aux_maps", wrapper=True) as f:
            metadata = dict(f.file["maps"].attrs)
            pixel_scheme = choose_pixelization(**metadata)
        try:
            if pixel_scheme.nside < nside:
                hsp_mask = hsp_mask.degrade(nside_out=pixel_scheme.nside)
            elif pixel_scheme.nside > nside:
                raise ValueError("""Nside for the sparse map is expected to be 
                                 larger than nside for the object map""")
            if pixel_scheme.nest == False:
                hsp_mask = hsp_mask.generate_healpix_map()
                aux = np.zeros_like(hsp_mask)
                pxnums = healpy.nest2ring(pixel_scheme.nside, np.where(hsp_mask != healpy.UNSEEN))
                aux[pxnums] = hsp_mask[hsp_mask!= healpy.UNSEEN]
                hsp_mask = aux
            else:
                hsp_mask = hsp_mask.generate_healpix_map()
        except AttributeError:
            raise ValueError('External masks can only be used with HEALpix scheme')
            
        hit = hsp_mask != healpy.UNSEEN

        # Overall mask
        
        num_hit = (hsp_mask.sum() * 1.0)
        area = pixel_scheme.pixel_area(degrees=True) * num_hit
        f_sky = area / 41252.96125
        print(f"f_sky = {f_sky}")
        print(f"area = {area:.2f} sq deg")
        metadata["area"] = area
        metadata["f_sky"] = f_sky

        hsp_mask[np.isnan(hsp_mask)] = 0.0
        hsp_mask[hsp_mask < 0] = 0

        pix = np.where(hsp_mask)[0]
        hsp_mask = hsp_mask[pix].astype(float)

        with self.open_output("mask", wrapper=True) as f:

            f.file.create_group("maps")
            f.write_map("mask", pix, hsp_mask, metadata)
