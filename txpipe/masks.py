import numpy as np
from .utils import choose_pixelization
from .base_stage import PipelineStage
from .data_types import MapsFile

class TXBaseMask(PipelineStage):
    """
    Base-class for making masks using auxiliary maps as inputs
    """
    name = "TXBaseMask"
    parallel = False
    # make a mask from the auxiliary maps
    outputs = [("mask", MapsFile)]

    def run(self):

        pix, mask, metadata = self.compute_binary_mask()

        with self.open_output("mask", wrapper=True) as f:
            f.file.create_group("maps")
            f.write_map("mask", pix, mask, metadata)

    def compute_binary_mask(self):
        mask, pixel_scheme, metadata = self.make_binary_mask()

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
        """
        import healsparse 

        #load healsparse map
        supreme_map_file = self.config["supreme_map_file"]
        spmap = healsparse.HealSparseMap.read(supreme_map_file)

        #fracdet it
        nside = metadata["nside"]
        fracdet = spmap.fracdet_map(nside)

        return fracdet

class TXSimpleMask(TXBaseMask):
    """
    Make a simple binary mask using a depth cut and bright object cut

    """
    name = "TXSimpleMask"
    inputs = [("aux_lens_maps", MapsFile)]
    config_options = {
        "depth_cut": 23.5,
        "bright_object_max": 10.0,
    }
    
    def make_binary_mask(self):
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
    name = "TXSimpleMaskSource"
    # make a mask from the source maps
    inputs = [("source_maps", MapsFile)]
    config_options = {
    }

    def make_binary_mask(self):
        lensing_weights = []
        with self.open_input("source_maps", wrapper=True) as f:
            metadata = dict(f.file["maps"].attrs)
            pixel_scheme = choose_pixelization(**metadata)
            for i in range(metadata['nbin_source']):
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
    inputs = [("aux_lens_maps", MapsFile),
    ]
    outputs = [("mask", MapsFile)]
    config_options = {
        "depth_cut": 23.5,
        "bright_object_max": 10.0,
        "supreme_map_file": str,
    }

    def run(self):

        pix, mask, metadata = self.compute_binary_mask()

        fracdet = self.compute_fracdet_from_hsp(metadata)

        #assign fracdec for all selected pixels
        assert (mask==1.).all()
        if metadata['nest']:
            mask = fracdet[pix]
        else:
            import healpy as hp 
            mask = fracdet[hp.ring2nest(metadata['nside'], pix)]

        with self.open_output("mask", wrapper=True) as f:
            f.file.create_group("maps")
            f.write_map("mask", pix, mask, metadata)
