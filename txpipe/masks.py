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

        mask, pixel_scheme, metadata = self.make_binary_mask()
        pix, mask, metadata = self.finalize_mask(mask, pixel_scheme, metadata)

        with self.open_output("mask", wrapper=True) as f:
            f.file.create_group("maps")
            f.write_map("mask", pix, mask, metadata)

    def finalize_mask(self, mask, pixel_scheme, metadata):

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

class TXCustomMask(TXSimpleMaskFrac):
    """
    Make a mask from a custom list of cuts to aux maps (e.g depth cut and bright object cuts)

    Fracdet currently taken from aux_lens_maps TODO: add option to compute this from hsp map
    """
    name = "TXCustomMask"
    # make a mask from the auxiliary maps
    inputs = [("aux_lens_maps", MapsFile),
    ]
    outputs = [("mask", MapsFile)]
    config_options = {
        "fracdet_name": "footprint/fracdet_griz",
        "cuts": [
            "footprint/fracdet_griz > 0"
            ], 
        "degrade": False, #if input map Nside differs from config nside, degrade
    }

    def run(self):

        pix, mask, metadata = self.compute_binary_mask()

        fracdet = self.get_fracdet()

        #assign fracdec for all selected pixels
        assert (mask==1.).all()
        mask = fracdet[pix]
        
        if self.config["degrade"]:
            if self.config["nside"] == metadata["nside"]:
                print('Nsides match, no degrading necessary')
            else:
                print(f'Input Nside={metadata["nside"]}, degrading to {self.config["nside"]}')
                pix, mask, metadata = self.degrade(pix, mask, metadata, self.config["nside"])

        with self.open_output("mask", wrapper=True) as f:
            f.file.create_group("maps")
            f.write_map("mask", pix, mask, metadata)

    def make_binary_mask(self):
        import healpy
        import re

        #get a list of quantities to be cut
        cuts = self.config["cuts"]
        compare_pattern = r'(==|!=|>=|<=|>|<)'
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
            masks.append( (name, eval(cut_string_eval)))

        # hit is only used to print fraction of pixels removed from each cut
        # Arbitrarily choose the first map specified
        hit = values[names[0]] > healpy.UNSEEN

        for name, m in masks:
            frac = 1 - (m & hit).sum() / hit.sum()
            print(f"Mask '{name}' removes fraction {frac:.3f} of {names[0]} pixels")

        # Overall mask
        mask = np.logical_and.reduce([mask for _, mask in masks])

        return mask, pixel_scheme, metadata

    def get_fracdet(self, ):
        with self.open_input("aux_lens_maps", wrapper=True) as f:
            fracdet = f.read_map(self.config["fracdet_name"])
        return fracdet
    
    def degrade(self, pix, mask, metadata_in, nside_out):
        """
        Degrades a fracdet map to a low res fracdet map using healsparse 
        """
        import healsparse as hsp
        import healpy as hp 
        import copy

        #convert our pixel, mask arrays into a healsparse map
        nside_coverage = 32
        map_hsp = hsp.HealSparseMap.make_empty(nside_coverage, metadata_in['nside'], dtype=type(mask[0]), sentinel=hp.UNSEEN )
        if not metadata_in['nest']:
            pix = hp.ring2nest(metadata_in['nside'], pix)
        map_hsp.update_values_pix( pixels=pix, values=mask)

        #do a "sum" degrade of the frac mask
        map_degraded_sum = map_hsp.degrade(nside_out, reduction='sum')

        # Divide the sum of the mask by the ratio of pixel areas
        degraded_pixels = np.unique(map_degraded_sum.valid_pixels) #TODO: figure out why there are sometimes duplicates in valid_pixels
        mask_out = map_degraded_sum[degraded_pixels]*(nside_out/metadata_in['nside'])**2.

        select_nonzero = (mask_out != 0.)
        if not metadata_in['nest']:
            pix_out = hp.nest2ring(nside_out, degraded_pixels[select_nonzero])
        else:
            pix_out = degraded_pixels[select_nonzero]
        mask_out = mask_out[select_nonzero]

        metadata_out = copy.copy(metadata_in)
        metadata_out['nside'] = nside_out

        #sanity_check: make sure area in == area out
        area_in  = np.sum(mask)*hp.nside2pixarea(metadata_in['nside'], degrees=True)
        area_out = np.sum(mask_out)*hp.nside2pixarea(metadata_out['nside'], degrees=True)
        assert np.round(area_in,3) == np.round(area_out,3)

        return pix_out, mask_out, metadata_out




        

