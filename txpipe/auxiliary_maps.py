from .maps import TXBaseMaps, map_config_options
import numpy as np
from .base_stage import PipelineStage
from .mapping import ShearMapper, LensMapper, FlagMapper, BrightObjectMapper, DepthMapperDR1
from .data_types import MapsFile, HDFFile, ShearCatalog
from .utils import choose_pixelization, rename_iterated, read_shear_catalog_type
from .maps import map_config_options, make_dask_maps

def make_dask_flag_maps(ra, dec, flag, max_exponent, pixel_scheme):
    import dask.array as da
    import healpy
    npix = pixel_scheme.npix
    # This seems to work directly, but we should check performance
    pix = pixel_scheme.ang2pix(ra, dec)
    maps = []
    for i in range(max_exponent):
        f = 2**i
        flag_map = da.where(flag & f, 1, 0)
        flag_map = da.bincount(pix, weights=flag_map, minlength=npix).astype(int)
        maps.append(flag_map)
    pix = da.unique(pix)
    return pix, maps

class TXAuxiliarySourceMaps(PipelineStage):
    name = "TXAuxiliarySourceMaps"
    dask_parallel = True
    
    inputs = [
        ("shear_catalog", ShearCatalog),  # for psfs
        ("shear_tomography_catalog", HDFFile),  # for per-bin psf maps
        ("source_maps", MapsFile),  # we copy the pixel scheme from here
    ]
    outputs = [
        ("aux_source_maps", MapsFile),
    ]
    config_options = {
        "block_size": 0,
        "flag_exponent_max": 8,  # flag bits go up to 2**8 by default
        "psf_prefix": "psf_",  # prefix name for columns
        **map_config_options
    }


    def choose_pixel_scheme(self):
        with self.open_input("source_maps", wrapper=True) as maps_file:
            pix_info = dict(maps_file.file["maps"].attrs)
        return choose_pixelization(**pix_info)

    def run(self):
        import dask
        import dask.array as da
        import healpy

        pixel_scheme = self.choose_pixel_scheme()
        block_size = self.config["block_size"]
        if block_size == 0:
            block_size = "auto"

        flag_exponent_max = self.config["flag_exponent_max"]

        # We have to keep this open throughout the process, because
        # dask will internally load chunks of the input hdf5 data.
        shear_cat = self.open_input("shear_catalog", wrapper=True)
        shear_tomo = self.open_input("shear_tomography_catalog", wrapper=True)
        nbin = shear_tomo.file['tomography'].attrs['nbin']

        # The "all" bin is the non-tomographic case.
        bins = list(range(nbin)) + ["all"]
        maps = {}
        group = shear_cat.get_primary_catalog_group()

        # These don't actually load all the data - everything is lazy
        ra = da.from_array(shear_cat.file[f"{group}/ra"], block_size)
        dec = da.from_array(shear_cat.file[f"{group}/dec"], block_size)
        psf_g1 = da.from_array(shear_cat.file[f"{group}/psf_g1"], block_size)
        psf_g2 = da.from_array(shear_cat.file[f"{group}/psf_g2"], block_size)
        weight = da.from_array(shear_cat.file[f"{group}/weight"], block_size)
        if shear_cat.catalog_type == "metacal":
            flag_name = "mcal_flags"
        else:
            flag_name = "flags"
        flag = da.from_array(shear_cat.file[f"{group}/{flag_name}"], block_size)
        b = da.from_array(shear_tomo.file["tomography/bin"], block_size)
        
        # collate metadata
        metadata = {
            key: self.config[key]
            for key in map_config_options
        }
        metadata["flag_exponent_max"] = flag_exponent_max
        metadata['nbin'] = nbin
        metadata['nbin_source'] = nbin


        for i in bins:
            if i == "all":
                w = b >= 0
            else:
                w = b == i

            count_map, g1_map, g2_map, weight_map, esq_map, var1_map, var2_map = make_dask_maps(
                ra[w], dec[w], psf_g1[w], psf_g2[w], weight[w], pixel_scheme)
            
            pix = da.where(weight_map > 0)[0]

            # Change output name
            if i == "all":
                i = "2D"

            maps[f"psf/counts_{i}"] = (pix, count_map[pix])
            maps[f"psf/g1_{i}"] = (pix, g1_map[pix])
            maps[f"psf/g2_{i}"] = (pix, g2_map[pix])
            maps[f"psf/var_g2_{i}"] = (pix, var1_map[pix])
            maps[f"psf/var_g2_{i}"] = (pix, var2_map[pix])
            maps[f"psf/var_{i}"] = (pix, esq_map[pix])
            maps[f"psf/lensing_weight_{i}"] = (pix, weight_map[pix])

        # Now add the flag maps. These are not tomographic.
        pix, flag_maps = make_dask_flag_maps(ra, dec, flag, flag_exponent_max, pixel_scheme)
        for j in range(flag_exponent_max):
            maps[f"flags/flag_{2**j}"] = (pix, flag_maps[j][pix])


        maps, = dask.compute(maps)

        # Print out some info about the flag maps
        for i in range(flag_exponent_max):
            f = 2**i
            count = maps[f"flags/flag_{f}"][1].sum()
            print(f"Map shows total {count} objects with flag {f}")

        # write the output maps
        with self.open_output("aux_source_maps", wrapper=True) as out:
            for map_name, (pix, m) in maps.items():
                out.write_map(map_name, pix, m, metadata)
            out.file['maps'].attrs.update(metadata)


class TXAuxiliaryLensMaps(TXBaseMaps):
    """
    Generate auxiliary maps from the lens catalog

    This class generates maps of:
        - depth
        - psf
        - bright object counts
        - flags
    """
    name = "TXAuxiliaryLensMaps"
    inputs = [
        ("photometry_catalog", HDFFile),  # for mags etc
    ]
    outputs = [
        ("aux_lens_maps", MapsFile),
    ]

    config_options = {
        "chunk_rows": 100_000,
        "sparse": True,
        "bright_obj_threshold": 22.0,  # The magnitude threshold for a object to be counted as bright
        "depth_band": "i",  # Make depth maps for this band
        "snr_threshold": 10.0,  # The S/N value to generate maps for (e.g. 5 for 5-sigma depth)
        "snr_delta": 1.0,  # The range threshold +/- delta is used for finding objects at the boundary
    }

    # we dont redefine choose_pixel_scheme for lens_maps to prevent circular modules

    def prepare_mappers(self, pixel_scheme):
        # We make a suite of mappers here.

        # for estimating depth per lens-bin
        depth_mapper = DepthMapperDR1(
            pixel_scheme,
            self.config["snr_threshold"],
            self.config["snr_delta"],
            sparse=self.config["sparse"],
            comm=self.comm,
        )

        # for mapping bright star fractions, for masks
        brobj_mapper = BrightObjectMapper(
            pixel_scheme,
            self.config["bright_obj_threshold"],
            sparse=self.config["sparse"],
            comm=self.comm,
        )

        return depth_mapper, brobj_mapper

    def data_iterator(self):
        band = self.config["depth_band"]
        cols = ["ra", "dec", "extendedness", f"snr_{band}", f"mag_{band}"]
        return self.iterate_hdf(
            "photometry_catalog", "photometry", cols, self.config["chunk_rows"]
        )

    def accumulate_maps(self, pixel_scheme, data, mappers):
        depth_mapper, brobj_mapper = mappers
        band = self.config["depth_band"]

        # Our different mappers want different data columns.
        # We pull out the bits they need and give them just those.
        brobj_data = {
            "mag": data[f"mag_{band}"],
            "extendedness": data["extendedness"],
            "ra": data["ra"],
            "dec": data["dec"],
        }

        depth_data = {
            "mag": data[f"mag_{band}"],
            "snr": data[f"snr_{band}"],
            "ra": data["ra"],
            "dec": data["dec"],
        }

        depth_mapper.add_data(depth_data)
        brobj_mapper.add_data(brobj_data)

    def finalize_mappers(self, pixel_scheme, mappers):
        depth_mapper, brobj_mapper = mappers

        # Four different mappers
        depth_pix, depth_count, depth, depth_var = depth_mapper.finalize(self.comm)
        brobj_pix, brobj_count, brobj_mag, brobj_mag_var = brobj_mapper.finalize(
            self.comm
        )

        # Collect all the maps
        maps = {}

        if self.rank != 0:
            return maps

        # Save depth maps
        maps["aux_lens_maps", "depth/depth"] = (depth_pix, depth)
        maps["aux_lens_maps", "depth/depth_count"] = (depth_pix, depth_count)
        maps["aux_lens_maps", "depth/depth_var"] = (depth_pix, depth_var)

        # Save bright object counts
        maps["aux_lens_maps", "bright_objects/count"] = (brobj_pix, brobj_count)

        return maps


class TXUniformDepthMap(PipelineStage):
    """
    Generate a uniform depth map from the mask

    This is useful for testing on uniform patches.
    It doesn't generate all the other maps that the other stages that
    make aux_lens_maps do, so may not always be useful.
    """
    name = "TXUniformDepthMap"
    parallel = False
    # make a mask from the auxiliary maps
    inputs = [("mask", MapsFile)]
    outputs = [("aux_lens_maps", MapsFile)]
    config_options = {
        "depth": 25.0,
    }

    def run(self):
        import healpy

        with self.open_input("mask", wrapper=True) as f:
            metadata = dict(f.file["maps/mask"].attrs)
            mask = f.read_mask()
            pix = f.file["maps/mask/pixel"][:]

        # Make a fake depth map
        depth = mask.copy()
        depth[pix] = self.config["depth"]  # e.g. 25 everywhere

        with self.open_output("aux_lens_maps", wrapper=True) as f:
            f.file.create_group("depth")
            print(len(pix))
            print(len(depth[pix]))
            f.write_map("depth/depth", pix, depth[pix], metadata)
