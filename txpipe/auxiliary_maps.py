from .maps import TXBaseMaps, map_config_options
import numpy as np
from .base_stage import PipelineStage
from .mapping import Mapper, FlagMapper, BrightObjectMapper, DepthMapperDR1
from .data_types import MapsFile, HDFFile, ShearCatalog
from .utils import choose_pixelization, rename_iterated, read_shear_catalog_type


class TXAuxiliarySourceMaps(TXBaseMaps):
    """
    Generate auxiliary maps from the source catalog

    This stage makes maps of:
    - the count of different flag values
    - the mean PSF

    These are currently only used for making visualizations in the later TXMapPlots
    stage, and are not otherwise used directly.

    Like most of the mapping stages it inherits most behavior from the TXBaseMaps
    parent class, which specifies the primary `run` method. This is because most
    mapper classes have the same overall structure. See that class for more details.
    """

    name = "TXAuxiliarySourceMaps"
    inputs = [
        ("shear_catalog", ShearCatalog),  # for psfs
        ("shear_tomography_catalog", HDFFile),  # for per-bin psf maps
        ("source_maps", MapsFile),  # we copy the pixel scheme from here
    ]
    outputs = [
        ("aux_source_maps", MapsFile),
    ]

    config_options = {
        "chunk_rows": 100_000,
        "sparse": True,
        "flag_exponent_max": 8,  # flag bits go up to 2**8 by default
        "psf_prefix": "psf_",  # prefix name for columns
    }

    def choose_pixel_scheme(self):
        with self.open_input("source_maps", wrapper=True) as maps_file:
            pix_info = dict(maps_file.file["maps"].attrs)
        return choose_pixelization(**pix_info)

    def prepare_mappers(self, pixel_scheme):
        # We make a suite of mappers here.
        # We read nbin_source because we want PSF maps per-bin
        with self.open_input("shear_tomography_catalog") as f:
            nbin_source = f["tomography"].attrs["nbin_source"]
        self.config["nbin_source"] = nbin_source  # so it gets saved later
        source_bins = list(range(nbin_source))

        # For making psf_g1, psf_g2 maps, per source-bin
        psf_mapper = Mapper(
            pixel_scheme, [], source_bins, do_lens=False, sparse=self.config["sparse"]
        )

        # for mapping the density of flagged objects
        flag_mapper = FlagMapper(
            pixel_scheme, self.config["flag_exponent_max"], sparse=self.config["sparse"]
        )

        return psf_mapper, flag_mapper

    def data_iterator(self):
        psf_prefix = self.config["psf_prefix"]
        shear_catalog_type = read_shear_catalog_type(self)

        # Flag column name depends on catalog type
        if shear_catalog_type == "metacal":
            shear_cols = [
                f"{psf_prefix}g1",
                f"{psf_prefix}g2",
                "mcal_flags",
                "weight",
                "ra",
                "dec",
            ]
            renames = {
                f"{psf_prefix}g1": "psf_g1",
                f"{psf_prefix}g2": "psf_g2",
                "mcal_flags": "flags",
            }
        elif shear_catalog_type == "metadetect":
            shear_cols = [
                f"00/{psf_prefix}g1",
                f"00/{psf_prefix}g2",
                "00/flags",
                "00/weight",
                "00/ra",
                "00/dec",
            ]
            renames = {
                f"00/{psf_prefix}g1": "psf_g1",
                f"00/{psf_prefix}g2": "psf_g2",
                "00/flags": "flags",
                "00/weight": "weight",
                "00/ra": "ra",
                "00/dec": "dec",
            }
        else:
            shear_cols = [
                f"{psf_prefix}g1",
                f"{psf_prefix}g2",
                "flags",
                "weight",
                "ra",
                "dec",
            ]
            renames = {
                f"{psf_prefix}g1": "psf_g1",
                f"{psf_prefix}g2": "psf_g2",
            }

        # See maps.py for an explanation of this
        it = self.combined_iterators(
            self.config["chunk_rows"],
            # first file
            "shear_catalog",
            "shear",
            shear_cols,
            # next file
            "shear_tomography_catalog",
            "tomography",
            ["source_bin"],
        )

        return rename_iterated(it, renames)

    def accumulate_maps(self, pixel_scheme, data, mappers):
        psf_mapper, flag_mapper = mappers

        psf_prefix = self.config["psf_prefix"]

        # Our different mappers want different data columns.
        # We pull out the bits they need and give them just those.

        psf_data = {
            "g1": data["psf_g1"],
            "g2": data["psf_g2"],
            "ra": data["ra"],
            "dec": data["dec"],
            "source_bin": data["source_bin"],
            "weight": data["weight"],
        }

        flag_data = {
            "ra": data["ra"],
            "dec": data["dec"],
            "flags": data["flags"],
        }

        psf_mapper.add_data(psf_data)
        flag_mapper.add_data(flag_data)

    def finalize_mappers(self, pixel_scheme, mappers):
        psf_mapper, flag_mapper = mappers

        # Four different mappers
        pix, _, _, g1, g2, var_g1, var_g2, weight, _ = psf_mapper.finalize(self.comm)
        flag_pixs, flag_maps = flag_mapper.finalize(self.comm)

        # Collect all the maps
        maps = {}

        if self.rank != 0:
            return maps

        # Save PSF maps
        for b in psf_mapper.source_bins:
            maps["aux_source_maps", f"psf/g1_{b}"] = (pix, g1[b])
            maps["aux_source_maps", f"psf/g2_{b}"] = (pix, g1[b])
            maps["aux_source_maps", f"psf/var_g2_{b}"] = (pix, var_g1[b])
            maps["aux_source_maps", f"psf/var_g2_{b}"] = (pix, var_g1[b])
            maps["aux_source_maps", f"psf/lensing_weight_{b}"] = (pix, weight[b])

        # Save flag maps
        for i, (p, m) in enumerate(zip(flag_pixs, flag_maps)):
            f = 2**i
            maps["aux_source_maps", f"flags/flag_{f}"] = (p, m)
            # also print out some stats
            t = m.sum()
            print(f"Map shows total {t} objects with flag {f}")

        return maps


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
        ("lens_maps", MapsFile),  # we copy the pixel scheme from here
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
    # instead of reading from config we match the basic maps
    def choose_pixel_scheme(self):
        with self.open_input("lens_maps", wrapper=True) as maps_file:
            pix_info = dict(maps_file.file["maps"].attrs)

        return choose_pixelization(**pix_info)

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
