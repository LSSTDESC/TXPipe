from .base_stage import PipelineStage
from .data_types import TomographyCatalog, MapsFile, HDFFile
import numpy as np
from .utils import unique_list, choose_pixelization, import_dask
from .mapping import  make_dask_shear_maps, make_dask_lens_maps


SHEAR_SHEAR = 0
SHEAR_POS = 1
POS_POS = 2

NON_TOMO_BIN = 999999

# These generic mapping options are used by multiple different
# map types.
# TODO: consider dropping support for gnomonic maps.
# Also consider adding support for pixell
map_config_options = {
    "chunk_rows": 100000,  # The number of rows to read in each chunk of data at a time
    "pixelization": "healpix",  # The pixelization scheme to use, currently just healpix
    "nside": 0,  # The Healpix resolution parameter for the generated maps. Only req'd if using healpix
    "sparse": True,  # Whether to generate sparse maps - faster and less memory for small sky areas,
    "ra_cent": np.nan,  # These parameters are only required if pixelization==tan
    "dec_cent": np.nan,
    "npix_x": -1,
    "npix_y": -1,
    "pixel_size": np.nan,  # Pixel size of pixelization scheme
}


class TXBaseMaps(PipelineStage):
    """
    A base class for mapping stages

    This is an abstract base class, which other subclasses
    inherit from to use the same basic structure, which is:
    - select pixelization
    - prepare some mapper objects
    - iterate through selected columns
        - update each mapper with each chunk
    - finalize the mappers
    - save the maps
    """

    name = "TXBaseMaps"
    inputs = []
    outputs = []
    config_options = {}

    def run(self):
        # Read input configuration information
        # and select pixel scheme. Also save the scheme
        # metadata so we can save it later
        pixel_scheme = self.choose_pixel_scheme()
        self.pixel_metadata = pixel_scheme.metadata
        self.config.update(self.pixel_metadata)

        # Initialize our maps
        mappers = self.prepare_mappers(pixel_scheme)

        # Loop through the data
        for s, e, data in self.data_iterator():
            # Give an idea of progress
            print(f"Process {self.rank} read data chunk {s:,} - {e:,}")
            # Build up any maps we are using
            self.accumulate_maps(pixel_scheme, data, mappers)

        # Finalize and output the maps
        maps = self.finalize_mappers(pixel_scheme, mappers)
        if self.rank == 0:
            self.save_maps(pixel_scheme, maps)

    def choose_pixel_scheme(self):
        """
        Subclasses can override to instead load pixelization
        from an existing map
        """
        return choose_pixelization(**self.config)

    def prepare_mappers(self, pixel_scheme):
        """
        Subclasses must override to init any mapper objects
        """
        raise RuntimeError("Do not use TXBaseMaps - use a subclass")

    def data_iterator(self):
        """
        Subclasses must override to create an iterator looping over
        input data
        """
        raise RuntimeError("Do not use TXBaseMaps - use a subclass")

    def accumulate_maps(self, pixel_scheme, data, mappers):
        """
        Subclasses must override to supply the next chunk "data" to
        their mappers
        """
        raise RuntimeError("Do not use TXBaseMaps - use a subclass")

    def finalize_mappers(self, pixel_scheme, mappers):
        """
        Subclasses must override to finalize their maps and return
        a dictionary of (output_tag, map_name) -> (pixels, values)
        """
        raise RuntimeError("Do not use TXBaseMaps - use a subclass")

    def save_maps(self, pixel_scheme, maps):
        """
        Subclasses can use this directly, by generating maps as described
        in finalize_mappers
        """
        # Find and open all output files
        tags = unique_list([outfile for outfile, _ in maps.keys()])
        output_files = {
            tag: self.open_output(tag, wrapper=True)
            for tag in tags
            if tag.endswith("maps")
        }

        # add a maps section to each
        for output_file in output_files.values():
            output_file.file.create_group("maps")
            
            #this is a bit of a hack, but if you have set an alias 
            #it can't be added as an attr to the hdf5 file
            #this saves everythinkg but aliases
            from ceci.config import StageConfig
            config_no_alias = {}
            for k in self.config.keys():
                if k == 'aliases':
                    continue
                config_no_alias[k] = self.config[k]
            config_no_alias = StageConfig(**config_no_alias)

            #output_file.file["maps"].attrs.update(self.config)
            output_file.file["maps"].attrs.update(config_no_alias)

        # same the relevant maps in each
        for (tag, map_name), (pix, map_data) in maps.items():
            output_files[tag].write_map(map_name, pix, map_data, self.pixel_metadata)



class TXSourceMaps(PipelineStage):
    """Generate source maps directly from binned, calibrated shear catalogs.

    This implementation uses DASK, which offers a numpy-like syntax and
    hides the complicated parallelization details.
    
    """
    name = "TXSourceMaps"
    dask_parallel = True

    inputs = [
        ("binned_shear_catalog", HDFFile),
    ]

    outputs = [
        ("source_maps", MapsFile),
    ]

    config_options = {
        "block_size": 0,
        **map_config_options
    }

    def run(self):
        dask, da = import_dask()
        import healpy

        # Configuration options
        pixel_scheme = choose_pixelization(**self.config)
        nside = self.config["nside"]
        npix = healpy.nside2npix(nside)
        block_size = self.config["block_size"]
        if block_size == 0:
            block_size = "auto"

        # We have to keep this open throughout the process, because
        # dask will internally load chunks of the input hdf5 data.
        f = self.open_input("binned_shear_catalog")
        nbin = f['shear'].attrs['nbin_source']

        # The "all" bin is the non-tomographic case.
        bins = list(range(nbin)) + ["all"]
        output = {}

        for i in bins:
            # These don't actually load all the data - everything is lazy
            ra = da.from_array(f[f"shear/bin_{i}/ra"], block_size)
            dec = da.from_array(f[f"shear/bin_{i}/dec"], block_size)
            g1 = da.from_array(f[f"shear/bin_{i}/g1"], block_size)
            g2 = da.from_array(f[f"shear/bin_{i}/g2"], block_size)
            weight = da.from_array(f[f"shear/bin_{i}/weight"], block_size)

            count_map, g1_map, g2_map, weight_map, esq_map, var1_map, var2_map = make_dask_shear_maps(
                ra, dec, g1, g2, weight, pixel_scheme)

            # slight change in output name
            if i == "all":
                i = "2D"


            # Save all the stuff we want here.
            output[f"count_{i}"] = count_map
            output[f"g1_{i}"] = g1_map
            output[f"g2_{i}"] = g2_map
            output[f"lensing_weight_{i}"] = weight_map
            output[f"count_{i}"] = count_map
            output[f"var_e_{i}"] = esq_map
            output[f"var_g1_{i}"] = var1_map
            output[f"var_g2_{i}"] = var2_map
        
        # mask is where a pixel is hit in any of the tomo bins
        mask = da.zeros(npix, dtype=bool)
        for i in bins:
            if i == "all":
                i = "2D"
            mask |= output[f"lensing_weight_{i}"] > 0

        output["mask"] = mask

        # Everything above is lazy - this forces the computation.
        # It works out an efficient (we hope) way of doing everything in parallel
        output, = dask.compute(output)
        f.close()

        # collate metadata
        metadata = {
            key: self.config[key]
            for key in map_config_options
        }
        metadata['nbin'] = nbin
        metadata['nbin_source'] = nbin
        metadata.update(pixel_scheme.metadata)

        pix = np.where(output["mask"])[0]

        # write the output maps
        with self.open_output("source_maps", wrapper=True) as out:
            for i in bins:
                # again rename "all" to "2D"
                if i == "all":
                    i = "2D"

                # We save the pixels in the mask - i.g. any pixel that is hit in any
                # tomographic bin is included. Some will be UNSEEN.
                for key in "g1", "g2", "count", "var_e", "var_g1", "var_g2", "lensing_weight":
                    out.write_map(f"{key}_{i}", pix, output[f"{key}_{i}"][pix], metadata)

            out.file['maps'].attrs.update(metadata)


class TXLensMaps(PipelineStage):
    """
    Make tomographic lens number count maps

    Uses photometry and lens tomography catalogs.

    Density maps are made later once masks are generated.
    """

    name = "TXLensMaps"
    dask_parallel = True

    inputs = [
        ("photometry_catalog", HDFFile),
        ("lens_tomography_catalog", TomographyCatalog),
    ]
    outputs = [
        ("lens_maps", MapsFile),
    ]

    config_options = {
        "block_size": 0,
        **map_config_options
    }

    def ra_dec_inputs(self):
        return "photometry_catalog", "photometry"


    def run(self):
        import healpy
        _, da = import_dask()



        # The subclass reads the ra and dec of the lenses
        # from a different input file, so we allow that here
        # using this method which is overwrites
        cat_name, cat_group = self.ra_dec_inputs()

        # open our two input files. They will be read lazily
        tomo_cat = self.open_input("lens_tomography_catalog", wrapper=True)
        photo_cat = self.open_input(cat_name, wrapper=True)

        nbin_lens = tomo_cat.read_nbin()
        self.config["nbin_lens"] = nbin_lens


        # Other config info.
        pixel_scheme = choose_pixelization(**self.config)
        block_size = self.config["block_size"]
        if block_size == 0:
            block_size = "auto"
        
        # Lazily open input data sets
        ra = da.from_array(photo_cat.file[f"{cat_group}/ra"], block_size)
        dec = da.from_array(photo_cat.file[f"{cat_group}/dec"], block_size)
        weight = da.from_array(tomo_cat.file["tomography/lens_weight"], block_size)
        tomo_bin = da.from_array(tomo_cat.file["tomography/bin"], block_size)

        # bins to generate maps for
        bins = list(range(nbin_lens)) + ["2D"]
        maps = {}

        # Generate the maps with dask. This is lazy until we do da.compute
        for b in bins:
            pix, count_map, weight_map = make_dask_lens_maps(ra, dec, weight, tomo_bin, b, pixel_scheme)
            maps[f"ngal_{b}"] = (pix, count_map[pix])
            maps[f"weighted_ngal_{b}"] = (pix, weight_map[pix])

        # Actually run everything
        maps, = da.compute(maps)

        # collate metadata
        metadata = {
            key: self.config[key]
            for key in map_config_options
        }
        metadata['nbin'] = nbin_lens
        metadata['nbin_lens'] = nbin_lens
        metadata.update(pixel_scheme.metadata)

        # Save all the maps
        with self.open_output("lens_maps", wrapper=True) as out:
            for name, (pix, m) in maps.items():
                out.write_map(name, pix, m, metadata)
            out.file['maps'].attrs.update(metadata)


class TXExternalLensMaps(TXLensMaps):
    """
    Make tomographic lens number count maps from external data

    Same as TXLensMaps except it reads from an external
    lens catalog.
    """

    name = "TXExternalLensMaps"

    inputs = [
        ("lens_catalog", HDFFile),
        ("lens_tomography_catalog", TomographyCatalog),
    ]

    def ra_dec_inputs(self):
        return "lens_catalog", "lens"






class TXDensityMaps(PipelineStage):
    """
    Convert galaxy count maps to overdensity delta maps

    delta = ngal / (weight * <ngal>/<weight>) - 1

    This has to be separate from the lens mappers above
    because it requires the mask, which is created elsewhere
    (right now in masks.py)
    """

    name = "TXDensityMaps"
    parallel = False
    inputs = [
        ("lens_maps", MapsFile),
        ("mask", MapsFile),
    ]
    outputs = [
        ("density_maps", MapsFile),
    ]
    config_options = {
        "mask_threshold": 0.0
    }

    def run(self):
        import healpy

        # Read the mask and set all pixels below the threshold to 0
        with self.open_input("mask", wrapper=True) as f:
            mask = f.read_mask(thresh=self.config["mask_threshold"])
        
        #identify unmasked pixels
        pix_keep = mask > 0.
        pix = np.where(pix_keep)[0]

        # Read the count maps
        with self.open_input("lens_maps", wrapper=True) as f:
            meta = dict(f.file["maps"].attrs)
            nbin_lens = meta["nbin_lens"]
            ngal_maps = [
                f.read_map(f"weighted_ngal_{b}").flatten() for b in range(nbin_lens)
            ]

        # Convert count maps into density maps
        density_maps = []
        for i, ng in enumerate(ngal_maps):
            ng[np.isnan(ng)] = 0.0
            ng[ng == healpy.UNSEEN] = 0
            delta_map = np.zeros(mask.shape, dtype=np.float64)
            #calculate mean of ng and mean of mask
            mu_n = np.mean(ng[pix_keep])
            mu_w = np.mean(mask[pix_keep])
            delta_map[pix_keep] = (ng[pix_keep] / (mask[pix_keep] * mu_n / mu_w)) - 1
            density_maps.append(delta_map)

        # write output
        with self.open_output("density_maps", wrapper=True) as f:
            # create group and save metadata there too.
            f.file.create_group("maps")
            f.file["maps"].attrs.update(meta)
            # save each density map
            for i, rho in enumerate(density_maps):
                f.write_map(f"delta_{i}", pix, rho[pix], meta)