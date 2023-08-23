from .base_stage import PipelineStage
from .data_types import TomographyCatalog, MapsFile, HDFFile, ShearCatalog
import numpy as np
from .utils import unique_list, choose_pixelization, rename_iterated
from .utils.calibration_tools import read_shear_catalog_type, apply_metacal_response
from .utils.calibrators import Calibrator
from .mapping import ShearMapper, LensMapper, FlagMapper


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
            output_file.file["maps"].attrs.update(self.config)

        # same the relevant maps in each
        for (tag, map_name), (pix, map_data) in maps.items():
            output_files[tag].write_map(map_name, pix, map_data, self.pixel_metadata)



class TXSourceMaps(PipelineStage):
    """Generate source maps directly from binned, calibrated shear catalogs.

    This is a lazy implementation - we just use the parent class and
    replace the input and calibration steps.  It could be improved.
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
        import dask
        import dask.array as da
        import healpy

        # Configuration options
        pixel_scheme = choose_pixelization(**self.config)
        nside = self.config["nside"]
        block_size = self.config["block_size"]
        if block_size == 0:
            block_size = "auto"

        # We have to keep this open throughout the process, because
        # dask will internally load chunks of the input hdf5 data.
        f = self.open_input("binned_shear_catalog")
        nbin = f['shear'].attrs['nbin_source']

        npix = healpy.nside2npix(nside)

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

            # This seems to work directly, but we should check performance
            pix = pixel_scheme.ang2pix(ra, dec)

            # count map is just the number of galaxies per pixel
            count_map = da.bincount(pix, minlength=npix)

            # For the other map we use bincount with weights - these are the
            # various maps by pixel. bincount gives the number of objects in each
            # vaue of the first argument, weighted by the weights keyword, so effectively
            # it gives us
            # p_i = sum_{j} x[j] * delta_{pix[j], i}
            # which is out map
            weight_map = da.bincount(pix, weights=weight, minlength=npix)
            g1_map = da.bincount(pix, weights=weight * g1, minlength=npix)
            g2_map = da.bincount(pix, weights=weight * g2, minlength=npix)
            esq_map = da.bincount(pix, weights=weight**2 * 0.5 * (g1**2 + g2**2), minlength=npix)

            # normalize by weights where we want a mean
            hit = da.where(weight_map > 0)
            g1_map[hit] /= weight_map[hit]
            g2_map[hit] /= weight_map[hit]

            # Generate a catalog-like vector of the means so we can
            # subtract from the full catalog.  Not sure if this ever actually gets
            # created.
            g1_mean = g1_map[pix]
            g2_mean = g2_map[pix]

            # Also generate variance maps
            var1_map = da.bincount(pix, weights=weight * (g1 - g1_mean)**2, minlength=npix)
            var2_map = da.bincount(pix, weights=weight * (g2 - g2_mean)**2, minlength=npix)

            # we want the variance on the mean, so we divide by both the weight
            # (to go from the sum to the variance) and then by the count (to get the
            # variance on the mean). Have verified that this is the same as using
            # var() on the original arrays.
            var1_map[hit] /= (weight_map[hit] * count_map[hit])
            var2_map[hit] /= (weight_map[hit] * count_map[hit])
            
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
        

        # Everything above is lazy - this forces the computation.
        # It works out an efficient (we hope) way of doing everything in parallel
        output, = dask.compute(output)
        f.close()

        # collate metadata
        metadata = {
            key: self.config[key]
            for key in map_config_options
        }

        # write the output maps
        with self.open_output("source_maps", wrapper=True) as out:
            for i in bins:

                # again rename "all" to "2D"
                if i == "all":
                    i = "2D"

                # use the lensing weight to decide which pixels to write
                # - we skip the empty ones so they read in as healpy.UNSEEN
                m = output[f"lensing_weight_{i}"]
                pix = np.where(m != 0)[0]
                out.write_map(f"lensing_weight_{i}", pix, m[pix], metadata)
                for key in "g1", "g2", "count", "var_e", "var_g1", "var_g2":
                    out.write_map(f"{key}_{i}", pix, output[f"{key}_{i}"][pix], metadata)




class TXLensMaps(TXBaseMaps):
    """
    Make tomographic lens number count maps

    Uses photometry and lens tomography catalogs.

    Density maps are made later once masks are generated.
    """

    name = "TXLensMaps"

    inputs = [
        ("photometry_catalog", HDFFile),
        ("lens_tomography_catalog", TomographyCatalog),
    ]

    outputs = [
        ("lens_maps", MapsFile),
    ]

    config_options = {**map_config_options}

    def prepare_mappers(self, pixel_scheme):
        # read nbin_lens and save
        with self.open_input("lens_tomography_catalog") as f:
            nbin_lens = f["tomography"].attrs["nbin"]
        self.config["nbin_lens"] = nbin_lens

        # create lone mapper
        lens_bins = list(range(nbin_lens))
        source_bins = []
        mapper = LensMapper(
            pixel_scheme,
            lens_bins,
            sparse=self.config["sparse"],
        )
        return [mapper]

    def data_iterator(self):
        # see TXSourceMaps abov for info on this
        return self.combined_iterators(
            self.config["chunk_rows"],
            # first file
            "photometry_catalog",
            "photometry",
            ["ra", "dec"],
            # next file
            "lens_tomography_catalog",
            "tomography",
            ["bin", "lens_weight"],
        )

    def accumulate_maps(self, pixel_scheme, data, mappers):
        # no need to rename cols here since just ra, dec, lens_bin
        mapper = mappers[0]
        mapper.add_data(data)

    def finalize_mappers(self, pixel_scheme, mappers):
        # Again just the one mapper
        mapper = mappers[0]
        # Ignored return values are empty dicts for shear
        pix, ngal, weighted_ngal = mapper.finalize(self.comm)
        maps = {}

        if self.rank != 0:
            return maps

        for b in mapper.bins:
            # keys are the output tag and the map name
            maps["lens_maps", f"ngal_{b}"] = (pix, ngal[b])
            maps["lens_maps", f"weighted_ngal_{b}"] = (pix, weighted_ngal[b])

        return maps


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

    config_options = {**map_config_options}

    def data_iterator(self):
        # See TXSourceMaps for an explanation of thsis
        return self.combined_iterators(
            self.config["chunk_rows"],
            # first file
            "lens_catalog",
            "lens",
            ["ra", "dec"],
            # next file
            "lens_tomography_catalog",
            "tomography",
            ["bin", "lens_weight"],
            # another section in the same file
        )


class TXDensityMaps(PipelineStage):
    """
    Convert galaxy count maps to overdensity delta maps

    delta = (ngal - <ngal>) / <ngal>

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

    def run(self):
        import healpy

        # Read the mask
        with self.open_input("mask", wrapper=True) as f:
            mask = f.read_map("mask")

        # set unseen pixels to weight zero
        mask[mask == healpy.UNSEEN] = 0
        mask[np.isnan(mask)] = 0
        mask = mask.flatten()
        pix = np.where(mask > 0)[0]

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
            delta_map[mask > 0] = (
                ng[mask > 0] / mask[mask > 0]
            )  # Assuming that the weights do not include the mask
            mu = np.mean(delta_map[mask > 0])
            delta_map[mask > 0] = delta_map[mask > 0] / mu - 1
            density_maps.append(delta_map)

        # write output
        with self.open_output("density_maps", wrapper=True) as f:
            # create group and save metadata there too.
            f.file.create_group("maps")
            f.file["maps"].attrs.update(meta)
            # save each density map
            for i, rho in enumerate(density_maps):
                f.write_map(f"delta_{i}", pix, rho[pix], meta)
