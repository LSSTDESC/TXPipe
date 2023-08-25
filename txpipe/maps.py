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


class TXSourceMaps(TXBaseMaps):
    """
    Make tomographic shear maps

    Make g1, g2, var(g1), var(g2), and lensing weight maps
    from shear catalogs and tomography.

    Should be replaced to use the binned_shear_catalog since that's
    calibrated already.
    """

    name = "TXSourceMaps"

    inputs = [
        ("shear_catalog", ShearCatalog),
        ("shear_tomography_catalog", TomographyCatalog),
    ]

    outputs = [
        ("source_maps", MapsFile),
    ]

    # Generic mapping options + one option
    # to use the truth shear columns
    config_options = {"true_shear": False, **map_config_options}

    def prepare_mappers(self, pixel_scheme):
        # read shear cols and calibration info
        nbin_source, cal = self.get_calibrators()

        # store in config so it is saved later
        self.config["nbin_source"] = nbin_source
        # create basic mapper object
        source_bins = list(range(nbin_source))
        mapper = ShearMapper(
            pixel_scheme,
            source_bins,
            sparse=self.config["sparse"],
        )
        return [mapper, cal]

    def get_calibrators(self):
        shear_catalog_type = read_shear_catalog_type(self)
        cal, cal2D = Calibrator.load(self.get_input("shear_tomography_catalog"))
        nbin_source = len(cal)
        return nbin_source, [cal, cal2D]

    def data_iterator(self):

        # can optionally read truth values
        with self.open_input("shear_catalog", wrapper=True) as f:
            shear_cols, renames = f.get_primary_catalog_names(self.config["true_shear"])

        # use utility function that combines data chunks
        # from different files. Reading from n file sections
        # takes 3n+1 arguments
        it = self.combined_iterators(
            self.config["chunk_rows"],  # number of rows to iterate at once
            # first file info
            "shear_catalog",  # tag of input file to iterate through
            "shear",  # data group within file to look at
            shear_cols,  # column(s) to read
            # next file
            "shear_tomography_catalog",  # tag of input file to iterate through
            "tomography",  # data group within file to look at
            ["bin"],  # column(s) to read
        )
        return rename_iterated(it, renames)

    def accumulate_maps(self, pixel_scheme, data, mappers):
        mapper = mappers[0]
        mapper.add_data(data)

    def calibrate_maps(self, g1, g2, var_g1, var_g2, cals):
        import healpy

        # We will return lists of calibrated maps
        g1_out = {}
        g2_out = {}
        var_g1_out = {}
        var_g2_out = {}

        cals, cal_2D = cals

        # We calibrate the 2D case separately
        for i in g1.keys():
            # we want to avoid accidentally calibrating any pixels
            # that should be masked.
            mask = (
                (g1[i] == healpy.UNSEEN)
                | (g2[i] == healpy.UNSEEN)
                | (var_g1[i] == healpy.UNSEEN)
                | (var_g2[i] == healpy.UNSEEN)
            )
            if i == "2D":
                cal = cal_2D
            else:
                cal = cals[i]
            g1_out[i], g2_out[i] = cal.apply(g1[i], g2[i])
            std1 = np.sqrt(var_g1[i])
            std2 = np.sqrt(var_g2[i])
            std1, std2 = cal.apply(std1, std2, subtract_mean=False)
            var_g1_out[i] = std1**2
            var_g2_out[i] = std2**2

            # re-apply the masking, just to make sure
            for x in [g1_out[i], g2_out[i], var_g1_out[i], var_g2_out[i]]:
                x[mask] = healpy.UNSEEN

        return g1_out, g2_out, var_g1_out, var_g2_out

    def finalize_mappers(self, pixel_scheme, mappers):
        # only one mapper here - we call its finalize method
        # to collect everything
        mapper, cal = mappers
        pix, counts, g1, g2, var_g1, var_g2, weights_g, esq = mapper.finalize(self.comm)

        # build up output
        maps = {}

        # only master gets full stuff
        if self.rank != 0:
            return maps

        # Calibrate the maps
        g1, g2, var_g1, var_g2 = self.calibrate_maps(g1, g2, var_g1, var_g2, cal)

        for b in mapper.bins:
            # keys are the output tag and the map name
            maps["source_maps", f"g1_{b}"] = (pix, g1[b])
            maps["source_maps", f"g2_{b}"] = (pix, g2[b])
            maps["source_maps", f"var_g1_{b}"] = (pix, var_g1[b])
            maps["source_maps", f"var_g2_{b}"] = (pix, var_g2[b])
            maps["source_maps", f"lensing_weight_{b}"] = (pix, weights_g[b])
            maps["source_maps", f"count_{b}"] = (pix, counts[b])
            # added from HSC branch, to get analytic noise in twopoint_fourier
            out_e = np.zeros_like(esq[b])
            out_e[esq[b] > 0] = esq[b][esq[b] > 0]


            # calibrate the esq value - this is hacky for now!
            shear_catalog_type = read_shear_catalog_type(self)
            if (shear_catalog_type == "metadetect") or (shear_catalog_type == "metacal"):
                print("DOING HACKY CAL OF VAR(e)")
                cal_1D, cal_2D = cal
                if b == "2D":
                    c = cal_2D
                else:
                    c = cal_1D[b]
                Rinv_approx = c.Rinv.diagonal().mean()
                esq[b] *= Rinv_approx**2

                # replace saved quantity
                out_e = np.zeros_like(esq[b])
                out_e[esq[b] > 0] = esq[b][esq[b] > 0]
            else:
                print("VAR(e) IS WRONG!")

            maps["source_maps", f"var_e_{b}"] = (pix, out_e)

        return maps


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
        print("TODO: add use of lens weights here")
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
