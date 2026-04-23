from .maps import TXBaseMaps, map_config_options
import numpy as np
from .base_stage import PipelineStage
from .mapping import (
    make_coverage_map,
    make_dask_shear_maps,
    make_dask_flag_maps,
    make_dask_bright_object_map,
    make_dask_depth_map,
    make_dask_depth_map_det_prob,
    make_dask_selection_function,
)
from .data_types import MapsFile, HDFFile, ShearCatalog
from .utils import choose_pixelization, import_dask
from ceci.config import StageParameter


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
        "block_size": StageParameter(int, 0, msg="Block size for dask processing (0 means auto)."),
        "flag_exponent_max": StageParameter(int, 8, msg="Maximum exponent for flag bits (default 8)."),
        "psf_prefix": StageParameter(str, "psf_", msg="Prefix for PSF column names."),
        **map_config_options,
    }

    def choose_pixel_scheme(self):
        with self.open_input("source_maps", wrapper=True) as maps_file:
            pix_info = dict(maps_file.file["maps"].attrs)
        return choose_pixelization(**pix_info)

    def run(self):
        dask, da = import_dask()
        import healsparse as hsp

        pixel_scheme = self.choose_pixel_scheme()
        block_size = self.config["block_size"]
        if block_size == 0:
            block_size = "auto"
        psf_prefix = self.config["psf_prefix"]

        flag_exponent_max = self.config["flag_exponent_max"]

        # We have to keep this open throughout the process, because
        # dask will internally load chunks of the input hdf5 data.
        shear_cat = self.open_input("shear_catalog", wrapper=True)
        shear_tomo = self.open_input("shear_tomography_catalog", wrapper=True)
        nbin = shear_tomo.file["tomography"].attrs["nbin"]

        # The "all" bin is the non-tomographic case.
        bins = list(range(nbin)) + ["all"]
        maps = {}
        group = shear_cat.get_primary_catalog_group()

        # These don't actually load all the data - everything is lazy
        ra = da.from_array(shear_cat.file[f"{group}/ra"], block_size)
        # force all columns to use the same block size
        block_size = ra.chunksize
        dec = da.from_array(shear_cat.file[f"{group}/dec"], block_size)
        psf_g1 = da.from_array(shear_cat.file[f"{group}/{psf_prefix}g1"], block_size)
        psf_g2 = da.from_array(shear_cat.file[f"{group}/{psf_prefix}g2"], block_size)
        weight = da.from_array(shear_cat.file[f"{group}/weight"], block_size)
        if shear_cat.catalog_type == "metacal":
            flag_name = "mcal_flags"
        else:
            flag_name = "flags"
        flag = da.from_array(shear_cat.file[f"{group}/{flag_name}"], block_size)
        b = da.from_array(shear_tomo.file["tomography/bin"], block_size)

        # collate metadata
        metadata = {key: self.config[key] for key in map_config_options}
        metadata["flag_exponent_max"] = flag_exponent_max
        metadata["nbin"] = nbin
        metadata["nbin_source"] = nbin
        metadata.update(pixel_scheme.metadata)

        cov_map = make_coverage_map(ra, dec, pixel_scheme)

        for i in bins:
            if i == "all":
                w = b >= 0
            else:
                w = b == i

            shear_map_results = make_dask_shear_maps(
                ra[w], dec[w], psf_g1[w], psf_g2[w], weight[w], pixel_scheme, cov_map
            )

            # Change output name
            if i == "all":
                i = "2D"

            maps[f"psf/count_{i}"] = shear_map_results["count_map"]
            maps[f"psf/g1_{i}"] = shear_map_results["g1_map"]
            maps[f"psf/g2_{i}"] = shear_map_results["g2_map"]
            maps[f"psf/var_g1_{i}"] = shear_map_results["var1_map"]
            maps[f"psf/var_g2_{i}"] = shear_map_results["var2_map"]
            maps[f"psf/var_e_{i}"] = shear_map_results["esq_map"]
            maps[f"psf/lensing_weight_{i}"] = shear_map_results["weight_map"]

        # Now add the flag maps. These are not tomographic.
        flag_maps = make_dask_flag_maps(
            ra, dec, flag, flag_exponent_max, pixel_scheme, cov_map
        )
        for j in range(flag_exponent_max):
            maps[f"flags/flag_{2**j}"] = flag_maps[j]

        (maps,) = dask.compute(maps)

        # Print out some info about the flag maps
        for i in range(flag_exponent_max):
            f = 2**i
            count = maps[f"flags/flag_{f}"].sum()
            print(f"Map shows total {count} objects with flag {f}")

        # convert sparse_map arrays into healsparse map objects
        hsp_maps = {}
        for name, map in maps.items():
            hsp_maps[name] = hsp.HealSparseMap(
                cov_map=cov_map, sparse_map=map, nside_sparse=cov_map.nside_sparse
            )

        # write the output maps
        with self.open_output("aux_source_maps", wrapper=True) as out:
            for map_name, hsp_map in hsp_maps.items():
                out.write_map(map_name, hsp_map, metadata)
            out.file["maps"].attrs.update(metadata)


class TXAuxiliaryLensMaps(TXBaseMaps):
    """
    Generate auxiliary maps from the lens catalog

    This class generates maps of:
        - depth
        - bright object counts
    """

    name = "TXAuxiliaryLensMaps"
    dask_parallel = True
    inputs = [
        ("photometry_catalog", HDFFile),  # for mags etc
    ]
    outputs = [
        ("aux_lens_maps", MapsFile),
    ]

    config_options = {
        "block_size": StageParameter(int, 0, msg="Block size for dask processing (0 means auto)."),
        "bright_obj_threshold": StageParameter(float, 22.0, msg="Magnitude threshold for bright objects."),
        "depth_band": StageParameter(str, "i", msg="Band for depth maps."),
        "snr_threshold": StageParameter(float, 10.0, msg="S/N value for depth maps."),
        "snr_delta": StageParameter(float, 1.0, msg="Delta for S/N thresholding."),
    }

    def run(self):
        # Import dask and alias it as 'da'
        _, da = import_dask()
        import healsparse as hsp

        # Retrieve configuration parameters
        block_size = self.config["block_size"]
        if block_size == 0:
            block_size = "auto"
        band = self.config["depth_band"]

        # Open the input photometry catalog file.
        # We can't use a "with" statement because we need to keep the file open
        # while we're using dask.
        f = self.open_input("photometry_catalog", wrapper=True)

        # Load photometry data into dask arrays.
        # This is lazy in dask, so we're not actually loading the data here.
        ra = da.from_array(f.file["photometry/ra"], block_size)
        block_size = ra.chunksize
        dec = da.from_array(f.file["photometry/dec"], block_size)
        extendedness = da.from_array(f.file["photometry/extendedness"], block_size)
        snr = da.from_array(f.file[f"photometry/snr_{band}"], block_size)
        mag = da.from_array(f.file[f"photometry/mag_{band}"], block_size)

        # Choose the pixelization scheme based on the configuration.
        # Might need to review this to make sure we use the same scheme everywhere
        pixel_scheme = choose_pixelization(**self.config)
        assert pixel_scheme.nest

        cov_map = make_coverage_map(ra, dec, pixel_scheme)

        # Initialize a dictionary to store the maps.
        # To start with this is all lazy too, until we call compute
        maps = {}

        # Create a bright object map using dask
        bright_object_results = make_dask_bright_object_map(
            ra,
            dec,
            mag,
            extendedness,
            self.config["bright_obj_threshold"],
            pixel_scheme,
            cov_map,
        )
        maps["bright_objects/count"] = bright_object_results["count"]

        # Create depth maps using dask
        depth_map_results = make_dask_depth_map(
            ra,
            dec,
            mag,
            snr,
            self.config["snr_threshold"],
            self.config["snr_delta"],
            pixel_scheme,
            cov_map,
        )
        maps["depth/depth"] = depth_map_results["depth_map"]
        maps["depth/depth_count"] = depth_map_results["count_map"]
        maps["depth/depth_var"] = depth_map_results["depth_var"]

        (maps,) = da.compute(maps)

        # convert sparse_map arrays into healsparse map objects
        hsp_maps = {}
        for name, map in maps.items():
            hsp_maps[name] = hsp.HealSparseMap(
                cov_map=cov_map, sparse_map=map, nside_sparse=cov_map.nside_sparse
            )

        # Prepare metadata for the maps. Copy the pixelization-related
        # configuration options only here
        metadata = {key: self.config[key] for key in map_config_options if key in self.config}
        # Then add the other configuration options
        metadata["depth_band"] = band
        metadata["depth_snr_threshold"] = self.config["snr_threshold"]
        metadata["depth_snr_delta"] = self.config["snr_delta"]
        metadata["bright_obj_threshold"] = self.config["bright_obj_threshold"]
        metadata.update(pixel_scheme.metadata)

        # Write the output maps to the output file
        with self.open_output("aux_lens_maps", wrapper=True) as out:
            for map_name, hsp_map in hsp_maps.items():
                out.write_map(map_name, hsp_map, metadata)
            out.file["maps"].attrs.update(metadata)


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
        "depth": StageParameter(float, 25.0, msg="Uniform depth value to assign everywhere."),
    }

    def run(self):
        import healsparse as hsp

        with self.open_input("mask", wrapper=True) as f:
            metadata = dict(f.file["maps/mask"].attrs)
            mask = f.read_mask()

        # Make a fake depth map
        depth = hsp.HealSparseMap.make_empty(
            mask.nside_coverage, mask.nside_sparse, dtype=float
        )
        depth[mask.valid_pixels] = self.config["depth"]  # e.g. 25 everywhere

        with self.open_output("aux_lens_maps", wrapper=True) as f:
            f.file.create_group("depth")
            f.write_map("depth/depth", depth, metadata)


class TXAuxiliarySSIMaps(TXBaseMaps):
    """
    Generate auxiliary maps from SSI catalogs

    This class generates maps of:
        - depth (measured magnitude)
        - depth (true magnitude)
    """

    name = "TXAuxiliarySSIMaps"
    dask_parallel = True
    inputs = [
        ("matched_ssi_photometry_catalog", HDFFile),  # injected objhects that were detected
        ("injection_catalog", HDFFile),  # injection locations
        ("ssi_detection_catalog", HDFFile),  # detection info on each injection
    ]
    outputs = [
        ("aux_ssi_maps", MapsFile),
    ]

    ###################
    ##################

    config_options = {
        "block_size": StageParameter(int, 0, msg="Block size for dask processing (0 means auto)."),
        "depth_band": StageParameter(str, "i", msg="Band for depth maps."),
        "snr_threshold": StageParameter(float, 10.0, msg="S/N value for depth maps."),
        "snr_delta": StageParameter(float, 1.0, msg="Delta for S/N thresholding."),
        "det_prob_threshold": StageParameter(float, 0.8, msg="Detection probability threshold for SSI depth."),
        "mag_delta": StageParameter(float, 0.01, msg="Magnitude bin size for detection probability depth."),
        "min_depth": StageParameter(float, 18, msg="Minimum magnitude for detection probability depth."),
        "max_depth": StageParameter(float, 26, msg="Maximum magnitude for detection probability depth."),
        "smooth_det_frac": StageParameter(bool, True, msg="Apply smoothing to detection fraction vs magnitude."),
        "smooth_window": StageParameter(float, 0.5, msg="Smoothing window size in magnitudes."),
    }

    def run(self):
        # Import dask and alias it as 'da'
        _, da = import_dask()
        import healsparse as hsp

        # Retrieve configuration parameters
        block_size = self.config["block_size"]
        if block_size == 0:
            block_size = "auto"
        band = self.config["depth_band"]

        # Open the input catalog files
        # We can't use "with" statements because we need to keep the file open
        # while we're using dask.
        f_matched = self.open_input("matched_ssi_photometry_catalog", wrapper=True)
        f_inj = self.open_input("injection_catalog", wrapper=True)
        f_det = self.open_input("ssi_detection_catalog", wrapper=True)

        # Load matched catalog data into dask arrays.
        # This is lazy in dask, so we're not actually loading the data here.
        ra = da.from_array(f_matched.file["photometry/ra"], block_size)
        block_size = ra.chunksize
        dec = da.from_array(f_matched.file["photometry/dec"], block_size)
        snr = da.from_array(f_matched.file[f"photometry/snr_{band}"], block_size)
        mag_meas = da.from_array(f_matched.file[f"photometry/mag_{band}"], block_size)
        mag_true = da.from_array(f_matched.file[f"photometry/inj_mag_{band}"], block_size)

        # Choose the pixelization scheme based on the configuration.
        # Might need to review this to make sure we use the same scheme everywhere
        pixel_scheme = choose_pixelization(**self.config)

        # make coverage map for these ra,dec
        cov_map = make_coverage_map(ra, dec, pixel_scheme)

        # Load detection catalog data into dask arrays.
        # This is lazy in dask, so we're not actually loading the data here.
        ra_inj = da.from_array(f_inj.file["photometry/ra"], block_size)
        dec_inj = da.from_array(f_inj.file["photometry/dec"], block_size)
        inj_mag = da.from_array(f_inj.file[f"photometry/inj_mag_{band}"], block_size)
        det = da.from_array(f_det.file[f"photometry/detected"], block_size)

        # Initialize a dictionary to store the maps.
        # To start with this is all lazy too, until we call compute
        maps = {}

        # Create depth maps using dask and measured magnitudes
        depth_map_results = make_dask_depth_map(
            ra,
            dec,
            mag_meas,
            snr,
            self.config["snr_threshold"],
            self.config["snr_delta"],
            pixel_scheme,
            cov_map,
        )
        maps["depth_meas/depth"] = depth_map_results["depth_map"]
        maps["depth_meas/depth_count"] = depth_map_results["count_map"]
        maps["depth_meas/depth_var"] = depth_map_results["depth_var"]

        # Create depth maps using dask and true magnitudes
        depth_map_results = make_dask_depth_map(
            ra,
            dec,
            mag_true,
            snr,
            self.config["snr_threshold"],
            self.config["snr_delta"],
            pixel_scheme,
            cov_map,
        )
        maps["depth_true/depth"] = depth_map_results["depth_map"]
        maps["depth_true/depth_count"] = depth_map_results["count_map"]
        maps["depth_true/depth_var"] = depth_map_results["depth_var"]

        # Create depth maps using injection catalog
        # depth is defined at given detection probability
        depth_map_results = make_dask_depth_map_det_prob(
            ra_inj,
            dec_inj,
            inj_mag,
            det,
            self.config["det_prob_threshold"],
            self.config["mag_delta"],
            self.config["min_depth"],
            self.config["max_depth"],
            pixel_scheme,
            cov_map,
            self.config["smooth_det_frac"],
            self.config["smooth_window"],
        )
        maps["depth_det_prob/depth"] = depth_map_results["depth_map"]
        maps["depth_det_prob/depth_det_count"] = depth_map_results["det_count_map"]
        maps["depth_det_prob/depth_inj_count"] = depth_map_results["inj_count_map"]
        maps["depth_det_prob/det_frac_by_mag_thres"] = depth_map_results[
            "det_frac_by_mag_thres"
        ]

        (maps,) = da.compute(maps)

        # convert sparse_map arrays into healsparse map objects
        hsp_maps = {}
        for name, map in maps.items():
            if map.ndim == 2:  # is a 2D map, save as recarray
                map = np.rec.fromarrays(
                    map, names=[f"bin{i}" for i in range(map.shape[0])]
                )
                primary = "bin0"
            else:
                primary = None
            hsp_maps[name] = hsp.HealSparseMap(
                cov_map=cov_map,
                sparse_map=map,
                nside_sparse=cov_map.nside_sparse,
                primary=primary,
            )

        # Prepare metadata for the maps. Copy the pixelization-related
        # configuration options only here
        metadata = {key: self.config[key] for key in map_config_options if key in self.config}
        # Then add the other configuration options
        metadata["depth_band"] = band
        metadata["depth_snr_threshold"] = self.config["snr_threshold"]
        metadata["depth_snr_delta"] = self.config["snr_delta"]
        metadata["mag_delta"] = self.config["mag_delta"]
        metadata["min_depth"] = self.config["min_depth"]
        metadata["max_depth"] = self.config["max_depth"]
        metadata["mag_edges"] = depth_map_results["mag_edges"]
        metadata.update(pixel_scheme.metadata)

        # Write the output maps to the output file
        with self.open_output("aux_ssi_maps", wrapper=True) as out:
            for map_name, m in hsp_maps.items():
                out.write_map(map_name, m, metadata)
            out.file["maps"].attrs.update(metadata)


class TXSelectionFunctionSSIMaps(TXBaseMaps):
    """
    Generate map of the selection function from SSI catalogs.

    This class generates maps of:
        - the selection function (in regions where SSI has been done)
        - the uncertainty on the measured selection function
    """

    name = "TXSelectionFunctionSSIMaps"
    dask_parallel = True
    inputs = [
        ("matched_ssi_photometry_catalog", HDFFile),  # injected objects that were detected
        ("injection_catalog", HDFFile),  # injection locations
        ("ssi_detection_catalog", HDFFile),  # detection info on each injection
    ]
    outputs = [
        ("sel_func_ssi_maps", MapsFile),
    ]

    config_options = {
        "block_size": StageParameter(int, 0, msg="Block size for dask processing (0 means auto)."),
        **map_config_options
    }

    def run(self):
        # Import dask and alias it as 'da'
        _, da = import_dask()
        import healsparse as hsp

        # Retrieve configuration parameters
        block_size = self.config["block_size"]
        if block_size == 0:
            block_size = "auto"

        # Open the input catalog files
        # We can't use "with" statements because we need to keep the file open
        # while we're using dask.
        f_matched = self.open_input("matched_ssi_photometry_catalog", wrapper=True)
        f_inj = self.open_input("injection_catalog", wrapper=True)
        f_det = self.open_input("ssi_detection_catalog", wrapper=True)

        # Load matched catalog data into dask arrays.
        # This is lazy in dask, so we're not actually loading the data here.
        ra = da.from_array(f_matched.file["photometry/ra"], block_size)
        block_size = ra.chunksize
        dec = da.from_array(f_matched.file["photometry/dec"], block_size)

        # Choose the pixelization scheme based on the configuration.
        # Might need to review this to make sure we use the same scheme everywhere
        pixel_scheme = choose_pixelization(**self.config)

        # Load detection catalog data into dask arrays.
        # This is lazy in dask, so we're not actually loading the data here.
        ra_inj = da.from_array(f_inj.file["photometry/ra"], block_size)
        dec_inj = da.from_array(f_inj.file["photometry/dec"], block_size)
        det = da.from_array(f_det.file[f"photometry/detected"], block_size)

        # Make coverage map for these ra,dec
        cov_map = make_coverage_map(ra_inj, dec_inj, pixel_scheme)

        # Initialize a dictionary to store the maps.
        # To start with this is all lazy too, until we call compute
        maps = {}

        # Create selection function map using injection catalog
        sel_func_results = make_dask_selection_function(
            ra_inj,
            dec_inj,
            det,
            pixel_scheme,
            cov_map
        )

        maps["selection_function"] = sel_func_results["sel_func_map"]
        maps["err_selection_function"] = sel_func_results["err_sel_func_map"]

        (maps,) = da.compute(maps)

        # convert sparse_map arrays into healsparse map objects
        hsp_maps = {}
        for name, map in maps.items():
            hsp_maps[name] = hsp.HealSparseMap(
                cov_map=cov_map,
                sparse_map=map,
                nside_sparse=cov_map.nside_sparse,
            )

        # Prepare metadata for the maps. Copy the pixelization-related
        # configuration options only here
        metadata = {key: self.config[key] for key in map_config_options if key in self.config}
        metadata.update(pixel_scheme.metadata)

        # Write the output maps to the output file
        with self.open_output("sel_func_ssi_maps", wrapper=True) as out:
            for map_name, m in hsp_maps.items():
                out.write_map(map_name, m, metadata)
            out.file["maps"].attrs.update(metadata)
