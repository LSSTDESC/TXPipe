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
        maps["depth_det_prob/inj_count_by_mag_thres"] = depth_map_results[
            "inj_count_by_mag_thres"
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


class TXModelSelectionFunction(TXBaseMaps):
    """
    Model selection function across footprint using survey property maps.

    Uses the selection measured in a subset of the footprint to model its dependence on
    survey property maps, then uses that model to predict the selection function across
    the remainder of the footprint. The model is fit via polynomial regression with
    respect to the survey property maps, where the degree of the polynomial is specified
    via the config.
    TODO: Make it possible to select the degree of the fit for each SP map individually.
    TODO: Add other options for the form of the model.
    """

    name = "TXModelSelectionFunction"
    dask_parallel = True
    inputs = [
        ("aux_ssi_maps", MapsFile),  # Measured selection function and uncertainties
        ("mask", MapsFile)  # Mask defining survey geometry

    ]
    outputs = [
        ("sel_func_pred_maps", MapsFile),  # Model prediction of selection function and its uncertainties
        ("sel_func_model_info", HDFFile),  # Best-fit params and their covariance matrix
    ]

    config_options = {
        "block_size": StageParameter(int, 0, msg="Block size for dask processing (0 means auto)."),
        "systmaps_dir": StageParameter(str, "", msg="Directory containing systematic maps."),
        "degree": StageParameter(int, 1, msg="Degree of the polynomial fit."),
        "mask_thresh": StageParameter(float, 0.0, msg="Threshold for masking pixels at native resolution of mask."),
        "mask_thresh_coarse": StageParameter(float, 0.0, msg="Threshold for masking pixels after mask is degraded."),
        "inj_count_thresh": StageParameter(int, 1, msg="Exclude pixels containing fewer injections than this number."),
        "sel_func_err_type": StageParameter(str, "none", msg="Type of uncertainty attributed to measured selection function."),
        **map_config_options
    }

    def run(self):
        import glob
        import healpy as hp
        import healsparse as hsp
        from functools import reduce
        # Import dask and alias it as 'da'
        _, da = import_dask()
        # Assign imports to self to avoid repeated imports in other functions
        self.da = da

        # Retrieve configuration parameters
        block_size = self.config["block_size"]
        if block_size == 0:
            block_size = "auto"
        
        # Type of uncertainty estimate to use
        err_type = self.config["sel_func_err_type"].lower()
        
        pixel_scheme = choose_pixelization(**self.config)

        # Load the map of the measured selection function and its uncertainties
        with self.open_input("aux_ssi_maps", wrapper=True) as f:
            sel_func_meas = f.read_map("depth_det_prob/det_frac_by_mag_thres")
            ninj = f.read_map("depth_det_prob/inj_count_by_mag_thres")
        # Ensure correct resolution (desired resolution can only be lower than
        # or equal to the native resolution of the selection function map).
        # "sum" reduction used for ninj since this is a counts map
        sel_func_meas = sel_func_meas.degrade(pixel_scheme.nside)
        ninj = ninj.degrade(pixel_scheme.nside, reduction="sum")

        with self.open_input("mask", wrapper=True) as f:
            mask = f.read_mask(
                thresh=self.config["mask_thresh"],
                degrade_nside=pixel_scheme.nside
            )
        mask_pix = mask.valid_pixels

        # Load survey property maps at the valid pixels
        # TODO: Define these as individual inputs rather than glob search?
        root = self.config["systmaps_dir"]
        sysfiles = glob.glob(f"{root}*.hs")
        nsys = len(sysfiles)
        print(f"Found {nsys} total systematic maps")
        spmaps = [
            hsp.HealSparseMap.read(
                sf,
                degrade_nside=pixel_scheme.nside
            )
            for sf in sysfiles 
        ]
        
        # Identify valid pixels across mask and all SP maps
        goodpix = reduce(
            np.intersect1d,
            [
                mask_pix,
                *[m.valid_pixels for m in spmaps]
            ]
        )

        # Training pixels must also be valid in the selection function
        pix_train = np.intersect1d(sel_func_meas.valid_pixels, goodpix)
        # Remove any pixels below specified coverage fraction after degrading
        pix_train = pix_train[mask[pix_train] >= self.config["mask_thresh_coarse"]]

        # Cycle through the samples for which selection functions have been measured
        samples = sel_func_meas.dtype.names
        maps = {
            "sel_func": [],
            "err_sel_func": []
        }
        model_info = {
            "alphas": [],
            "cov_alphas": []
        }
        for k in samples:
            print(f"Modelling selection function for sample {k}")

            # Remove pixels with fewer than the specified no. of injections
            pix_train_k = pix_train[ninj[k][pix_train] >= self.config["inj_count_thresh"]]
            # Select training data
            sel_func_train = da.from_array(sel_func_meas[k][pix_train_k], block_size)
            block_size = sel_func_train.chunksize
            ninj_train = da.from_array(ninj[k][pix_train_k], block_size)
            spmaps_train = da.stack(
                [
                    da.from_array(m[pix_train_k], block_size) for m in spmaps
                ]
            ).T
            # Get uncertainties on training data
            err_sel_func_train = self.sel_func_uncertainties(
                sel_func_train,
                err_type=err_type,
                ninj=(None if err_type == "none" else ninj_train)
            )

            # Get survey property values across remainder of footprint
            pix_pred = np.setdiff1d(goodpix, pix_train_k)
            spmaps_pred = da.stack(
                [
                    da.from_array(m[pix_pred], block_size) for m in spmaps
                ]
            ).T

            # Perform polynomial regression and retrieve the following:
            # - mean prediction on the selection function (in each pixel)
            # - 1-sigma uncertainty on these predictions (in each pixel)
            # - best-fit coefficients for each survey property (+ an intercept term)
            # - covariance matrix for the best-fit parameters
            sel_func_pred, err_sel_func_pred, alphas, cov_alphas = self.polynomial_model(
                spmaps_train,
                sel_func_train,
                err_sel_func_train,
                spmaps_pred
            )

            # Combine training data and predictions into single HealSparse maps
            pix_all = np.concatenate([pix_train_k, pix_pred])
            sel_func_full = da.concatenate([sel_func_train, sel_func_pred])
            err_sel_func_full = da.concatenate([err_sel_func_train, err_sel_func_pred])
            # Sort by pixel number (will make saving to HealSparse format easier)
            inds = pix_all.argsort()
            chunk_sizes = sel_func_full.chunks[0]
            chunk_bounds = np.cumsum([0] + list(chunk_sizes))
            sel_func_full = self.dask_sort(sel_func_full, inds, chunk_bounds)
            err_sel_func_full = self.dask_sort(err_sel_func_full, inds, chunk_bounds)
            maps["sel_func"].append(sel_func_full)
            maps["err_sel_func"].append(err_sel_func_full)
            # Store model info
            model_info["alphas"].append(alphas)
            model_info["cov_alphas"].append(cov_alphas)

        (maps,) = da.compute(maps)
        
        # Convert maps to healsparse map objects
        hsp_maps = {}
        dtypes = [(k, float) for k in samples]
        for n, m in maps.items():
            # Both sel_func and err_sel_func are 2D; save as recarrays
            primary = "bin0"
            m_hsp =  hsp.HealSparseMap.make_empty(
                pixel_scheme.nside_coverage,
                pixel_scheme.nside,
                dtypes,
                primary=primary
            )
            m_hsp.update_values_pix(
                goodpix,
                np.zeros(len(goodpix), dtype=dtypes)
            )
            for i,k in enumerate(samples):
                m_hsp[k][goodpix] = m[i]
            hsp_maps[n] = m_hsp

        # Prepare metadata for the maps. Copy the pixelization-related
        # configuration options only here
        metadata = {key: self.config[key] for key in map_config_options if key in self.config}
        # Then add the other configuration options
        metadata.update(pixel_scheme.metadata)

        # Write the output maps to the output file
        with self.open_output("sel_func_pred_maps", wrapper=True) as out:
            for map_name, hsp_map in hsp_maps.items():
                out.write_map(map_name, hsp_map, metadata)
            out.file["maps"].attrs.update(metadata)

        # Prepare metadata for model parameter info
        params_metadata = {
            "model": f"polynomial (deg = {self.config['degree']})",
            "inputs": [
                sf.split('/')[-1] for sf in sysfiles
            ]
        }

        # Write the model parameters and covariances to the output file
        with self.open_output("sel_func_model_info", wrapper=True) as out:
            mi = out.file.create_group("model_info")
            out.file["model_info"].attrs.update(params_metadata)
            gp_alphas = mi.create_group("alphas")
            gp_cov = mi.create_group("cov_alphas")
            for i,k in enumerate(samples):
                gp_alphas.create_dataset(k, data=model_info["alphas"][i])
                gp_cov.create_dataset(k, data=model_info["cov_alphas"][i])

    def sel_func_uncertainties(self, sel_func, err_type="none", ninj=None):
        """
        Estimates uncertainties on the measured selection function.

        Currently two values for err_type are accommodated (case-independent):
        - "none": selection function is treated as exact; zero uncertainty
        - "gaussian": uncertainties are calculated under a Gaussian approx.
        TODO: add other options, e.g. Wilson score interval.
        """
        if err_type == "none":
            return self.da.zeros_like(sel_func)
        elif err_type == "gaussian":
            return self.da.sqrt(sel_func * (1 - sel_func) / ninj)
        else:
            raise ValueError(
                "err_type must be either 'none' or 'gaussian'."
            )

    def sel_func_weights(self, err_sel_func):
        """
        Converts selection function uncertainties into weights for modelling.
        """
        if all(err_sel_func == 0):
            # Return ones if uncertanties are all zero
            return self.da.ones(len(err_sel_func))
        else:
            # Return inverse of variance
            return 1 / (self.da.power(err_sel_func, 2))

    def polynomial_model(self, X_train, y_train, yerr_train, X_pred):
        """
        Polynomial regression and model predictions for test data.
        
        Performs a polynomial regression of y with respect to X_train, then
        generates predictions at X_pred. In this context, X is an array
        containing survey property maps in certain pixels, and y is
        the selection function measured in those pixels.
        """
        da = self.da
        # Degree of polynomial fit
        deg = self.config["degree"]

        # Weights for each training pixel
        W = self.sel_func_weights(yerr_train)
        # If Gaussian approx was used for uncertainty estimation, pixels in
        # which the selection function is 0 or 1 will have zero uncertainty
        # (and thus infinite weight); remove these pixels here.
        (keep,) = da.compute(da.isfinite(W)) 
        X_train = X_train[keep, :]
        y_train = y_train[keep]
        W = W[keep]

        # Construct inputs matrix for regression
        m = len(X_train[:, 0])
        ones = da.ones((m, 1), chunks=(X_train.chunks[0], 1))
        X = da.hstack([ones, X_train])
        for n in range(2, deg + 1):
            X = da.hstack([X, da.power(X_train, n)])

        # Explicitly compute necessary array combinations
        # NOTE: this assumes a diagonal covariance matrix for y,
        # i.e. cov_y = diag(W)
        (XtWX,) = da.compute(X.T @ (W[:, None] * X))
        (XtWy,) = da.compute(X.T @ (W * y_train))

        # Compute best-fit coeffs (alphas) and their covariance
        alphas = np.linalg.solve(XtWX, XtWy)
        cov_alphas = np.linalg.inv(XtWX)

        # Now make predictions at X_pred
        m = len(X_pred[:,0])
        ones = da.ones((m, 1), chunks=(X_pred.chunks[0], 1))
        X = da.hstack([ones, X_pred])
        for n in range(2, deg + 1):
            X = da.hstack([X, da.power(X_pred, n)])
        y_pred = alphas @ X.T
        err_y_pred = np.sqrt(np.diag(X @ cov_alphas @ X.T))

        return y_pred, err_y_pred, alphas, cov_alphas

    def dask_sort(self, X, inds, chunk_boundaries):
        """
        Reorders a dask array using `inds`, without creating more chunks.
        """
        n_chunks = len(chunk_boundaries) - 1

        # Assign each index in `inds` to a chunk
        chunk_ids = np.searchsorted(chunk_boundaries[1:], inds, side='right')

        # Sort inds by chunk, preserving the mapping back to output positions
        order = np.argsort(chunk_ids, kind='stable')
        inds_sorted = inds[order]

        # Gather from D one chunk at a time
        X_sorted = self.da.empty(len(inds), dtype=X.dtype, chunks=X.chunks)

        for c in range(n_chunks):
            mask = chunk_ids[order] == c
            if not mask.any():
                continue
            local_inds = inds_sorted[mask] - chunk_boundaries[c]
            chunk_data = X[chunk_boundaries[c]:chunk_boundaries[c+1]]
            X_sorted[order[mask]] = chunk_data[local_inds]
        
        return X_sorted
