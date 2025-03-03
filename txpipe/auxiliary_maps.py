from .maps import TXBaseMaps, map_config_options
import numpy as np
from .base_stage import PipelineStage
from .mapping import make_dask_shear_maps, make_dask_flag_maps, make_dask_bright_object_map, make_dask_depth_map, make_dask_depth_map_det_prob
from .data_types import MapsFile, HDFFile, ShearCatalog
from .utils import choose_pixelization, import_dask
from .maps import map_config_options





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
        dask, da = import_dask()
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
        # force all columns to use the same block size
        block_size = ra.chunksize
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
        metadata.update(pixel_scheme.metadata)

        for i in bins:
            if i == "all":
                w = b >= 0
            else:
                w = b == i

            count_map, g1_map, g2_map, weight_map, esq_map, var1_map, var2_map = make_dask_shear_maps(
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
        "block_size": 0,
        "bright_obj_threshold": 22.0,  # The magnitude threshold for a object to be counted as bright
        "depth_band": "i",  # Make depth maps for this band
        "snr_threshold": 10.0,  # The S/N value to generate maps for (e.g. 5 for 5-sigma depth)
        "snr_delta": 1.0,  # The range threshold +/- delta is used for finding objects at the boundary
    }

    def run(self):
        # Import dask and alias it as 'da'
        _, da = import_dask()
        
        
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
        # This is lazy in dask, so we're not actually loading the data here.
        ra = da.from_array(f.file["photometry/ra"], block_size)
        block_size = ra.chunksize
        dec = da.from_array(f.file["photometry/dec"], block_size)
        extendedness = da.from_array(f.file["photometry/extendedness"], block_size)
        snr = da.from_array(f.file[f"photometry/snr_{band}"], block_size)
        mag = da.from_array(f.file[f"photometry/mag_{band}"], block_size)

        # Choose the pixelization scheme based on the configuration.
        # Might need to review this to make sure we use the same scheme everywhere
        pixel_scheme = choose_pixelization(**self.config)
        
        # Initialize a dictionary to store the maps.
        # To start with this is all lazy too, until we call compute
        maps = {}

        # Create a bright object map using dask
        pix1, bright_object_count_map = make_dask_bright_object_map(
            ra, dec, mag, extendedness, self.config["bright_obj_threshold"], pixel_scheme)
        maps["bright_objects/count"] = (pix1, bright_object_count_map[pix1])

        # Create depth maps using dask
        pix2, count_map, depth_map, depth_var = make_dask_depth_map(
            ra, dec, mag, snr, self.config["snr_threshold"], self.config["snr_delta"], pixel_scheme)
        maps["depth/depth"] = (pix2, depth_map[pix2])
        maps["depth/depth_count"] = (pix2, count_map[pix2])
        maps["depth/depth_var"] = (pix2, depth_var[pix2])


        maps, = da.compute(maps)

        # Prepare metadata for the maps. Copy the pixelization-related
        # configuration options only here
        metadata = {
            key: self.config[key]
            for key in map_config_options
            if key in self.config
        }
        # Then add the other configuration options
        metadata["depth_band"] = band
        metadata["depth_snr_threshold"] = self.config["snr_threshold"]
        metadata["depth_snr_delta"] = self.config["snr_delta"]
        metadata["bright_obj_threshold"] = self.config["bright_obj_threshold"]
        metadata.update(pixel_scheme.metadata)

        # Write the output maps to the output file
        with self.open_output("aux_lens_maps", wrapper=True) as out:
            for map_name, (pix, m) in maps.items():
                out.write_map(map_name, pix, m, metadata)
            out.file['maps'].attrs.update(metadata)



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
        ("matched_ssi_photometry_catalog", HDFFile), # injected objhects that were detected
        ("injection_catalog", HDFFile),  # injection locations
        ("ssi_detection_catalog", HDFFile), # detection info on each injection
    ]
    outputs = [
        ("aux_ssi_maps", MapsFile),
    ]

        ###################
        ##################

    config_options = {
        "block_size": 0,
        "depth_band": "i",  # Make depth maps for this band
        "snr_threshold": 10.0,  # The S/N value to generate maps for (e.g. 5 for 5-sigma depth)
        "snr_delta": 1.0,  # The range threshold +/- delta is used for finding objects at the boundary
        "det_prob_threshold": 0.8, #detection probability threshold for SSI depth (i.e. 0.9 for magnitude at which 90% of brighter objects are detected)
        "mag_delta": 0.01,  # Size of the magnitude bins used to determine detection probability depth
        "min_depth": 18, # Min magnitude used in detection probability depth
        "max_depth": 26, # Max magnitude used in detection probability depth
    }

    def run(self):
        # Import dask and alias it as 'da'
        _, da = import_dask()
        
        
        # Retrieve configuration parameters
        block_size = self.config["block_size"]
        if block_size == 0:
            block_size = "auto"
        band = self.config["depth_band"]

        # Open the input catalog files
        # We can't use "with" statements because we need to keep the file open
        # while we're using dask.
        f_matched = self.open_input("matched_ssi_photometry_catalog", wrapper=True)
        f_inj     = self.open_input("injection_catalog", wrapper=True)
        f_det     = self.open_input("ssi_detection_catalog", wrapper=True)
        
        # Load matched catalog data into dask arrays.
        # This is lazy in dask, so we're not actually loading the data here.
        ra = da.from_array(f_matched.file["photometry/ra"], block_size)
        block_size = ra.chunksize
        dec = da.from_array(f_matched.file["photometry/dec"], block_size)
        snr = da.from_array(f_matched.file[f"photometry/snr_{band}"], block_size)
        mag_meas = da.from_array(f_matched.file[f"photometry/mag_{band}"], block_size)
        mag_true = da.from_array(f_matched.file[f"photometry/inj_mag_{band}"], block_size)

        # Load detection catalog data into dask arrays.
        # This is lazy in dask, so we're not actually loading the data here.
        ra_inj = da.from_array(f_inj.file["photometry/ra"], block_size)
        dec_inj = da.from_array(f_inj.file["photometry/dec"], block_size)
        inj_mag = da.from_array(f_inj.file[f"photometry/inj_mag_{band}"], block_size)
        det = da.from_array(f_det.file[f"photometry/detected"], block_size)

        # Choose the pixelization scheme based on the configuration.
        # Might need to review this to make sure we use the same scheme everywhere
        pixel_scheme = choose_pixelization(**self.config)
        
        # Initialize a dictionary to store the maps.
        # To start with this is all lazy too, until we call compute
        maps = {}

        # Create depth maps using dask and measured magnitudes
        pix2, count_map, depth_map, depth_var = make_dask_depth_map(
            ra, dec, mag_meas, snr, self.config["snr_threshold"], self.config["snr_delta"], pixel_scheme)
        maps["depth_meas/depth"] = (pix2, depth_map[pix2])
        maps["depth_meas/depth_count"] = (pix2, count_map[pix2])
        maps["depth_meas/depth_var"] = (pix2, depth_var[pix2])

        # Create depth maps using dask and true magnitudes
        pix2, count_map, depth_map, depth_var = make_dask_depth_map(
            ra, dec, mag_true, snr, self.config["snr_threshold"], self.config["snr_delta"], pixel_scheme)
        maps["depth_true/depth"] = (pix2, depth_map[pix2])
        maps["depth_true/depth_count"] = (pix2, count_map[pix2])
        maps["depth_true/depth_var"] = (pix2, depth_var[pix2])

        # Create depth maps using injection catalog
        # depth is defined at given detection probability
        pix2, det_count_map, inj_count_map, depth_map, frac_stack, mag_edges = make_dask_depth_map_det_prob(
            ra_inj, dec_inj, inj_mag, det, self.config["det_prob_threshold"], self.config["mag_delta"], self.config["min_depth"], self.config["max_depth"], pixel_scheme)
        maps["depth_det_prob/depth"] = (pix2, depth_map[pix2])
        maps["depth_det_prob/depth_det_count"] = (pix2, det_count_map[pix2])
        maps["depth_det_prob/depth_inj_count"] = (pix2, inj_count_map[pix2])
        maps["depth_det_prob/frac_stack"] = (pix2, frac_stack[:,pix2])

        maps, = da.compute(maps)

        # Prepare metadata for the maps. Copy the pixelization-related
        # configuration options only here
        metadata = {
            key: self.config[key]
            for key in map_config_options
            if key in self.config
        }
        # Then add the other configuration options
        metadata["depth_band"] = band
        metadata["depth_snr_threshold"] = self.config["snr_threshold"]
        metadata["depth_snr_delta"] = self.config["snr_delta"]
        metadata["mag_delta"] = self.config["mag_delta"]
        metadata["min_depth"] = self.config["min_depth"]
        metadata["max_depth"] = self.config["max_depth"]
        metadata["mag_edges"] = mag_edges
        metadata.update(pixel_scheme.metadata)

        # Write the output maps to the output file
        with self.open_output("aux_ssi_maps", wrapper=True) as out:
            for map_name, (pix, m) in maps.items():
                out.write_map(map_name, pix, m, metadata)
            out.file['maps'].attrs.update(metadata)
