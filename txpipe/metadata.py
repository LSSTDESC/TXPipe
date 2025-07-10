import numpy as np
import yaml
from .base_stage import PipelineStage
from .data_types import TomographyCatalog, MapsFile, HDFFile, YamlFile, ShearCatalog
from .utils.calibration_tools import read_shear_catalog_type
from .utils import choose_pixelization

def copy(tomo, in_section, out_section, name, meta_file, metadata, new_name=None):
    if new_name is None:
        new_name = name
    x = tomo[f"{in_section}/{name}"][:]
    meta_file.create_dataset(f"{out_section}/{new_name}", data=x)
    metadata[new_name] = x.tolist()

def copy_attrs(tomo, name, out_name, meta_file, metadata):
    for k, v in tomo[name].attrs.items():
        meta_file[out_name].attrs[k] = v
        if isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, np.float64):
            v = float(v)
        elif isinstance(v, np.int64):
            v = int(v)
        metadata[k] = v

class TXTracerMetadata(PipelineStage):
    """
    Collate metadata from various other files

    This stage doesn't actually calculate anything, it just
    collates together metadata about our sources, so that we
    don't need to pass around catalog sized objects as much.
    """

    name = "TXTracerMetadata"
    parallel = False

    inputs = [
        ("shear_catalog", ShearCatalog),
        ("shear_tomography_catalog", TomographyCatalog),
        ("lens_tomography_catalog", TomographyCatalog),
        ("mask", MapsFile),
    ]

    outputs = [
        ("tracer_metadata", HDFFile),
        ("tracer_metadata_yml", YamlFile),  # human-readable version
    ]

    def copy_source_metadata(self, meta_file, metadata, area, area_sq_arcmin):
        if self.get_input("shear_tomography_catalog") == "none":
            print("Skipping source metadata")
            return
        shear_catalog_type = read_shear_catalog_type(self)

        with self.open_input("shear_tomography_catalog") as shear_tomo_file:
            if shear_catalog_type == "metacal":
                copy(shear_tomo_file, "response", "tracers", "R_gamma_mean", meta_file, metadata)
                copy(shear_tomo_file, "response", "tracers", "R_S", meta_file, metadata)
                copy(shear_tomo_file, "response", "tracers", "R_total", meta_file, metadata)
                copy(shear_tomo_file, "response", "tracers", "R_gamma_mean_2d", meta_file, metadata)
                copy(shear_tomo_file, "response", "tracers", "R_S_2d", meta_file, metadata)
                copy(shear_tomo_file, "response", "tracers", "R_total_2d", meta_file, metadata)
            elif shear_catalog_type == "metadetect":
                copy(shear_tomo_file, "response", "tracers", "R", meta_file, metadata)
                copy(shear_tomo_file, "response", "tracers", "R_2d", meta_file, metadata)
            elif shear_catalog_type == "lensfit":
                copy(shear_tomo_file, "response", "tracers", "K", meta_file, metadata)
                copy(shear_tomo_file, "response", "tracers", "C_N", meta_file, metadata)
                copy(shear_tomo_file, "response", "tracers", "C_S", meta_file, metadata)
                copy(shear_tomo_file, "response", "tracers", "K_2d", meta_file, metadata)
                copy(shear_tomo_file, "response", "tracers", "C_2d_N", meta_file, metadata)
                copy(shear_tomo_file, "response", "tracers", "C_2d_S", meta_file, metadata)
            elif shear_catalog_type == "hsc":
                copy(shear_tomo_file, "response", "tracers", "R", meta_file, metadata)
                copy(shear_tomo_file, "response", "tracers", "K", meta_file, metadata)
                copy(shear_tomo_file, "response", "tracers", "R_mean_2d", meta_file, metadata)
                copy(shear_tomo_file, "response", "tracers", "K_2d", meta_file, metadata)

            copy(shear_tomo_file, "counts", "tracers", "N_eff", meta_file, metadata)
            copy(shear_tomo_file, "counts", "tracers", "sigma_e", meta_file, metadata)
            copy(shear_tomo_file, "counts", "tracers", "mean_e1", meta_file, metadata)
            copy(shear_tomo_file, "counts", "tracers", "mean_e2", meta_file, metadata)
            copy(shear_tomo_file, "counts", "tracers", "counts", meta_file, metadata, "source_counts")

            copy(shear_tomo_file, "counts", "tracers", "N_eff_2d", meta_file, metadata)
            copy(shear_tomo_file, "counts", "tracers", "sigma_e_2d", meta_file, metadata)
            copy(shear_tomo_file, "counts", "tracers", "mean_e1_2d", meta_file, metadata)
            copy(shear_tomo_file, "counts", "tracers", "mean_e2_2d", meta_file, metadata)
            copy(shear_tomo_file, "counts", "tracers", "counts_2d", meta_file, metadata, "source_counts_2d")

            N_eff = shear_tomo_file["counts/N_eff"][:]
            N_eff_2d = shear_tomo_file["counts/N_eff_2d"][:]
            n_eff = N_eff / area_sq_arcmin
            n_eff_2d = N_eff_2d / area_sq_arcmin

            source_counts = shear_tomo_file["counts/counts"][:]
            source_counts_2d = shear_tomo_file["counts/counts_2d"][:]

            source_density = source_counts / area_sq_arcmin
            source_density_2d = source_counts_2d / area_sq_arcmin

            meta_file.create_dataset("tracers/n_eff", data=n_eff)
            meta_file.create_dataset("tracers/source_density", data=source_density)
            meta_file.create_dataset("tracers/source_density_2d", data=source_density_2d)

            meta_file["tracers"].attrs["area"] = area
            meta_file["tracers"].attrs["area_unit"] = "deg^2"
            meta_file["tracers"].attrs["density_unit"] = "arcmin^{-2}"
            metadata["n_eff"] = n_eff.tolist()
            metadata["n_eff_2d"] = n_eff_2d.tolist()
            metadata["source_density_2d"] = source_density_2d.tolist()
            metadata["source_density"] = source_density.tolist()

            copy_attrs(shear_tomo_file, "tomography", "tracers", meta_file, metadata)
            

    def copy_lens_metadata(self, meta_file, metadata, area, area_sq_arcmin):
        if self.get_input("lens_tomography_catalog") == "none":
            print("Skipping lens metadata")
            return

        with self.open_input("lens_tomography_catalog") as lens_tomo_file:

            copy(lens_tomo_file, "counts", "tracers", "counts", meta_file, metadata, "lens_counts")
            copy(lens_tomo_file, "counts", "tracers", "counts_2d", meta_file, metadata, "lens_counts_2d")

            lens_counts = lens_tomo_file["counts/counts"][:]
            lens_counts_2d = lens_tomo_file["counts/counts_2d"][:]

            lens_density = lens_counts / area_sq_arcmin
            lens_density_2d = lens_counts_2d / area_sq_arcmin

            meta_file.create_dataset("tracers/lens_density", data=lens_density)
            meta_file.create_dataset("tracers/lens_density_2d", data=lens_density_2d)

            meta_file["tracers"].attrs["area"] = area
            meta_file["tracers"].attrs["area_unit"] = "deg^2"
            meta_file["tracers"].attrs["density_unit"] = "arcmin^{-2}"
            metadata["lens_density"] = lens_density.tolist()
            metadata["lens_density_2d"] = lens_density_2d.tolist()

            copy_attrs(lens_tomo_file, "tomography", "tracers", meta_file, metadata)

    def run(self):
        # Read the area
        area = self.read_area()
        area_sq_arcmin = area * 60**2

        with self.open_output("tracer_metadata") as meta_file:
            metadata = {}
            metadata["area"] = area
            metadata["area_unit"] = "deg^2"
            metadata["density_unit"] = "arcmin^{-2}"
            self.copy_source_metadata(meta_file, metadata, area, area_sq_arcmin)
            self.copy_lens_metadata(meta_file, metadata, area, area_sq_arcmin)

        yaml_out = self.open_output("tracer_metadata_yml", wrapper=True)
        yaml_out.write(metadata)
        yaml_out.close()


    def read_area(self):
        with self.open_input("mask", wrapper=True) as f:
            m = f.read_map("mask")
            pixel_scheme = choose_pixelization(**f.read_map_info("mask"))

        num_hit = np.sum(m[m > 0])  # Assuming fracdet mask
        area_sq_deg = pixel_scheme.pixel_area(degrees=True) * num_hit
        f_sky = float(area_sq_deg) / 41252.96125
        print(f"Area = {area_sq_deg:.2f} deg^2")
        return float(area_sq_deg)
