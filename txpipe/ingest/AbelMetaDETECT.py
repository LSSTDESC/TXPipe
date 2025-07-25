from ..base_stage import PipelineStage
from ..data_types import ShearCatalog, HDFFile, TextFile, QPBaseFile, ParquetFile
from ..utils import band_variants, metadetect_variants, Timer, nanojansky_err_to_mag_ab, nanojansky_to_mag_ab, moments_to_shear, mag_ab_to_nanojansky
import numpy as np
import glob
import math
import pandas as pd


class TXAbelMetaDetect(PipelineStage):
    """
    This is a first pass at implementing an ingestion stage for the MetaDetect
    Catalog coming out of the Rubin pipeline. 

    Currently we expect Rubin to give us the data, one Tract at a time, so we will
    have to loop over the tracts to generate one catalog file.

    This first itteration will ingest HSC data processed by the Rubin pipeline.
    """

    name = "TXAbelMetaDetect"
    parallel = False
    inputs = []

    outputs = [
        ("shear_catalog", ShearCatalog),
        ("photometry_catalog", QPBaseFile)
    ]

    config_options = {
        "pkl_path": "",
        "tracts": "",
        "bands": "gri",
        "input_tracts": True, #Assume that data is split into files per tract
        "input_variations": False, # Assume that the data is split into files per variation
        "moment": "wmom", #Which type of moment are we using for our files
    }

    def run(self):
        
        shear_table = pd.read_pickle(self.config["pkl_path"])
        n = len(shear_table)
        input_columns = shear_table.columns

        self.shear_output_cols = metadetect_variants(
            "id",
            "tract",
            "g1",
            "g2",
            "T",
            "s2n",
            "T_err",
            "ra",
            "dec",
            "psf_g1",
            "psf_g2",
            "mcal_psf_g1",
            "mcal_psf_g2",
            "mcal_psf_T_mean",
            "weight",
        ) + band_variants(bands, "mag", "mag_err", "flux_flag",
                          shear_catalog_type="metadetect")
        self.phot_output_cols = metadetect_variants(
            "id",
            "tract",
            "ra",
            "dec",
        ) + band_variants(bands,  "mag", "mag_err", "flux_flag",
                          shear_catalog_type="metadetect")

        created_files = False

        phot_output = {name: [] for name in self.phot_output_cols}
        shear_output = {name: [] for name in self.shear_output_cols}
        for i, data in shear_table.iterrows():
            sheartype = data['shear_type']
            phot_output[f"{sheartype}/ra"].append(data['ra'])
            phot_output[f"{sheartype}/dec"].append(data['dec'])
            phot_output[f"{sheartype}/id"].append(data['id'])
            phot_output[f"{sheartype}/tract"].append(data["tract"])
            for band in self.config['bands']:
                phot_output[f"{sheartype}/mag_{band}"].append(nanojansky_to_mag_ab( data[f"{moment}_band_flux_{band}"]))
                phot_output[f"{sheartype}/mag_{band}_err"].append(nanojansky_err_to_mag_ab(data[f"{moment}_band_flux_err_{band}"]))
                phot_output[f"{sheartype}/flux_{band}_flag"].append(data[f"{moment}_band_flux_flags_{band}"])

        photo_outfile = self.setup_output("photometry_catalog", "photometry", phot_output, n)

        for i, d in shear_table.itterows():
            sheartype = d['shear_type']
            shear_output[f"{sheartype}/ra"].append(d['ra'])
            shear_output[f"{sheartype}/id"].append(d['id'])
            shear_output[f"{sheartype}/dec"].append(d['dec'])
            shear_output[f"{sheartype}/tract"].append(d["tract"])
            shear_output[f"{sheartype}/psf_flags"].append(d["{moment}_psf_flags"])
            shear_output[f"{sheartype}/psf_g1"].append(d[f"{moment}_psf_g_1"])
            shear_output[f"{sheartype}/psf_g2"].append(d[f"{moment}_psf_g_2"])
            shear_output[f"{sheartype}/psf_T"].append(d[f"{moment}_psf_T"])
            shear_output[f"{sheartype}/obj_flags"].append(d[f"{moment}_obj_flags"])
            shear_output[f"{sheartype}/s2n"].append(d[f"{moment}_s2n"])
            shear_output[f"{sheartype}/g1"].append(d[f"{moment}_g_1"])
            shear_output[f"{sheartype}/g2"].append(d[f"{moment}_g_2"]) 
            shear_output[f"{sheartype}/T"].append(d[f"{moment}_T"])
            shear_output[f"{sheartype}/T_err"].append(d[f"{moment}_T_err"])
            shear_output[f"{sheartype}/T_flags"].append(d[f"{moment}_T_flags"])
            shear_output[f"{sheartype}/T_ration"].append(d[f"{moment}_T_ration"])
            for band in self.config["bands"]:
                shear_output[f"{sheartype}/mag_{band}"].append(nanojansky_to_mag_ab( d[f"{moment}_band_flux_{band}"]))
                shear_output[f"{sheartype}/mag_err_{band}"].append(nanojansky_err_to_mag_ab( d[f"{moment}_band_flux_err_{band}"]))
                shear_output[f"{sheartype}/flux_flag_{band}"].append(d[f"{moment}_band_flux_flags_{band}"])

        shear_outfile = self.setup_output("shear_catalog", "shear", shear_output, n)


    def setup_output(self, tag, group, first_chunk, n):
        f = self.open_output(tag)
        g = f.create_group(group)

        for name, col in first_chunk.items():
            g.create_dataset(name, shape=(n,), dtype=col.dtype)
        return f

    def write_output(self, outfile, group, data, start, end):
        g = outfile[group]
        for name, col in data.items():
            # replace masked values with nans
            if np.ma.isMaskedArray(col):
                col = col.filled(np.nan)
            g[name][start:end] = col
        