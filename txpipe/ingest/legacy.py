from ..base_stage import PipelineStage
from ..data_types import HDFFile
import numpy as np
from .base import TXIngestCatalogH5

class TXIngestDESY3Gold(TXIngestCatalogH5):
    """
    Ingest the DES Y3 Gold from hdf5 format
    """
    name = "TXIngestDESY3Gold"
    parallel = False
    inputs = [
        ("des_photometry_catalog", HDFFile),
    ]

    outputs = [
        ("photometry_catalog", HDFFile),
    ]

    config_options = {
        "chunk_rows": 100_000,
    }

    def run(self):
        """
        Run the analysis for this stage.
        """
        print("Ingesting DES Y3 Gold catalog")

        # we will only load a subset of columns to save space
        column_names = {
            "coadd_object_id": "id", 
            "ra": "ra",
            "dec": "dec",
            "sof_cm_mag_corrected_g": "mag_g",
            "sof_cm_mag_corrected_i": "mag_i",
            "sof_cm_mag_corrected_r": "mag_r",
            "sof_cm_mag_corrected_z": "mag_z",
            "sof_cm_mag_err_g": "mag_err_g",
            "sof_cm_mag_err_i": "mag_err_i",
            "sof_cm_mag_err_r": "mag_err_r",
            "sof_cm_mag_err_z": "mag_err_z",
            "sof_cm_fracdev": "sof_cm_fracdev",
            "sof_cm_t": "T", #Size parameter 
            #"sof_flags": "sof_flags", #Flags from SOF photometry 
            "extended_class_sof": "EXTENDED_CLASS_SOF", #star galaxy separator using SOF photometry
            "extended_class_mash_sof": "EXTENDED_CLASS_MASH_SOF", #alternative star galaxy separator using SOF photometry as primary source
            "flags": "flags", #combined flags column TODO: confirm if this combines all of the below flags
            #"flags_badregions": "flags_badregions",
            #"flags_footprint": "flags_footprint",
            #"flags_foreground": "flags_foreground",
            #"flags_gold": "flags_gold",
            #"flags_phot": "flags_phot",
            #"hpix_16384": "hpix_16384",
            "tilename": "tilename", #Name of DES tile
        }
        dummy_columns = {
        }

        self.process_catalog(
            "des_photometry_catalog",
            "photometry_catalog",
            column_names,
            dummy_columns,
        )