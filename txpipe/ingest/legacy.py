from ..data_types import HDFFile, MapsFile, FitsFile
from .base import TXIngestCatalogH5, TXIngestMapsHsp, TXIngestCatalogFits
from ceci.config import StageParameter


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
        "input_group_name": StageParameter(str, "catalog/gold", msg="Input group name in the HDF5 file."),
        "chunk_rows": StageParameter(int, 100_000, msg="Number of rows to process in each chunk."),
    }

    def run(self):
        """
        Run the analysis for this stage.
        """
        print("Ingesting DES Y3 Gold catalog")

        # we will only load a subset of columns to save space
        column_names = {
            "id": "coadd_object_id",
            "ra": "ra",
            "dec": "dec",
            "mag_g": "sof_cm_mag_corrected_g",
            "mag_i": "sof_cm_mag_corrected_i",
            "mag_r": "sof_cm_mag_corrected_r",
            "mag_z": "sof_cm_mag_corrected_z",
            "mag_err_g": "sof_cm_mag_err_g",
            "mag_err_i": "sof_cm_mag_err_i",
            "mag_err_r": "sof_cm_mag_err_r",
            "mag_err_z": "sof_cm_mag_err_z",
            "sof_cm_fracdev": "sof_cm_fracdev",
            "T": "sof_cm_t",  # Size parameter
            # "sof_flags": "sof_flags",  # Flags from SOF photometry
            "EXTENDED_CLASS_SOF": "extended_class_sof",  # star galaxy separator using SOF photometry
            "EXTENDED_CLASS_MASH_SOF": "extended_class_mash_sof",  # alternative star galaxy separator using SOF photometry as primary source
            "flags": "flags",
            # "flags_badregions": "flags_badregions",
            # "flags_footprint": "flags_footprint",
            # "flags_foreground": "flags_foreground",
            "FLAGS_GOLD": "flags_gold",
            # "flags_phot": "flags_phot",
            # "hpix_16384": "hpix_16384",
            "tilename": "tilename",  # Name of DES tile
        }
        dummy_columns = {}

        self.process_catalog(
            "des_photometry_catalog",
            "photometry_catalog",
            column_names,
            dummy_columns,
        )


class TXIngestDESY3Footprint(TXIngestMapsHsp):
    """
    Ingest the DES Y3 Footprint maps (incl. badregions, foregrounds etc) from healsparse format
    """

    name = "TXIngestDESY3Footprint"
    parallel = False
    inputs = []

    outputs = [
        ("aux_lens_maps", MapsFile),
    ]

    config_options = {
        **TXIngestMapsHsp.config_options,
        "input_filepaths": StageParameter(list, [""], msg="List of input file paths."),
        "input_labels": StageParameter(list, [""], msg="Labels to give the input maps."),
        "chunk_rows": StageParameter(int, 100_000, msg="Number of rows to process in each chunk."),
    }

    def run(self):
        """
        Run the analysis for this stage.
        """
        print(self.config["input_filepaths"])
        print(self.config["input_labels"])
        assert len(self.config["input_filepaths"]) == len(self.config["input_labels"])

        self.process_maps(self.config["input_filepaths"], self.config["input_labels"], "aux_lens_maps")


class TXIngestDESY3SpeczCat(TXIngestCatalogFits):
    """
    Ingest the spectroscopic catalog used for DES Y3 training of DNF

    file contains spectroscopic redshifts and DES *fluxes*
    """

    name = "TXIngestDESY3SpeczCat"
    parallel = False
    inputs = [
        ("des_specz_catalog", FitsFile),
    ]

    outputs = [
        ("spectroscopic_catalog", HDFFile),
    ]

    config_options = {
        "chunk_rows": StageParameter(int, 100_000, msg="Number of rows to process in each chunk."),
    }

    def run(self):
        """
        Run the analysis for this stage.
        """
        print("Ingesting DES Y3 spec-z catalog")

        # we will only load a subset of columns to save space
        # TODO: these are the y6 magnitudes, i need to match this to y3
        column_names = {
            "mag_g": "sof_cm_mag_corrected_g",
            "mag_r": "sof_cm_mag_corrected_r",
            "mag_i": "sof_cm_mag_corrected_i",
            "mag_z": "sof_cm_mag_corrected_z",
            "mag_err_g": "sof_cm_mag_err_g",
            "mag_err_r": "sof_cm_mag_err_r",
            "mag_err_i": "sof_cm_mag_err_i",
            "mag_err_z": "sof_cm_mag_err_z",
            "redshift": "Z",
        }

        dummy_columns = {}

        self.process_catalog(
            "des_specz_catalog",
            "spectroscopic_catalog",
            column_names,
            dummy_columns,
        )
