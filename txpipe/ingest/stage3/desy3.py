from ...data_types import HDFFile, MapsFile, FitsFile, ShearCatalog
from ..base import TXIngestCatalogH5, TXIngestMapsHsp, TXIngestCatalogFits, PipelineStage
from ceci.config import StageParameter
import numpy as np

def to_native_endian(arr: np.ndarray) -> np.ndarray:
    """
    Convert a NumPy array to the machine's native byte order.

    Various downstream stages use NUMBA, which doesn't like
    non-native endianness, and some of the DES Y3 data files seem
    to be big-endian.
    """
    native_dtype = arr.dtype.newbyteorder('=')
    # copy=False avoids copying if already correct
    return arr.astype(native_dtype, copy=False)



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


class TXIngestDESY3Shear(PipelineStage):
    name = "TXIngestDESY3Shear"
    parallel = False
    inputs = [
        ("des_shear_catalog", HDFFile),
    ]

    outputs = [
        ("shear_catalog", ShearCatalog),
    ]

    config_options = {
        "input_group_name": StageParameter(str, "catalog/", msg="Input group name in the HDF5 file."),
        "chunk_rows": StageParameter(int, 100_000, msg="Number of rows to process in each chunk."),
    }

    def des_metacal_flux_to_mag(self, flux, flux_err):
        # Seems to use a zero-point of 30 based on comparison
        # to the gold catalog.
        mag = -2.5 * np.log10(flux) + 30
        mag_err = 2.5 / np.log(10) * (flux_err / flux)
        return mag, mag_err


    def run(self):
        """
        Run the analysis for this stage.
        """
        print("Ingesting DES Y3 Metacal catalog from")


        subgroups_to_suffixes = {
            "unsheared": "",
            "sheared_1p": "_1p",
            "sheared_1m": "_1m",
            "sheared_2p": "_2p",
            "sheared_2m": "_2m",
        }

        single_columns = {
            "unsheared/coadd_object_id": "id",
            "unsheared/ra": "ra",
            "unsheared/dec": "dec",
            "unsheared/psf_T": "psf_T_mean",
            "unsheared/psf_e1": "psf_e1",
            "unsheared/psf_e2": "psf_e2",
            "unsheared/flags": "flags",
        }

        sheared_columns = {
            "T": "T",
            "T_err": "T_err",
            "e_1": "g1",
            "e_2": "g2",
            "snr": "s2n",  # we originally called the TXPipe version "s2n" instead of "snr" 
                           # precisely to be more consistent with metacal, but now apparently
                           # metacal uses "snr" again. After this PR we should update to use "snr" everywhere.
            "weight": "weight",
        }

        fluxes = ["r", "i", "z"]

        # The input and output sections in the files
        fin = self.open_input("des_shear_catalog")
        fout = self.open_output("shear_catalog")
        gin = fin.file["catalog/"]
        gout = fout.file.create_group("shear")
        gout.attrs['catalog_type'] = 'metacal'

        # save all the columns that are only measured once, not on all the
        # sheared versions. In metacal this includes positions and PSF properties.
        for input_col, output_col in single_columns.items():
            d = to_native_endian(gin[input_col][:])
            gout.create_dataset(output_col, data=d, chunks=True, compression="gzip")
        
        # Now save the stuff that is different for each sheared version
        for variant, suffix in subgroups_to_suffixes.items():
            gin_sub = gin[variant]

            # First handle the ones that are just renamed, including the shears
            # themselves and the SNR and weight etc.
            for input_col, output_col in sheared_columns.items():
                d = to_native_endian(gin_sub[input_col][:])
                gout.create_dataset(output_col + suffix, data=d, chunks=True, compression="gzip")
            
            # Now deal with fluxes, which we have to turn into magnitudes.
            # Note that we might change our mind on this and at some point
            # just start storing fluxes.
            for band in fluxes:
                # convert fluxes to mags
                flux = to_native_endian(gin_sub[f"flux_{band}"][:])
                flux_err = to_native_endian(gin_sub[f"flux_err_{band}"][:])

                flux[flux < 0] = 0

                mag, mag_err = self.des_metacal_flux_to_mag(flux, flux_err)

                # save mags and mag_errs
                gout.create_dataset(f"mag_{band}{suffix}", data=mag, chunks=True)
                gout.create_dataset(f"mag_err_{band}{suffix}", data=mag_err, chunks=True)

