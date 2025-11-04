from base import TXIngestCatalogFits
from ..data_types import ShearCatalog, PhotometryCatalog, HDFFile, FileCollection, FitsFile
from .lsst import process_photometry_data, process_shear_data
from ceci.config import StageParameter
import numpy as np
from ..utils import nanojansky_err_to_mag_ab, nanojansky_to_mag_ab,
from ..utils.hdf_tools import h5py_shorten, repack

class TXIngestAnacal(TXIngestCatalogFits):
    """
    Ingest an anacal catalog!, 
    """

    name = "TXIngestAnacal"
    input = [
        ("Aanacal_catalog", FitsFile)
    ]
    outputs = [
        ("photometry_catalog", PhotometryCatalog),
        ("shear_catalog", ShearCatalog),
        ("exposures", HDFFile),
        ("survey_propety_maps", FileCollection),
    ]
    config_options = {
        "tracts": StageParameter(str, "", msg="Comma-separated list of tracts to use (empty for all)."),
        "prefix": StageParameter(str, "fpfs", msg="prefix indicating the method used to calculate the "),
        "bands": StageParameter(str, "grizy", msg="string of flux bands"),
        "scale": StageParameter(str, "gauss2", msg="scale radius for the convolution with Gaussian PSF")
    }

    def run(self):
        tracts = self.config["tracts"]
        file_path = self.config["file_path"]

        n, dtypes = self.get_meta(f"{file_path}/anacal_anacal_table.fits")
        cols = self.setup_input("shear_catalog")
        prefix = self.config["prefix"]

        file = self.open_input("Anacal_catalog")
        data = file[1][cols]
        shear_data = self.process_anacal_shear_data(data)

        shear_outfile = self.setup_output("shear_catalog", "shear", shear_data, n)
        shear_outfile["shear"].attrs["catalog_type"] = "Anacal"
        #self.write_output(shear_outfile, "shear", shear_data)

        print("Trimming shear columns:")
        for col in shear_data.keys():
            print("    ", col)
            h5py_shorten(shear_outfile["shear"], col, len(shear_data["ra"]))
        
        shear_outfile.close()


    def setup_input(self):
        prefix = self.config["prefix"]
        cols = (
            [
                "ra",
                "dec", 
                "wsel",
                "wdet",
                f"{prefix}_e1",
                f"{prefix}_e2",
            ])
        cols += ["dwsel"+ suffix for suffix in ["_dg1", "_dg2"]]
        cols += [
                 prefix +delta + suffix 
                 for delta in ["_de1", "_de2"]
                 for suffix in ["_dg1", "_dg2"]
                 ]
        bands = self.config["bands"]
        for i in range(4):
            cols += [band + "_flux_gauss" + i for band in bands]
            cols += [band + "_flux_gauss" + i + "_err" for band in bands]
            cols += [
                     band + "_dflux_gauss" + i + suffix
                     for band in bands
                     for suffix in ["_dg1", "_dg2"]
                     ]
        return cols
    
    def process_anacal_shear_data(data):
        bands = self.config["bands"]
        s = self.config["scale"]
        output = {
                  "ra": data["ra"],
                  "dec": data["dec"],
                  "weight":data["wsel"], 
                  "weight_detection": data["wdet"],
                  "weight_dg1": data["dwsel_dg1"],
                  "weight_dg2": data["dwsel_dg2"],
                  }
        for band in bands:
            f = data[f"{band}_flux_{s}"]
            f_err = data[f"{band}_flux_{s}_err"]
            output[f"mag_{band}"] = nanojansky_to_mag_ab(f)
            output[f"mag_err_{band}"] = nanojansky_err_to_mag_ab(f, f_err)

            if band == "i":
                output["s2n"] = f / f_err
            
            for d in ["_dg1", "_dg2"]:
                dd = data[f"{band}_dflux_{s}"+d]
                output[f"mag_{band}_{d}"] = nanojansky_to_mag_ab(dd)
                
        return output

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

