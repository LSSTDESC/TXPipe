from .base import TXIngestCatalogFits
from ..data_types import ShearCatalog, PhotometryCatalog, HDFFile, FileCollection, FitsFile
from .lsst import process_photometry_data, process_shear_data
from .dp1_details import DP1_COSMOLOGY_FIELDS, DP1_TRACTS, DP1_COSMOLOGY_TRACTS, DP1_FIELD_CENTERS, DP1_SURVEY_PROPERTIES, ALL_TRACTS
from ceci.config import StageParameter
import numpy as np
from ..utils import nanojansky_err_to_mag_ab, nanojansky_to_mag_ab, anacal_mag_response
from ..utils.hdf_tools import h5py_shorten, repack

# Suffixes on the merged catalog's photo-z point-estimate columns
# (zmode_0, zmode_1p, zmode_1m, zmode_2p, zmode_2m).  These become
# ``mean_z`` / ``mean_z_{1p,1m,2p,2m}`` in the ingested shear catalog
# and are consumed by TXSourceSelectorAnaCal for the tomographic
# bin-migration term of R_sel via _DataWrapper suffix lookup.
PZ_SUFFIXES = ("0", "1p", "1m", "2p", "2m")

class TXIngestAnacal(TXIngestCatalogFits):
    """
    Ingestion of an anacal catalog, generated from actual Rubin data. This
    stage, will take an anacal catalog, from either the butler, or a file
    (parquet), and ingest it into TXPipe format (HDF5).
    """

    name = "TXIngestAnacal"
    inputs = [
        ("anacal_catalog", FitsFile)
    ]
    outputs = [
        ("shear_catalog", ShearCatalog),
    ]
    config_options = {
        "use_butler": StageParameter(bool, True,
                                     msg="Should be left on, unless you got an external file, in that case knock yourself out!"),
        "butler_config_file": StageParameter(str,
                                             "/global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/butler.yaml",
                                             msg="Path to the LSST butler config file."),
        "butler_object_name": StageParameter(str, "deep_coadd_cell_anacal_merged"),
        "cosmology_tracts_only": StageParameter(bool, True, msg="Use only cosmology tracts."),
        "select_field": StageParameter(str, "", msg="Field to select (overrides cosmology_tracts_only)."),
        "select_tracts": StageParameter(list, [], msg="list of tracts (overrides cosmology_tracts_only, but not select_field)."),
        "collections": StageParameter(str, "LSSTComCam/DP1", msg="Butler collections to use."),
        "tracts": StageParameter(str, "", msg="Comma-separated list of tracts to use (empty for all)."),
        "prefix": StageParameter(str, "fpfs", msg="prefix indicating the method used to calculate the "),
        "bands": StageParameter(list, ["g","r","i","z","y"], msg="string of flux bands"),
        "scale": StageParameter(str, "gauss2", msg="scale radius for the convolution with Gaussian PSF")
    }

    def run(self):
        if self.config["use_butler"]:
            self.butler_run()
        else:
            self.file_run()

        print("repacking files")
        repack(self.get_output("shear_catalog"))

    def butler_run(self):
        error_msg = (
            "The LSST Science Pipelines are not installed in this environment, "
            "or are not configured correctly to access the data. "
            "See the note in the file example/dp1/ingest.yml for how to set "
            "this up on NERSC."
        )
        try:
            from lsst.daf.butler import Butler
        except Exception as e:
            raise ImportError(error_msg) from e

        # Configure and create the butler. There are several ways to do this,
        # Here we use a central collective butler yaml file from NERSC.

        butler_config_file = self.config["butler_config_file"]
        collections = self.config["collections"]
        error_msg2 = error_msg + (
            ' Or there is a typo in the collection you have looked for.'
        )
        try:
            butler = Butler(butler_config_file, collections=collections)
        except Exception as e:
            raise RuntimeError(error_msg2) from e

        if self.config["select_field"]:
            tracts = DP1_TRACTS[self.config["select_field"]]
        elif self.config["select_tracts"]:
            tracts = self.config["select_tracts"]
        elif self.config["cosmology_tracts_only"]:
            tracts = DP1_COSMOLOGY_TRACTS
        else:
            tracts = ALL_TRACTS

        object_name = self.config["butler_object_name"]
        n = self.get_catalog_size(butler, object_name)

        created_files = False
        data_set_refs = butler.query_datasets(object_name)
        n_chunks = len(data_set_refs)
        input_columns = self.setup_input()

        shear_start = 0
        for i, ref in enumerate(data_set_refs):
            tract = ref.dataId["tract"]
            if tract not in tracts:
                print(f"Skipping chunk {i + 1} / {n_chunks} since tract {tract} is not selected")
                continue

            d = butler.get(object_name,
                           dataId=ref.dataId,
                           parameters={"columns": input_columns}
                           )
            chunk_size = len(d)

            if chunk_size == 0:
                print(f"Skipping chunk {i + 1} / {n_chunks} since it is empty")
                continue

            shear_data = self.process_anacal_shear_data(d)
            if not created_files:
                created_files = True
                shear_outfile = self.setup_output("shear_catalog", "shear",
                                                  shear_data, n)

                shear_outfile["shear"].attrs["catalog_type"] = "anacal"

            shear_end = shear_start + len(shear_data["ra"])
            self.write_output(shear_outfile, "shear", shear_data, shear_start,
                              shear_end)

            print(f"Processing chunk {i + 1} / {n_chunks} into rows {shear_start:,} - {shear_end:,}")
            shear_start = shear_end

        print("Trimming shear columns:")
        for col in shear_data.keys():
            print("    ", col)
            h5py_shorten(shear_outfile["shear"], col, shear_end)

        shear_outfile.close()

    def file_run(self):
        tracts = self.config["tracts"]

        n, dtypes = self.get_meta("anacal_catalog")
        cols = self.setup_input()
        prefix = self.config["prefix"]

        file = self.open_input("anacal_catalog")
        data = file[1][cols]
        shear_data = self.process_anacal_shear_data(data)

        shear_outfile = self.setup_output("shear_catalog", "shear", shear_data, n)
        shear_outfile["shear"].attrs["catalog_type"] = "anacal"

        print("Trimming shear columns:")
        for col in shear_data.keys():
            print("    ", col)
            h5py_shorten(shear_outfile["shear"], col, len(shear_data["ra"]))

        shear_outfile.close()

    def setup_input(self):
        prefix = self.config["prefix"]
        scale = self.config["scale"]
        cols = (
            [
                "ra",
                "dec",
                "wsel",
                "mask_value",
                f"{prefix}_e1",
                f"{prefix}_e2",
                f"{prefix}_m00",
                f"{prefix}_m20"
            ])
        cols += ["dwsel"+ suffix for suffix in ["_dg1", "_dg2"]]
        cols += [
                 prefix +delta + suffix
                 for delta in ["_de1", "_de2", "_dm00", "_dm20"]
                 for suffix in ["_dg1", "_dg2"]
                 ]
        bands = self.config["bands"]
        cols += [band + "_flux_" + scale for band in bands]
        cols += [band + "_flux_" + scale + "_err" for band in bands]
        cols += [band + "_dflux_" + scale + suffix for band in bands for suffix in ["_dg1", "_dg2"]]

        # zmode_0 → mean_z; zmode_1p, zmode_1m, zmode_2p, zmode_2m → the
        # metacal-style shifted variants (built with dg=0.01 in xlens'
        # photoZPipe, so TXSourceSelectorAnaCal must use delta_gamma=0.01).
        cols += [f"zmode_{s}" for s in PZ_SUFFIXES]

        return cols

    def process_anacal_shear_data(self, data):
        bands = self.config["bands"]
        s = self.config["scale"]
        prefix = self.config["prefix"]
        # The dm computed e1/e2 columns store the pre-multiplied observable
        # e_meas = wsel · e_raw, and "weight" is uniformly set to 1.
        # This way downstream GGCorrelation with weight_column="weight"
        # computes xi_e =  Σ (wsel_i e_i)(wsel_j e_j) / N_pairs instead of a
        # ⟨wsel wsel⟩-weighted mean of raw shapes.
        # xi_g = xi_e / <R_total>^2

        # The raw shapes and wsel are
        # still exposed as separate columns (wsel, e1_raw, e2_raw) so
        # TXSourceSelectorAnaCal can compute R_shape (⟨wsel · de/dg⟩) and
        # R_detect (⟨(dwsel/dg) · e_raw⟩).
        wsel = data["wsel"][:]
        e1_raw = data[f"{prefix}_e1"][:]
        e2_raw = data[f"{prefix}_e2"][:]
        output = {
            "ra": data["ra"][:],
            "dec": data["dec"][:],
            "weight": np.ones_like(wsel),   # uniform 1 for treecorr
            "wsel": wsel,                   # raw wsel (for R_shape)
            "mask_value": data["mask_value"][:],
            "weight_dg1": data["dwsel_dg1"][:],
            "weight_dg2": data["dwsel_dg2"][:],
            "e1": wsel * e1_raw,            # e_meas ≡ wsel · e_raw
            "e2": wsel * e2_raw,
            "e1_raw": e1_raw,               # raw shape (for R_detect)
            "e2_raw": e2_raw,
            "m00": data[f"{prefix}_m00"][:],
            "m20": data[f"{prefix}_m20"][:],
        }
        for delta in ["de1", "de2", "dm00", "dm20"]:
            output[f"{delta}_dg1"] = data[f"{prefix}_{delta}_dg1"][:]
            output[f"{delta}_dg2"] = data[f"{prefix}_{delta}_dg2"][:]
        for band in bands:
            f = data[f"{band}_flux_{s}"][:]
            f_err = data[f"{band}_flux_{s}_err"][:]
            output[f"mag_{band}"] = nanojansky_to_mag_ab(f)
            output[f"mag_err_{band}"] = nanojansky_err_to_mag_ab(f, f_err)

            if band == "i":
                output["s2n"] = f / f_err

            for d in ["dg1", "dg2"]:
                dd = data[f"{band}_dflux_{s}_"+d][:]
                output[f"mag_{band}_{d}"] = anacal_mag_response(f, dd)
                if band == "i":
                    output[f"ds2n_{d}"] = dd/f_err

        # zmode_0 → mean_z (baseline photo-z used by TXSourceSelectorAnaCal
        # in input_pz mode for tomographic binning).
        # zmode_{1p,1m,2p,2m} → mean_z_{1p,1m,2p,2m} (shifted variants
        # used by the AnaCal calculator's ±γ selection response — the
        # _DataWrapper suffix lookup routes them into the selector when it
        # runs on the shifted samples).
        output["mean_z"] = data["zmode_0"][:]
        for suf in ("1p", "1m", "2p", "2m"):
            output[f"mean_z_{suf}"] = data[f"zmode_{suf}"][:]

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

    def get_catalog_size(self, butler, dataset_type):
        import pyarrow.parquet

        n = 0
        for ref in butler.query_datasets(dataset_type):
            uri = butler.getURI(ref)
            if not uri.path.endswith(".parq"):
                raise ValueError(f"Some data in dataset {dataset_type} was not in parquet format: {uri.path}")
            with pyarrow.parquet.ParquetFile(uri.path) as f:
                n += f.metadata.num_rows
        return n
