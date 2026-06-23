from .base import TXIngestCatalogBase
from ..data_types import ShearCatalog, PhotometryCatalog, HDFFile, FileCollection, FitsFile, ParquetFile
from .lsst import process_photometry_data, process_shear_data
from .dp1 import DP1_COSMOLOGY_FIELDS, DP1_TRACTS, DP1_COSMOLOGY_TRACTS, DP1_FIELD_CENTERS, DP1_SURVEY_PROPERTIES, ALL_TRACTS
from ceci.config import StageParameter
import numpy as np
from ..utils import nanojansky_err_to_mag_ab, nanojansky_to_mag_ab
from ..utils.hdf_tools import h5py_shorten, repack

METASTEP_GROUPS = {"ns": "00", "1p": "1p", "1m": "1m", "2p": "2p", "2m": "2m"}

class TXIngestMetadetect(TXIngestCatalogBase):
    """
    Ingest an metadetect catalog.
    """

    name = "TXIngestMetadetect"
    inputs = [
        ("metadetect_catalog", ParquetFile),
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
        "cosmology_tracts_only": StageParameter(bool, True, msg="Use only cosmology tracts."),
        "select_field": StageParameter(str, "", msg="Field to select (overrides cosmology_tracts_only)."),
        "collections": StageParameter(str, "LSSTComCam/DP1", msg="Butler collections to use."),
        "tracts": StageParameter(str, "", msg="Comma-separated list of tracts to use (empty for all)."),
        "prefix": StageParameter(str, "gauss", msg="prefix indicating the method used to calculate the "),
        "bands": StageParameter(str, "grizy", msg="string of flux bands"),
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
        except:
            raise ImportError(error_msg)
        
        # Configure and create the butler. There are several ways to do this,
        # Here we use a central collective butler yaml file from NERSC.

        butler_config_file = self.config["butler_config_file"]
        collections = self.config["collections"]
        try:
            butler = Butler(butler_config_file, collections=collections)
        except:
            raise RuntimeError(error_msg)

        if self.config["select_field"]:
            tracts = DP1_TRACTS[self.config["select_field"]]
        elif self.config["cosmology_tracts_only"]:
            tracts = DP1_COSMOLOGY_TRACTS
        else:
            tracts = ALL_TRACTS

        #n = self.get_catalog_size(butler, "deep_coadd_cell_anacal_merged")
        #shear_outfile = self.open_output("shear_catalog")
        #group = shear_outfile.create_group("shear")
        #shear_outfile["shear"].attrs["catalog_type"] = "Anacal"

        created_files = False
        data_set_refs = butler.query_datasets('object_shear_all')
        n_chunks = len(data_set_refs)
        input_columns = self.setup_input()
        n = self.get_catalog_size(butler, 'object_shear_all')

        shear_starts = {g: 0 for g in METASTEP_GROUPS.values()}
        shear_outfile = None

        for i, ref in enumerate(data_set_refs):
            tract = ref.dataId["tract"]
            if tract not in tracts:
                print(f"Skipping chunk {i + 1} / {n_chunks} since tract {tract} is not selected")
                continue

            d = butler.get("deep_coadd_cell_anacal_merged", # TODO: change
                           dataId=ref.dataId,
                           parameters={"columns": input_columns}
                           )

            if len(d) == 0:
                print(f"Skipping chunk {i + 1} / {n_chunks} since it is empty")
                continue

            d = self.apply_flag_cuts(d)

            step_data = {}
            for step, group_name in METASTEP_GROUPS.items():
                subset = d[d["metaStep"] == step]
                step_data[group_name] = self.process_mdet_shear_data(subset)

            if not created_files:
                created_files = True
                n_per_step = {g: n for g in METASTEP_GROUPS.values()}
                first_chunk = next(iter(step_data.values()))
                shear_outfile = self.setup_output("shear_catalog", first_chunk, n_per_step)

            for group_name, processed in step_data.items():
                start = shear_starts[group_name]
                end = start + len(processed["ra"])
                self.write_output(shear_outfile, f"shear/{group_name}", processed, start, end)
                shear_starts[group_name] = end

            print(f"Processing chunk {i + 1} / {n_chunks}")

        print("Trimming shear columns:")
        for group_name in METASTEP_GROUPS.values():
            for col in step_data[group_name].keys():
                h5py_shorten(shear_outfile[f"shear/{group_name}"], col, shear_starts[group_name])

        shear_outfile.close()

    def file_run(self):
        cols = self.setup_input()

        file = self.open_input("metadetect_catalog")
        data = file.read(columns=cols).to_pandas()
        data = self.apply_flag_cuts(data)

        step_data = {}
        for step, group_name in METASTEP_GROUPS.items():
            subset = data[data["metaStep"] == step]
            step_data[group_name] = self.process_mdet_shear_data(subset)

        n_per_step = {g: len(d["ra"]) for g, d in step_data.items()}
        first_chunk = next(iter(step_data.values()))
        shear_outfile = self.setup_output("shear_catalog", first_chunk, n_per_step)

        for group_name, processed in step_data.items():
            n = n_per_step[group_name]
            self.write_output(shear_outfile, f"shear/{group_name}", processed, 0, n)

        shear_outfile.close()

    def setup_input(self):
        prefix = self.config["prefix"]

        cols = [
            "shearObjectId",
            "tract",
            "patch",
            "cell_x",
            "cell_y",
            "is_cell_inner",
            "is_patch_inner",
            "is_tract_inner",
            "is_primary",
            "metaStep",
            "x",
            "y",
            "ra",
            "dec",
            "psfOriginal_g1",
            "psfOriginal_g2",
            "psfOriginal_T",
            "mfrac",
            f"{prefix}_psfReconvolved_g1",
            f"{prefix}_psfReconvolved_g2",
            f"{prefix}_psfReconvolved_T",
            f"{prefix}_g1",
            f"{prefix}_g2",
            f"{prefix}_g1_g1_Cov",
            f"{prefix}_g1_g2_Cov",
            f"{prefix}_g2_g2_Cov",
            f"{prefix}_snr",
            f"{prefix}_T",
            f"{prefix}_TErr",
            "pgauss_snr",
            "pgauss_T",
            "pgauss_TErr",
        ]

        bands = self.config["bands"]
        cols += [f"{band}_{prefix}Flux" for band in bands]
        cols += [f"{band}_{prefix}FluxErr" for band in bands]
        cols += [f"{band}_pgaussFlux" for band in bands]
        cols += [f"{band}_pgaussFluxErr" for band in bands]

        # flag columns needed for apply_flag_cuts
        cols += [
            "image_flags",
            "psfOriginal_flags",
            "bmask_flags",
            "ormask_flags",
            f"{prefix}_psfReconvolved_flags",
            f"{prefix}_shape_flags",
            f"{prefix}_object_flags",
            f"{prefix}_flags",
            "pgauss_shape_flags",
            "pgauss_object_flags",
            "pgauss_flags",
        ]
        cols += [f"{band}_{prefix}Flux_flags" for band in bands]
        cols += [f"{band}_pgaussFlux_flags" for band in bands]

        return cols

    def apply_flag_cuts(self, data):
        flag_cols = ['g_pgaussFlux_flags', 'r_pgaussFlux_flags', 'i_pgaussFlux_flags', 'z_pgaussFlux_flags', 'image_flags', 'psfOriginal_flags', 'bmask_flags', 'gauss_psfReconvolved_flags', 'gauss_shape_flags', 'gauss_object_flags', 'gauss_flags', 'pgauss_object_flags']
        if not flag_cols:
            return data
        mask = np.zeros(len(data), dtype=bool)
        for col in flag_cols:
            mask |= data[col].values != 0
        n_cut = mask.sum()
        print(f"Flag cuts: removing {n_cut:,} / {len(data):,} rows ({100 * n_cut / len(data):.1f}%)")
        return data[~mask]
    
    def process_mdet_shear_data(self, data):
        bands = self.config["bands"]
        prefix = self.config["prefix"]
        output = {
                  "ra": data["ra"][:],
                  "dec": data["dec"][:],
                  "mfrac": data["mfrac"][:],
                  "g1": data[f"{prefix}_g1"][:],
                  "g2": data[f"{prefix}_g2"][:],
                  "T": data[f"{prefix}_T"][:],
                  "psf_T_mean": data[f"{prefix}_psfReconvolved_T"][:],
                  "s2n": data[f"{prefix}_snr"][:],
                  "weight": np.ones_like(data["ra"][:]), #TODO:placeholder
                  "flags": np.zeros_like(data["ra"][:]), #TODO:placeholder
                  }

        for band in bands:
            f = data[f"{band}_pgaussFlux"][:]
            f_err = data[f"{band}_pgaussFluxErr"][:]
            output[f"mag_{band}"] = nanojansky_to_mag_ab(f)
            output[f"mag_err_{band}"] = nanojansky_err_to_mag_ab(f, f_err)

            # if band == "i":
            #     output["s2n"] = f / f_err

            # for d in ["_dg1", "_dg2"]:
            #     dd = data[f"{band}_dflux_{s}"+d][:]
            #     output[f"mag_{band}_{d}"] = nanojansky_to_mag_ab(dd)

        return output

    def setup_output(self, tag, first_chunk, n_per_step):
        f = self.open_output(tag)
        shear = f.create_group("shear")
        shear.attrs["catalog_type"] = "metadetect"
        for group_name, n in n_per_step.items():
            g = shear.create_group(group_name)
            for col_name, col in first_chunk.items():
                g.create_dataset(col_name, shape=(n,), dtype=col.dtype)
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