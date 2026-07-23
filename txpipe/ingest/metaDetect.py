from ..base_stage import PipelineStage
from ..data_types import ShearCatalog, PhotometryCatalog, HDFFile, FileCollection
from .lsst import process_metadetect_data, sanitize
from .dp1_info import DP1_COSMOLOGY_TRACTS, ALL_TRACTS, DP1_TRACTS
from ceci.config import StageParameter
from ..utils.hdf_tools import h5py_shorten, repack
from ..utils.splitters import MetaDetectSplitter
from ..shear_calibration.names import META_VARIANTS
import numpy as np
import os
import pyarrow.parquet as pq

TXPPIPE_COLUMNS = {
    "g1": "gauss_g1",
    "g2": "gauss_g2",
    "g1_err": "gauss_g1_g1_Cov",
    "g2_err": "gauss_g2_g2_Cov",
    "g_cross": "gauss_g1_g2_Cov",
    "T": "gauss_T",
    "s2n": "gauss_snr",
    "psf_g1_original": "psfOriginal_g1",
    "psf_g2_original": "psfOriginal_g2",
    "psf_T_mean_original": "psfOriginal_T",
    "psf_g1": "gauss_psfReconvolved_g1",
    "psf_g2": "gauss_psfReconvolved_g2",
    "psf_T_mean": "gauss_psfReconvolved_T",
    "object_mask_fraction": "mfrac",
    "id": "shearObjectId",
}


class TXIngestRubinMetaDetect(PipelineStage):
    """
    Initial ingestion of the Rubin MetaDetect catalog
    """

    name = "TXIngestRubinMetaDetect"
    inputs = []
    outputs = [
        ("shear_catalog", ShearCatalog),
    ]
    config_options= {
        "butler_config_file": StageParameter(
            str, 
            "/global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/butler.yaml",
            msg="Path to the LSST butler config file."
        ),
        "cosmology_tracts_only": StageParameter(bool, True, msg="Use only cosmology tracts."),
        "select_field": StageParameter(str, "", msg="Field to select (overrides cosmology_tracts_only)."),
        "select_tracts": StageParameter(list, [], msg="list of tracts (overrides cosmology_tracts_only, but not select_field)."),
        "collections": StageParameter(str, "LSSTComCam/DP1", msg="Butler collections to use."),
        "exclusion_flag": StageParameter(bool, False, msg="Decide if flags are used for exclusion or just flagged."),
        "flag_list": StageParameter(list, ["is_primary"], msg="list of flags to use for combined."),
        "all_columns": StageParameter(bool, False, msg="do we want to save all columns or just the ones TXPipe needs.")
        }

    def run(self):
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
        try:
            butler = Butler(butler_config_file, collections=collections)
        except Exception as e:
            raise RuntimeError(error_msg) from e

        if self.config["select_field"]:
            tracts = DP1_TRACTS[self.config["select_field"]]
        elif self.config["select_tracts"]:
            tracts = self.config["select_tracts"]
        elif self.config["cosmology_tracts_only"]:
            tracts = DP1_COSMOLOGY_TRACTS
        else:
            tracts = ALL_TRACTS

        shear_outfile = self.open_output("shear_catalog")
        group = shear_outfile.create_group("shear")
        shear_outfile["shear"].attrs["catalog_type"] = "metadetect"

        created_files = False
        data_set_refs = butler.query_datasets('object_shear_all')
        n_chunks = len(data_set_refs)
        all_columns_flag = self.config["all_columns"]
        exclusion_flag = self.config["exclusion_flag"]
        flag_list = self.config["flag_list"]
        for i, ref in enumerate(data_set_refs):
            tract = ref.dataId["tract"]
            if tract not in tracts:
                print(f"Skipping chunk {i + 1} / {n_chunks} since tract {tract} is not selected")
                continue

            d = butler.get('object_shear_all',
                           dataId=ref.dataId,
                           )
            chunk_size = len(d)

            if chunk_size == 0:
                print(f"Skipping chunk {i + 1} / {n_chunks} since it is empty")
                continue

            shear_data = process_metadetect_data(d, flag_list, exclusion_flag, 
                                                 full_columns=all_columns_flag)
            if not created_files:
                created_files = True
                variants = {
                    "ns": len(shear_data["ns"]),
                    "1p": len(shear_data["1p"]),
                    "1m": len(shear_data["1m"]),
                    "2p": len(shear_data["2p"]),
                    "2m": len(shear_data["2m"]),
                    }
                columns = list(shear_data["ns"].keys())
                dtypes = {key: shear_data["ns"][key].dtype for key in shear_data["ns"]}
                splitter = MetaDetectSplitter(group, columns, variants, dtypes=dtypes)

            for variant in META_VARIANTS:
                splitter.write_bin(shear_data[variant], variant)
            print(f"Processing chunk {i + 1} / {n_chunks}")

        splitter.finish()
        print("adding in aliases")
        self.aliasing(shear_outfile, group)
        shear_outfile.close()
        print("Repacking files")
        repack(self.get_output("shear_catalog"))

    def aliasing(self, outfile, group):
        g = group
        for variant in ["ns", "1p", "1m", "2p", "2m"]:
            k = g[variant]
            for txname, original in TXPPIPE_COLUMNS.items():
                k[txname] = k[original]

