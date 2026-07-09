from ..base_stage import PipelineStage
from ..data_types import ShearCatalog, PhotometryCatalog, HDFFile, FileCollection
from .lsst import process_metadetect_data, sanitize
from .dp1_info import DP1_COSMOLOGY_TRACTS, ALL_TRACTS
from ceci.config import StageParameter
from ..utils.hdf_tools import h5py_shorten, repack
from ..utils.splitters import MetaDetectSplitter
from ..shear_calibration.names import META_VARIANTS
import numpy as np
import os
import pyarrow.parquet as pq




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
        "use_butler": StageParameter(
                                     bool,
                                     True,
                                     msg="We only support Butler run for now, so leave this on"),
        "butler_config_file": StageParameter(
            str, 
            "/global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/butler.yaml",
            msg="Path to the LSST butler config file."
        ),
        "cosmology_tracts_only": StageParameter(bool, True, msg="Use only cosmology tracts."),
        "select_field": StageParameter(str, "", msg="Field to select (overrides cosmology_tracts_only)."),
        "select_tracts": StageParameter(list, [], msg="list of tracts (overrides cosmology_tracts_only, but not select_field)."),
        "collections": StageParameter(str, "LSSTComCam/DP1", msg="Butler collections to use."),
        }

    def run(self):
        if self.config["use_butler"]:
            self.butler_run()
        else:
            raise NotImplementedError("we only support Butler run for now")

        # Run h5repack on the file
        print("Repacking files")
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

            shear_data = process_metadetect_data(d)
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

