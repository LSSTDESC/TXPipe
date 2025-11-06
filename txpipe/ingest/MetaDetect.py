from ..base_stage import PipelineStage
from ..data_types import ShearCatalog, PhotometryCatalog, HDFFile, FileCollection
from .lsst import process_metadetect_data
from ceci.config import StageParameter
from ..utils.hdf_tools import h5py_shorten, repack
from ..utils.splitters import MetaDetectSplitter
import numpy as np
import os
import pyarrow.parquet as pq

class TXIngestMetaDetect(PipelineStage):
    """
    Initial ingestion of the Rubin MetaDetect catalog
    """

    name = "TXIngestMetaDetect"
    inputs = []
    outputs = [
        ("shear_catalog", ShearCatalog),
    ]
    config_options= {
        "use_butler": StageParameter(
                                     bool,
                                     True,
                                     msg="Should be left on, unless you got an external file, in that case knock yourself out!"),
        "butler_config_file": StageParameter(
            str, 
            "/global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/butler.yaml",
            msg="Path to the LSST butler config file."
        ),
        "cosmology_tracts_only": StageParameter(bool, True, msg="Use only cosmology tracts."),
        "select_field": StageParameter(str, "", msg="Field to select (overrides cosmology_tracts_only)."),
        "collections": StageParameter(str, "LSSTComCam/DP1", msg="Butler collections to use."),
        "file_path": StageParameter(str, None, msg="if not using a Butler, you need to give a path to the file.")
    }

    def run(self):
        if self.config("use_butler"):
            self.butler_run()
        else:
            self.file_run()

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

        #n = self.get_catalog_size(butler, "ShearObject")
        shear_outfile = self.open_output("shear_catalog")
        group = shear_outfile.create_group("shear")
        shear_outfile["shear"].attrs["catalog_type"] = "metadetect"

        created_files = False
        data_set_refs = butler.query_datasets("ShearObject")
        n_chunks = len(data_set_refs)
        input_columns = self.get_input_columns()

        for i, ref in enumerate(data_set_refs):
            tract = ref.dataId["tract"]
            if tract not in tracts:
                print(f"Skipping chunk {i + 1} / {n_chunks} since tract {tract} is not selected")
                continue

            d = butler.get("ShearObject",
                           dataId=ref.dataId,
                           parameters={"columns": input_columns}
                           )
            chunk_size = len(d)

            if chunk_size == 0:
                print(f"Skipping chunk {i + 1} / {n_chunks} since it is empty")
                continue

            shear_data = process_metadetect_data(d)
            if not created_files:
                created_files = True
                variants = {
                    "00": len(shear_data["00"]),
                    "1p": len(shear_data["1p"]),
                    "1m": len(shear_data["1m"]),
                    "2p": len(shear_data["2p"]),
                    "2m": len(shear_data["2m"]),
                    }
                columns = list(shear_data["00"].keys())
                splitter = MetaDetectSplitter(group, columns, variants)

            for variant in ["00", "1p", "1m", "2p", "2m"]:
                splitter.write_bin(shear_data[variant], variant)
            print(f"Processing chunk {i + 1} / {n_chunks}")

        splitter.finish()
        shear_outfile.close()

    def file_run(self):
        file_path = self.config("file_path")
        if file_path == None:
            raise RuntimeError("You must either use a butler, or specify a file_path to your metadetect catalog.")

        if not os.path.exists(file_path):
            raise RuntimeError("No file where you said it would be.")

        shear_outfile = self.open_output("shear_catalog")
        group = shear_outfile.create_group("shear")
        shear_outfile["shear"].attrs["catalog_type"] = "metadetect"

        created_files = False
        input_columns = self.get_input_columns()

        chunk_size = self.config("chunk_rows")
        pf = pq.ParquetFile(file_path)
        for batch in pf.iter_batches(columns=input_columns, batch_size=chunk_size):
            shear_data = process_metadetect_data(batch)
            if not created_files:
                created_files = True
                variants = {
                    "00": len(shear_data["00"]),
                    "1p": len(shear_data["1p"]),
                    "1m": len(shear_data["1m"]),
                    "2p": len(shear_data["2p"]),
                    "2m": len(shear_data["2m"]),
                    }
                columns = list(shear_data["00"].keys())
                splitter = MetaDetectSplitter(group, columns, variants)

            for variant in ["00", "1p", "1m", "2p", "2m"]:
                splitter.write_bin(shear_data[variant], variant)

        splitter.finish()
        shear_outfile.close()

    def get_input_columns(self):
        input_columns = [
            "shearObjectId",
            "cellId",
            "metaStep",
            "radec",
            "maskFractionObj",
            "maskFractionCell",
            "nEpochCell",
            "g1",
            "g2",
            "gCov",
            "T",
            "SNR",
            "TErr",
            "g1PSFMeta", 
            "g2PSFMeta",
            "g1PSFOrig",
            "g2PSFOrig",
            "TPSFOrig",
            "stdFlux",
            "stdFluxErr",
            "stdFluxT",
            "stdFluxTErr",
            "flags"
        ]
        return input_columns

    def setup_output(self, tag, group, first_chunk):
        f = self.open_output(tag)
        g = f.create_group(group)
        variants =  {
            "00": len(first_chunk["00"]),
            "1p": len(first_chunk["1p"]),
            "1m": len(first_chunk["1m"]),
            "2p": len(first_chunk["2p"]),
            "2m": len(first_chunk["2m"]),
            }
        columns = list(first_chunk["00"].keys())
        splitter = MetaDetectSplitter(g, columns, variants)
        for variant in ["00", "1p", "1m", "2p", "2m"]:
            splitter.write_bin(first_chunk[variant], variant)
        return f

    def write_output(self, outfile, group, data):
        g = outfile[group]
        for variant in ["00", "1p", "1m", "2p", "2m"]:
            k = g[variant]
            for name, col in data[variant].items():
                # replace masked values with nans
                if np.ma.isMaskedArray(col):
                    col = col.filled(np.nan)
                k[name].append(col) #NOT SURE THIS WORKS EITHER TBD


# Outstanding issues! 
# - 1 we don't have a fixed length on the things we add to the seperate variants, hence try append? need to figure out if it works
# - 2 same issue means the h5py shorten thing probably wont work? do we need it to or should we just drop it?
# - 3 write the version where data is taken from a parquet file instead of a butler file!
