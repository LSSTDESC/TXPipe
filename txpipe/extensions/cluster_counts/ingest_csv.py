"""
Ingest a cluster catalog from a CSV file.

CLIngestRedmapper (ingest.py) reads the cosmoDC2 redMaPPer catalog through
GCRCatalogs. For Rubin DP2 the external cluster catalogs (DES Y3 redMaPPer, ACT,
HSC, MCXC-II, Planck, SPT ECS cross-matched to DP2 objects) come as a CSV file for now,
so this stage converts the CSV into the "clusters" HDF5 group that
CLClusterBinningRedshiftRichness and CLClusterShearCatalogs read. The CSV
column names are config options; the defaults match the DP2 cross-match file.
"""

import numpy as np
from ...base_stage import PipelineStage
from ...data_types import CSVFile, HDFFile
from ceci.config import StageParameter


class CLIngestClusterCSV(PipelineStage):
    """Convert a CSV cluster catalog into TXPipe's cluster_catalog HDF5 format."""

    name = "CLIngestClusterCSV"
    inputs = [("cluster_catalog_csv", CSVFile)]
    outputs = [("cluster_catalog", HDFFile)]

    config_options = {
        # CSV column for each column of the output "clusters" group
        "cluster_id_col": StageParameter(str, "catalog_id_clusters", msg="CSV column with the cluster ID"),
        "ra_col": StageParameter(str, "ra", msg="CSV column with the cluster RA [deg]"),
        "dec_col": StageParameter(str, "dec", msg="CSV column with the cluster Dec [deg]"),
        "redshift_col": StageParameter(str, "redshift", msg="CSV column with the cluster redshift"),
        "redshift_err_col": StageParameter(str, "redshift_unc_clusters", msg="CSV column with the redshift uncertainty"),
        "richness_col": StageParameter(str, "richness", msg="CSV column with the cluster richness"),
        "richness_err_col": StageParameter(str, "richness_unc_clusters", msg="CSV column with the richness uncertainty"),
    }

    def run(self):
        import pandas as pd

        df = pd.read_csv(self.get_input("cluster_catalog_csv"))
        print(f"Read {len(df)} clusters from {self.get_input('cluster_catalog_csv')}")

        with self.open_output("cluster_catalog") as f:
            g = f.create_group("clusters")
            for out_col in ["cluster_id", "ra", "dec", "redshift", "redshift_err", "richness", "richness_err"]:
                g.create_dataset(out_col, data=df[self.config[f"{out_col}_col"]].to_numpy())
            # CLClusterBinningRedshiftRichness copies a "scaleval" column through but
            # nothing downstream uses it; set it to 1 as in the cosmoDC2 catalogs.
            g.create_dataset("scaleval", data=np.ones(len(df)))
