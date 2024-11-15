from ...base_stage import PipelineStage
from ...data_types import HDFFile


class CLIngestRedmapper(PipelineStage):
    name = "CLIngestRedmapper"
    inputs = []
    outputs = [("cluster_catalog", HDFFile)]

    config_options = {
        "cat_name": "cosmoDC2_v1.1.4_redmapper_v0.8.1",
    }

    def run(self):
        import GCRCatalogs

        cat = GCRCatalogs.load_catalog(self.config["cat_name"])
        cols = [
            "ra",
            "dec",
            "richness",
            "richness_err",
            "cluster_id",
            "scaleval",
            "redshift",
            "redshift_err",
        ]
        data = cat.get_quantities(cols)

        with self.open_output("cluster_catalog") as f:
            g = f.create_group("clusters")
            for name, col in data.items():
                g.create_dataset(name, data=col)
