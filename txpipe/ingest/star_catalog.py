from ..base_stage import PipelineStage
from ..data_types import HDFFile
from ceci.config import StageParameter


class TXIngestStarCatalog(PipelineStage):
    """
    Ingest the star catalog from the DP2 object table

    This uses lsdb for fast parallel access and writes a single hd5f file for
    all requested bands. Each band is in its own HDF5 group.
    """

    name = "TXIngestStarCatalog"
    parallel = False

    outputs = [
        ("dp2_star_catalog", HDFFile),
    ]

    config_options = {
        "bands": StageParameter(list, ["r", "i", "z"], msg="Bands to get for catalog."),
        "hats_path": str,
        "num_lsdb_daskwokers": int
    }

    def get_star_catalog(self, band_name):
        """
        Function to retrieve a lsdb catalog from the HATS files.
        """

        import lsdb

        columns = [
            "objectId",
            "parentObjectId",
            "tract",
            "patch",
            "coord_ra",
            "coord_dec",
            f"{band_name}_psfMag",
            f"{band_name}_extendedness",
            f"{band_name}_calib_psf_reserved",
            f"{band_name}_calib_psf_used",
            f"{band_name}_moments_psf_flag",
            f"{band_name}_moments_g1",
            f"{band_name}_moments_g2",
            f"{band_name}_moments_trace",
            f"{band_name}_moments_psf_g1",
            f"{band_name}_moments_psf_g2",
            f"{band_name}_moments_psf_trace",
            f"{band_name}_hsm_moments_flag",

            f"{band_name}_hsm_moments_03",
            f"{band_name}_hsm_moments_04",
            f"{band_name}_hsm_moments_12",
            f"{band_name}_hsm_moments_13",
            f"{band_name}_hsm_moments_21",
            f"{band_name}_hsm_moments_22",
            f"{band_name}_hsm_moments_30",
            f"{band_name}_hsm_moments_31",
            f"{band_name}_hsm_moments_40",

            f"{band_name}_hsm_momentsPsf_03",
            f"{band_name}_hsm_momentsPsf_04",
            f"{band_name}_hsm_momentsPsf_12",
            f"{band_name}_hsm_momentsPsf_13",
            f"{band_name}_hsm_momentsPsf_21",
            f"{band_name}_hsm_momentsPsf_22",
            f"{band_name}_hsm_momentsPsf_30",
            f"{band_name}_hsm_momentsPsf_31",
            f"{band_name}_hsm_momentsPsf_40"
        ]

        hats_path = self.config["hats_path"]
        cuts = [(f"{band_name}_calib_psf_reserved", "==", True)]
        catalog = lsdb.open_catalog(hats_path, columns=columns, filters=cuts)

        return catalog

    def run(self):
        """
        For each band, get the catalog and write into a group of the
        HDF5 file while stripping the band name from the variables.
        """

        from dask.distributed import Client
        import lsdb

        # Make Dask cluster
        num_workers = self.config["num_lsdb_daskwokers"]
        print(f"Setting up dask cluster for LSDB with {num_workers} workers.")
        client = Client(n_workers=num_workers, memory_limit="auto")

        # Get the catalog
        bands = self.config["bands"]
        common_variables = ["objectId", "parentObjectId", "tract", "patch", "coord_ra", "coord_dec"]

        with self.open_output("dp2_star_catalog") as f:
            stars = f.create_group("stars", track_order=True)

            for band_name in bands:

                print(f"Querying {band_name}-band...")

                result_table = self.get_star_catalog(band_name).compute()

                print(f"Number of {band_name}-band rows returned: {len(result_table)}")

                band = stars.create_group(band_name, track_order=True)

                # Common variables
                for name in common_variables:
                    band[name] = result_table[name]

                # Band-dependent variables
                prefix = f"{band_name}_"

                for name in result_table.columns:
                    if name.startswith(prefix):
                        band[name.removeprefix(prefix)] = result_table[name]

        # Close cluster
        client.close()
