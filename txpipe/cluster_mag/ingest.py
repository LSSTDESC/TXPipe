import numpy as np
from ..base_stage import PipelineStage
from ..data_types import HDFFile

class CMIngestHalosCosmoDC2(PipelineStage):
    name = "CMIngestHalosCosmoDC2"
    parallel = False
    inputs = []
    outputs = [("cluster_mag_halo_catalog", HDFFile)]
    config_options = {
        "cat_name": "cosmoDC2_v1.1.4_image",
        "halo_mass_min": 0.5e13,
        "initial_size": 100_000,
        "ra_range": [50.0, 73.1],
        "dec_range": [-45.0, -27.0],
    }

    def run(self):
        import GCRCatalogs

        # Configuration options
        mass_min = self.config["halo_mass_min"]
        cat_name = self.config["cat_name"]
        sz = self.config["initial_size"]
        ra_range = self.config['ra_range']
        dec_range = self.config['dec_range']

        # Open the cosmoDC2 catalog
        overwrite = {
            "check_md5": False,
            "check_size": False,
            "ensure_meta_consistent": False,
        }
        cat = GCRCatalogs.load_catalog(cat_name, config_overwrite=overwrite)

        # Selection of data we will read from it below
        cols = ["halo_mass", "redshift", "ra", "dec", "halo_id"]
        filters = [
            f"halo_mass > {mass_min}",
            "is_central == True",
            f"ra > {ra_range[0]}",
            f"ra < {ra_range[1]}",
            f"dec > {dec_range[0]}",
            f"dec < {dec_range[1]}",
        ]

        # Create output data file with extensible data sets
        f = self.open_output("cluster_mag_halo_catalog")
        g = f.create_group("halos")
        g.create_dataset("halo_mass", (sz,), maxshape=(None,), dtype="f8", chunks=True)
        g.create_dataset("redshift", (sz,), maxshape=(None,), dtype="f8", chunks=True)
        g.create_dataset("ra", (sz,), maxshape=(None,), dtype="f8", chunks=True)
        g.create_dataset("dec", (sz,), maxshape=(None,), dtype="f8", chunks=True)
        g.create_dataset("halo_id", (sz,), maxshape=(None,), dtype="i8", chunks=True)

        # Prepare the iterator to loop through GCR
        it = cat.get_quantities(cols, filters=filters, return_iterator=True)

        # s is the start index for the next data chunk
        s = 0
        for data in it:
            # e is the end index for this data chunk
            e = s + data["ra"].size
            print(f"Read data chunk {s:,} - {e:,}")

            # Expand the data sets if we are exceeding the current
            # size.  Grow by 50% each time.
            if e > sz:
                sz = int(1.5 * e)
                print(f"Resizing data to {sz:,}")
                for col in cols:
                    g[col].resize((sz,))

            # Output this chunk of data to the file
            for col in cols:
                g[col][s:e] = data[col]

            # Update the starting index for the next chunk
            s = e

        print(f"Ingestion complete. Resizing to final halo count {e:,}")
        # Now we have finished we can truncate any
        # excess space in the output data
        for col in cols:
            g[col].resize((e,))

        # And that's all.
        f.close()
