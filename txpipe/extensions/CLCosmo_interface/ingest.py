import numpy as np
from ...base_stage import PipelineStage
from ...data_types import HDFFile


class CMSkysimPhotometry(PipelineStage):
    """
    Ingest noise-free photometry from SkySim GCR

    """
    name = "CMSkysimPhotometry"
    inputs = []
    outputs = [("photometry_catalog", HDFFile)]
    
    config_options = {
        # this is 10 year 5 sigma depth from https://www.lsst.org/scientists/keynumbers
        "r_limit": 27.5,
        # getting the catalog size takes ages in GCR, so
        # if you know it alreay then better to put it here.
        # this is a max size before cuts
        "cat_size": 8_503_061_280,
        # if you change this then remember to set cat_size above
        "cat_name": "skysim5000_v1.1.1"

    }

    def run(self):
        import GCRCatalogs
        gc = GCRCatalogs.load_catalog(self.config["cat_name"])

        # avoid measuring cat length if known
        N = self.config['cat_size']
        if N == 0:
            N = len(gc)


        # Columns we need from the cosmo simulation,
        # and the new names we give them
        cols = {
            'mag_true_u_lsst': 'u_mag',
            'mag_true_g_lsst': 'g_mag',
            'mag_true_r_lsst': 'r_mag',
            'mag_true_i_lsst': 'i_mag',
            'mag_true_z_lsst': 'z_mag',
            'mag_true_y_lsst': 'y_mag',
            'ra': 'ra',
            'dec': 'dec',
            'galaxy_id': 'id',
            'redshift_true': 'redshift_true',
        }

        photo_file = self.setup_output(cols, N)
        photo_grp = photo_file["photometry"]

        # Set up the iterator to load catalog
        r_limit = self.config["r_limit"]
        filters = [f"mag_true_r_lsst < {r_limit}"]
        it = gc.get_quantities(cols, filters = filters, return_iterator=True)
        nfile = len(gc._file_list) if hasattr(gc, '_file_list') else "?"

        # Loop through the input data
        s = 0
        for i, data in enumerate(it):
            print(f"Loading chunk {i+1}/{nfile}")
            # save each chunk of data to output
            s = self.save_chunk(data, cols, photo_grp, s)

        # Resize all the columns because we filtered
        # out some objects so the end of the catalog will
        # be empty
        for col in photo_grp.keys():
            photo_grp[col].resize((s,))


    def save_chunk(self, data, cols, photo_grp, s):

        # Range of this data chunk
        n = len(data["ra"])
        e = s + n

        # rename columns
        data = {new: data[old] for old, new in cols.items()}

        # Make zero error columns
        zeros = np.zeros(n)
        for b in "ugrizy":
            data[f"{b}_mag_err"] = zeros

        # save outputs to file
        for col in cols.values():
            photo_grp[f"{col}"][s:e] = data[col]

        # new index
        return e


    def setup_output(self, cols, N):
        f = self.open_output('photometry_catalog')
        g = f.create_group("photometry")
        for col in cols.values():
            dtype = int if col == "id" else float
            g.create_dataset(col, dtype=dtype, shape=(N,), maxshape=(N,))
        return f




class CMIngestHalosCosmoDC2(PipelineStage):
    """
    Load halos from CosmoDC2, by querying the central galaxies.
    """
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

        
        
        
        
class CMIngestHalosSkySim(PipelineStage):
    """
    Load halos from SkySim, by querying the central galaxies.
    """
    name = "CMIngestHalosSkySim"
    parallel = False
    inputs = []
    outputs = [("cluster_SkySim_halo_catalog", HDFFile)]
    config_options = {
        "cat_name": "skysim5000_v1.1.1_small",
        "halo_mass_min": 1e12,
        "initial_size": 100_000,
        "ra_range": [60, 73],
        "dec_range": [-47.0, -33.0],
    }

    def run(self):
        import GCRCatalogs

        # Configuration options
        mass_min = self.config["halo_mass_min"]
        cat_name = self.config["cat_name"]
        sz = self.config["initial_size"]
        ra_range = self.config['ra_range']
        dec_range = self.config['dec_range']

        # Open the SkySim catalog
        overwrite = {
            "check_md5": False,
            "check_size": False,
            "ensure_meta_consistent": False,
        }
        cat = GCRCatalogs.load_catalog(cat_name, config_overwrite=overwrite)

        # Selection of data we will read from it below
        cols = ["halo_mass", "redshift", "ra", "dec", "halo_id", "baseDC2/sod_halo_mass", "baseDC2/sod_halo_radius"]
        filters = [
            f"halo_mass > {mass_min}",
            "is_central == True",
            f"ra > {ra_range[0]}",
            f"ra < {ra_range[1]}",
            f"dec > {dec_range[0]}",
            f"dec < {dec_range[1]}",
        ]

        # Create output data file with extensible data sets
        f = self.open_output("cluster_SkySim_halo_catalog")
        g = f.create_group("halos")
        g.create_dataset("halo_mass", (sz,), maxshape=(None,), dtype="f8", chunks=True)
        g.create_dataset("redshift", (sz,), maxshape=(None,), dtype="f8", chunks=True)
        g.create_dataset("ra", (sz,), maxshape=(None,), dtype="f8", chunks=True)
        g.create_dataset("dec", (sz,), maxshape=(None,), dtype="f8", chunks=True)
        g.create_dataset("halo_id", (sz,), maxshape=(None,), dtype="i8", chunks=True)
        g.create_dataset("baseDC2/sod_halo_mass", (sz,), maxshape=(None,), dtype="f8", chunks=True)
        g.create_dataset("baseDC2/sod_halo_radius", (sz,), maxshape=(None,), dtype="f8", chunks=True)

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
