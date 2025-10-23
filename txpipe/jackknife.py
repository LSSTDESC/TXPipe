from .base_stage import PipelineStage
from .data_types import RandomsCatalog, ShearCatalog, PNGFile, TextFile
import numpy as np

class TXJackknifeCenters(PipelineStage):
    """
    Generate jack-knife centers from random catalogs.

    This uses TreeCorr but cuts down the amount of data by taking
    only every n'th point.
    """

    name = "TXJackknifeCenters"
    parallel = False

    inputs = [
        ("random_cats", RandomsCatalog),
    ]
    outputs = [
        ("patch_centers", TextFile),
        ("jk", PNGFile),
    ]
    config_options = {
        "npatch": 10,
        "every_nth": 100,
    }

    def plot(self, ra, dec, patch):
        """
        Plot the jackknife regions.
        """
        import matplotlib

        matplotlib.rcParams["xtick.direction"] = "in"
        matplotlib.rcParams["ytick.direction"] = "in"
        import matplotlib.pyplot as plt

        jk_plot = self.open_output("jk", wrapper=True, figsize=(6.0, 4.5))
        # Choose colormap
        #cm = plt.cm.get_cmap("tab20c")
        rng = np.random.default_rng(12345)
        cm = matplotlib.colors.ListedColormap(rng.random(size=(256,3)))
        sc = plt.scatter(ra, dec, c=patch,cmap=cm,  s=1, vmin=0)
        plt.xlabel("RA")
        plt.ylabel("DEC")
        plt.tight_layout()
        jk_plot.close()
    
    def generate_catalog(self):
        import treecorr
        input_filename = self.get_input("random_cats")

        # Build config info
        npatch = self.config["npatch"]
        every_nth = self.config["every_nth"]
        config = {
            "ext": "randoms",
            "ra_col": "ra",
            "dec_col": "dec",
            "ra_units": "degree",
            "dec_units": "degree",
            "every_nth": every_nth,
            "npatch": npatch,
        }

        # Create the catalog
        cat = treecorr.Catalog(input_filename, config)
        return cat

    def run(self):
        import treecorr
        import matplotlib

        matplotlib.use("agg")

        cat = self.generate_catalog()

        #  Generate and write the output patch centres
        output_filename = self.get_output("patch_centers")
        npatch = self.config["npatch"]
        print(f"generating {npatch} centers")
        cat.write_patch_centers(output_filename)

        # Should have loaded at this point
        self.plot(np.degrees(cat.ra), np.degrees(cat.dec), cat.patch)


class TXJackknifeCentersSource(TXJackknifeCenters):
    """
    Generate jack-knife centers from a shear catalog.
    """
    name = "TXJackknifeCentersSource"
    parallel = False

    inputs = [
        ("shear_catalog", ShearCatalog),
    ]
    def generate_catalog(self):
        import treecorr
        input_filename = self.get_input("shear_catalog")

        with self.open_input("shear_catalog", wrapper=True) as f:
            group = f.get_primary_catalog_group()

        # Build config info
        npatch = self.config["npatch"]
        every_nth = self.config["every_nth"]
        config = {
            "ext": group,
            "ra_col": "ra",
            "dec_col": "dec",
            "ra_units": "degree",
            "dec_units": "degree",
            "every_nth": every_nth,
            "npatch": npatch,
            "file_type": "HDF5",
        }

        # Create the catalog
        cat = treecorr.Catalog(input_filename, config)
        return cat
