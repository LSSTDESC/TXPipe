from .base_stage import PipelineStage
from .data_types import RandomsCatalog, PNGFile, TextFile
import numpy as np


class TXJackknifeCenters(PipelineStage):
    """
    This is the pipeline stage that is run to generate the patch centers for
    the Jackknife method.
    """

    name = "TXJackknifeCenters"

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
        cm = plt.cm.get_cmap("Set3")
        sc = plt.scatter(ra, dec, c=patch, cmap=cm, s=20, vmin=0)
        plt.xlabel("RA")
        plt.ylabel("DEC")
        plt.tight_layout()
        jk_plot.close()

    def run(self):
        import treecorr
        import matplotlib

        matplotlib.use("agg")

        input_filename = self.get_input("random_cats")
        output_filename = self.get_output("patch_centers")

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

        #  Generate and write the output patch centres
        print(f"generating {npatch} centers")
        cat.write_patch_centers(output_filename)

        # Should have loaded at this point
        self.plot(np.degrees(cat.ra), np.degrees(cat.dec), cat.patch)
