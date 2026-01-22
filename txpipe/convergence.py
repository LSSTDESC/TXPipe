import numpy as np
from .base_stage import PipelineStage
from .data_types import MapsFile, PNGFile
from ceci.config import StageParameter


class TXConvergenceMaps(PipelineStage):
    """
    Make a convergence map from a source map using Kaiser-Squires

    This uses the wlmassmap library, which is included as a submodule in TXPipe.
    """

    name = "TXConvergenceMaps"
    parallel = False
    inputs = [
        ("source_maps", MapsFile),
    ]

    outputs = [
        ("convergence_maps", MapsFile),
    ]

    config_options = {
        "lmax": StageParameter(int, 0, msg="Maximum multipole for convergence map (0 means 2*nside)."),
        "smoothing_sigma": StageParameter(float, 10.0, msg="Smoothing scale in arcmin."),
    }

    def run(self):
        from wlmassmap.kaiser_squires import healpix_KS_map
        import healpy

        # Open the input file and read bit of metadata.
        # We will pass the entire metadata on to the outside
        source_maps = self.open_input("source_maps", wrapper=True)
        metadata = source_maps.file["maps"].attrs
        nbin_source = metadata["nbin_source"]
        nside = metadata["nside"]

        # There is a flat-sky function in WLMassMap - if we need
        # expose that later than we can do so easily, but for now
        # let's stick with Healpix only
        if metadata["pixelization"] != "healpix":
            raise ValueError("TXConvergenceMaps currently only runs on Healpix maps")

        # Prepare the output file
        output = self.open_output("convergence_maps", wrapper=True)
        # Set up the output group, and copy in
        # metadata from the input file and then
        # our own config.
        group = output.file.create_group("maps")
        group.attrs.update(metadata)
        group.attrs.update(self.config)

        # Use config value if it is non-zero, otherwise
        # use the default 2*nside as in WLMassMap
        lmax = self.config["lmax"] or 2 * nside
        sigma = self.config["smoothing_sigma"]

        # Loop through all our source bins
        maps = list(range(nbin_source)) + ["2D"]
        for i in maps:
            print(f"Producing convergence map for bin {i}")
            # Load input shear maps
            g1 = source_maps.read_map(f"g1_{i}")
            g2 = source_maps.read_map(f"g2_{i}")
            mask = (g1 == healpy.UNSEEN) | (g2 == healpy.UNSEEN)
            gmap = np.vstack([g1, g2])
            print(" - read maps")

            # Run main worker function to get convergence map
            kappa_E, kappa_B = healpix_KS_map(gmap, lmax=lmax, sigma=sigma)
            kappa_E[mask] = healpy.UNSEEN
            kappa_B[mask] = healpy.UNSEEN
            print(" - computed convergence")

            # Save pixels, just where they are valid.
            # Should be same for kappa_E and kappa_B.
            pix = np.where(kappa_E != healpy.UNSEEN)[0]
            output.write_map(f"kappa_E_{i}", pix, kappa_E[pix], metadata)
            output.write_map(f"kappa_B_{i}", pix, kappa_B[pix], metadata)
            print(" - saved")

        output.close()


class TXConvergenceMapPlots(PipelineStage):
    """
    Make plots convergence maps

    Makes a PNG plot of both Kappa_E and Kappa_B
    """

    name = "TXConvergenceMapPlots"
    parallel = False

    inputs = [
        ("convergence_maps", MapsFile),
    ]

    outputs = [
        ("convergence_map", PNGFile),
    ]

    config_options = {
        "projection": StageParameter(
            str, "cart", msg="Projection type for convergence map plots (e.g., cart, moll, orth)."
        ),
    }

    def run(self):
        import matplotlib

        matplotlib.use("agg")
        import matplotlib.pyplot as plt

        # Open input maps file and check number of plots to make
        m = self.open_input("convergence_maps", wrapper=True)
        nbin_source = m.file["maps"].attrs["nbin_source"]

        # Open a PNG output file, specifying output size
        fig = self.open_output("convergence_map", wrapper=True, figsize=(5 * nbin_source, 5))

        # 2 x nbin for kappa_E and kappa_B
        _, axes = plt.subplots(2, nbin_source, num=fig.file.number)

        # Loop through bins
        for i in range(nbin_source):
            # Set current axis to use (i.e. subplot)
            plt.sca(axes[0, i])
            # and plot E-mode kappa map
            m.plot(f"kappa_E_{i}", view=self.config["projection"])

            # and B-mode
            plt.sca(axes[1, i])
            m.plot(f"kappa_B_{i}", view=self.config["projection"])

        # This saves the full plot
        fig.close()
