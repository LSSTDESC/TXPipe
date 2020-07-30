from .data_types import MapsFile, PNGFile
from .base_stage import PipelineStage


class TXMapPlots(PipelineStage):
    """
    Make plots of all the available maps.
    """

    name = "TXMapPlots"

    inputs = [
        ("source_maps", MapsFile),
        ("lens_maps", MapsFile),
        ("density_maps", MapsFile),
        ("mask", MapsFile),
        ("aux_maps", MapsFile),
        ("convergence_maps", MapsFile),
    ]

    outputs = [
        ("depth_map", PNGFile),
        ("lens_map", PNGFile),
        ("shear_map", PNGFile),
        ("flag_map", PNGFile),
        ("psf_map", PNGFile),
        ("mask_map", PNGFile),
        ("bright_object_map", PNGFile),
        ("convergence_map", PNGFile),
    ]
    config_options = {
        # can also set Moll
        "projection": "cart",
    }

    def run(self):
        # PSF tests
        import matplotlib

        matplotlib.use("agg")
        import matplotlib.pyplot as plt

        # Plot from each file separately, just
        # to organize this file a bit
        self.aux_plots()
        self.source_plots()
        self.lens_plots()
        self.mask_plots()
        self.convergence_plots()

    def aux_plots(self):
        import matplotlib.pyplot as plt

        m = self.open_input("aux_maps", wrapper=True)

        # Get these two config options from the maps where
        # they were originally saved
        nbin_source = m.file["maps"].attrs["nbin_source"]
        flag_max = m.file["maps"].attrs["flag_exponent_max"]

        # Depth plots
        fig = self.open_output("depth_map", wrapper=True, figsize=(5, 5))
        m.plot("depth/depth", view=self.config["projection"])
        fig.close()

        # Bright objects
        fig = self.open_output("bright_object_map", wrapper=True, figsize=(5, 5))
        m.plot("bright_objects/count", view=self.config["projection"])
        fig.close()

        # Flag count plots - flags are assumed to be bitsets, so
        # we make maps of 1, 2, 4, 8, 16, ...
        fig = self.open_output("flag_map", wrapper=True, figsize=(5 * flag_max, 5))
        for i in range(flag_max):
            plt.subplot(1, flag_max, i + 1)
            f = 2 ** i
            m.plot(f"flags/flag_{f}", view=self.config["projection"])
        fig.close()

        # PSF plots - 2 x n, for g1 and g2
        fig = self.open_output("psf_map", wrapper=True, figsize=(5 * nbin_source, 10))
        _, axes = plt.subplots(2, nbin_source, squeeze=False, num=fig.file.number)
        for i in range(nbin_source):
            plt.sca(axes[0, i])
            m.plot(f"psf/g1_{i}", view=self.config["projection"])
            plt.sca(axes[1, i])
            m.plot(f"psf/g2_{i}", view=self.config["projection"])
        fig.close()

    def source_plots(self):
        import matplotlib.pyplot as plt

        m = self.open_input("source_maps", wrapper=True)

        nbin_source = m.file["maps"].attrs["nbin_source"]

        fig = self.open_output("shear_map", wrapper=True, figsize=(5 * nbin_source, 10))

        # Plot 2 x nbin, g1 and g2
        _, axes = plt.subplots(2, nbin_source, squeeze=False, num=fig.file.number)

        for i in range(nbin_source):
            # g1
            plt.sca(axes[0, i])
            m.plot(f"g1_{i}", view=self.config["projection"])

            # g2
            plt.sca(axes[1, i])
            m.plot(f"g2_{i}", view=self.config["projection"])
        fig.close()

    def lens_plots(self):
        import matplotlib.pyplot as plt

        m = self.open_input("lens_maps", wrapper=True)
        rho = self.open_input("density_maps", wrapper=True)
        nbin_lens = m.file["maps"].attrs["nbin_lens"]

        # Plot both density and ngal as 2 x n
        fig = self.open_output("lens_map", wrapper=True, figsize=(5 * nbin_lens, 5))
        _, axes = plt.subplots(2, nbin_lens, squeeze=False, num=fig.file.number)

        for i in range(nbin_lens):
            plt.sca(axes[0, i])
            m.plot(f"ngal_{i}", view=self.config["projection"])
            plt.sca(axes[1, i])
            rho.plot(f"delta_{i}", view=self.config["projection"])
        fig.close()

    def mask_plots(self):
        import matplotlib.pyplot as plt

        m = self.open_input("mask", wrapper=True)

        fig = self.open_output("mask_map", wrapper=True, figsize=(5, 5))
        m.plot("mask", view=self.config["projection"])
        fig.close()

    def convergence_plots(self):
        import matplotlib.pyplot as plt

        m = self.open_input("convergence_maps", wrapper=True)

        nbin_source = m.file["maps"].attrs["nbin_source"]

        fig = self.open_output("convergence_map",
                               wrapper=True,
                               figsize=(5 * nbin_source, 5))

        # 2 x nbin for kappa_E and kappa_B
        _, axes = plt.subplots(2, nbin_source, num=fig.file.number)

        for i in range(nbin_source):
            # Set current axis to use (i.e. subplot)
            plt.sca(axes[0, i])
            # and plot E-mode kappa map
            m.plot(f"kappa_E_{i}", view=self.config["projection"])

            # B-mode
            plt.sca(axes[1, i])
            m.plot(f"kappa_B_{i}", view=self.config["projection"])
        fig.close()
