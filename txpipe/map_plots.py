from .data_types import MapsFile, PNGFile
from .base_stage import PipelineStage
import sys


class TXMapPlots(PipelineStage):
    """
    Make plots of all the available maps

    This makes plots of:
    - depth
    - lens density
    - shear
    - flag values
    - PSF
    - mask
    - bright object counts

    If one map fails for any reason it is just skipped.
    """

    name = "TXMapPlots"
    parallel = False
    inputs = [
        ("source_maps", MapsFile),
        ("lens_maps", MapsFile),
        ("density_maps", MapsFile),
        ("mask", MapsFile),
        ("aux_source_maps", MapsFile),
        ("aux_lens_maps", MapsFile),
    ]

    outputs = [
        ("depth_map", PNGFile),
        ("lens_map", PNGFile),
        ("shear_map", PNGFile),
        ("flag_map", PNGFile),
        ("psf_map", PNGFile),
        ("mask_map", PNGFile),
        ("bright_object_map", PNGFile),
    ]
    config_options = {
        # can also set moll
        "projection": "cart",
        "rot180": False, 
        "debug": False,
    }

    def run(self):
        # PSF tests
        import matplotlib

        matplotlib.use("agg")
        import matplotlib.pyplot as plt

        # Plot from each file separately, just
        # to organize this file a bit
        methods = [
            self.aux_source_plots,
            self.aux_lens_plots,
            self.source_plots,
            self.lens_plots,
            self.mask_plots,
        ]

        # We don't want this to fail if some maps are missing.
        for m in methods:
            try:
                m()
            except:
                if self.config["debug"]:
                    raise
                sys.stderr.write(f"Failed to make maps with method {m.__name__}")

    def aux_source_plots(self):
        import matplotlib.pyplot as plt
        
        if self.get_input("aux_source_maps") == "none":
            # Make empty plots if no data available, so that the
            # pipeline thinks it is complete.
            for map_type in ["flag_map", 'psf_map']:
                with self.open_output(map_type, wrapper=True) as f:
                    plt.title(f'No map generated for {map_type}')
            return

        m = self.open_input("aux_source_maps", wrapper=True)

        # Get these two config options from the maps where
        # they were originally saved
        nbin_source = m.file["maps"].attrs["nbin_source"]
        flag_max = m.file["maps"].attrs["flag_exponent_max"]

        # Flag count plots - flags are assumed to be bitsets, so
        # we make maps of 1, 2, 4, 8, 16, ...
        fig = self.open_output("flag_map", wrapper=True, figsize=(5 * flag_max, 5))
        for i in range(flag_max):
            plt.subplot(1, flag_max, i + 1)
            f = 2**i
            m.plot(f"flags/flag_{f}", view=self.config["projection"], rot180=self.config["rot180"])
        fig.close()

        # PSF plots - 2 x n, for g1 and g2
        fig = self.open_output("psf_map", wrapper=True, figsize=(5 * nbin_source, 10))
        _, axes = plt.subplots(2, nbin_source, squeeze=False, num=fig.file.number)
        for i in range(nbin_source):
            plt.sca(axes[0, i])
            m.plot(f"psf/g1_{i}", view=self.config["projection"], rot180=self.config["rot180"])
            plt.sca(axes[1, i])
            m.plot(f"psf/g2_{i}", view=self.config["projection"], rot180=self.config["rot180"])
        fig.close()

    def aux_lens_plots(self):
        import matplotlib.pyplot as plt
        if self.get_input("aux_lens_maps") == "none":
            for map_type in ["depth_map", "bright_object_map"]:
                with self.open_output(map_type, wrapper=True) as f:
                    plt.title(f'No map generated for {map_type}')
            return

        m = self.open_input("aux_lens_maps", wrapper=True)

        # Depth plots
        with self.open_output("depth_map", wrapper=True, figsize=(5, 5)) as fig:
            m.plot("depth/depth", view=self.config["projection"], rot180=self.config["rot180"])

        # Bright objects
        with self.open_output("bright_object_map", wrapper=True, figsize=(5, 5)) as fig:
            m.plot("bright_objects/count", view=self.config["projection"], rot180=self.config["rot180"])

    def source_plots(self):
        import matplotlib.pyplot as plt

        if self.get_input("source_maps") == "none":
            for map_type in ["shear_map"]:
                with self.open_output(map_type, wrapper=True) as f:
                    plt.title(f'No map generated for {map_type}')
            return

        m = self.open_input("source_maps", wrapper=True)

        nbin_source = m.file["maps"].attrs["nbin_source"]

        fig = self.open_output("shear_map", wrapper=True, figsize=(5 * nbin_source, 10))

        # Plot 2 x nbin, g1 and g2
        _, axes = plt.subplots(2, nbin_source, squeeze=False, num=fig.file.number)

        for i in range(nbin_source):
            # g1
            plt.sca(axes[0, i])
            m.plot(f"g1_{i}", view=self.config["projection"], rot180=self.config["rot180"], min=-0.1, max=0.1)

            # g2
            plt.sca(axes[1, i])
            m.plot(f"g2_{i}", view=self.config["projection"], rot180=self.config["rot180"], min=-0.1, max=0.1)
        fig.close()

    def lens_plots(self):
        import matplotlib.pyplot as plt

        if self.get_input("lens_maps") == "none":
            for map_type in ["lens_map"]:
                with self.open_output(map_type, wrapper=True) as f:
                    plt.title(f'No map generated for {map_type}')
            return

        m = self.open_input("lens_maps", wrapper=True)
        rho = self.open_input("density_maps", wrapper=True)
        nbin_lens = m.file["maps"].attrs["nbin_lens"]

        # Plot both density and ngal as 2 x n
        fig = self.open_output("lens_map", wrapper=True, figsize=(5 * nbin_lens, 5))
        _, axes = plt.subplots(2, nbin_lens, squeeze=False, num=fig.file.number)

        for i in range(nbin_lens):
            plt.sca(axes[0, i])
            m.plot(f"ngal_{i}", view=self.config["projection"], rot180=self.config["rot180"])
            plt.sca(axes[1, i])
            rho.plot(f"delta_{i}", view=self.config["projection"], rot180=self.config["rot180"])
        fig.close()

    def mask_plots(self):
        import matplotlib.pyplot as plt

        if self.get_input("mask") == "none":
            for map_type in ["mask_map"]:
                with self.open_output(map_type, wrapper=True) as f:
                    plt.title(f'No map generated for {map_type}')
            return

        m = self.open_input("mask", wrapper=True)

        fig = self.open_output("mask_map", wrapper=True, figsize=(5, 5))
        m.plot("mask", view=self.config["projection"], rot180=self.config["rot180"])
        fig.close()

class TXMapPlotsSSI(TXMapPlots):
    """
    Make plots of all the available maps that use SSI inputs

    This makes plots of:
    - depth (using meas mag)
    - depth (using true mag)
    - depth (using detection fraction)
    """
    name = "TXMapPlotsSSI"

    inputs = [
        ("aux_ssi_maps", MapsFile),
    ]

    outputs = [
        ("depth_ssi_meas_map", PNGFile),
        ("depth_ssi_true_map", PNGFile),
        ("depth_ssi_det_prob_map", PNGFile),
    ]

    def run(self):
        # PSF tests
        import matplotlib

        matplotlib.use("agg")
        import matplotlib.pyplot as plt

        # Plot from each file separately, just
        # to organize this file a bit
        methods = [
            self.aux_ssi_plots,
        ]

        # We don't want this to fail if some maps are missing.
        for m in methods:
            try:
                m()
            except:
                if self.config["debug"]:
                    raise
                sys.stderr.write(f"Failed to make maps with method {m.__name__}")


    def aux_ssi_plots(self):
        import matplotlib.pyplot as plt
        if self.get_input("aux_ssi_maps") == "none":
            # Make empty plots if no data available, so that the
            # pipeline thinks it is complete.
            for map_type in ["depth_ssi_meas_map", "depth_ssi_true_map", "depth_det_prob_map"]:
                with self.open_output(map_type, wrapper=True) as f:
                    plt.title(f'No map generated for {map_type}')
            return

        m = self.open_input("aux_ssi_maps", wrapper=True)

        # Depth plots (measured magnitude)
        with self.open_output("depth_ssi_meas_map", wrapper=True, figsize=(5, 5)) as fig:
            m.plot("depth_meas/depth", view=self.config["projection"], rot180=self.config["rot180"])

        # Depth plots (true magnitude)
        with self.open_output("depth_ssi_true_map", wrapper=True, figsize=(5, 5)) as fig:
            m.plot("depth_true/depth", view=self.config["projection"], rot180=self.config["rot180"])

        # Depth plots (true magnitude)
        with self.open_output("depth_ssi_det_prob_map", wrapper=True, figsize=(5, 5)) as fig:
            m.plot("depth_det_prob/depth", view=self.config["projection"], rot180=self.config["rot180"])