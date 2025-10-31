from .base_stage import PipelineStage
from .data_types import SACCFile, PNGFile
import numpy as np
import os
from ceci.config import StageParameter


class TXTwoPointPlots(PipelineStage):
    """
    Make plots of the correlation functions and their ratios to theory
    The theory prediction is taken from CCL's calculation.
    """

    name = "TXTwoPointPlots"
    parallel = False
    inputs = [
        ("twopoint_data_real", SACCFile),
        ("twopoint_gamma_x", SACCFile),
    ]
    outputs = [
        ("shear_xi_plus", PNGFile),
        ("shear_xi_minus", PNGFile),
        ("shearDensity_xi", PNGFile),
        ("density_xi", PNGFile),
        ("shearDensity_xi_x", PNGFile),
    ]

    config_options = {
        "wspace": StageParameter(float, 0.05, msg="Width space between subplots."),
        "hspace": StageParameter(float, 0.05, msg="Height space between subplots."),
    }

    def run(self):
        import sacc
        import matplotlib
        import pyccl
        from .plotting import full_3x2pt_plots

        matplotlib.use("agg")
        matplotlib.rcParams["xtick.direction"] = "in"
        matplotlib.rcParams["ytick.direction"] = "in"

        filename = self.get_input("twopoint_data_real")
        s = sacc.Sacc.load_fits(filename)
        nbin_source, nbin_lens = self.read_nbin(s)

        outputs = {
            "galaxy_density_xi": self.open_output(
                "density_xi", figsize=(3.5 * nbin_lens, 3 * nbin_lens), wrapper=True
            ),
            "galaxy_shearDensity_xi_t": self.open_output(
                "shearDensity_xi",
                figsize=(3.5 * nbin_lens, 3 * nbin_source),
                wrapper=True,
            ),
            "galaxy_shear_xi_plus": self.open_output(
                "shear_xi_plus",
                figsize=(3.5 * nbin_source, 3 * nbin_source),
                wrapper=True,
            ),
            "galaxy_shear_xi_minus": self.open_output(
                "shear_xi_minus",
                figsize=(3.5 * nbin_source, 3 * nbin_source),
                wrapper=True,
            ),
        }

        figures = {key: val.file for key, val in outputs.items()}

        full_3x2pt_plots(
            [filename],
            ["twopoint_data_real"],
            figures=figures,
        )

        for fig in outputs.values():
            fig.close()



        outputs = {
            "galaxy_shearDensity_xi_x": self.open_output(
                "shearDensity_xi_x",
                figsize=(3.5 * nbin_lens, 3 * nbin_source),
                wrapper=True,
            ),
        }

        figures = {key: val.file for key, val in outputs.items()}

        filename = self.get_input("twopoint_gamma_x")
        if os.path.exists(filename):
            full_3x2pt_plots([filename], ["twopoint_gamma_x"], figures=figures)

        for fig in outputs.values():
            fig.close()

    def read_nbin(self, s):
        import sacc

        xip = sacc.standard_types.galaxy_shear_xi_plus
        wtheta = sacc.standard_types.galaxy_density_xi

        source_tracers = set()
        for b1, b2 in s.get_tracer_combinations(xip):
            source_tracers.add(b1)
            source_tracers.add(b2)

        lens_tracers = set()
        for b1, b2 in s.get_tracer_combinations(wtheta):
            lens_tracers.add(b1)
            lens_tracers.add(b2)

        # matplotlib crashes if you try to plot with zero bins
        # as you get a 0 x 0 figure. So we set a minimum of 1.
        nbin_source = max(len(source_tracers), 1)
        nbin_lens = max(len(lens_tracers), 1)

        return nbin_source, nbin_lens



class TXTwoPointPlotsFourier(PipelineStage):
    """
    Make plots of the C_ell and their ratios to theory
    """

    name = "TXTwoPointPlotsFourier"
    parallel = False
    inputs = [
        ("summary_statistics_fourier", SACCFile),
        ("twopoint_theory_fourier", SACCFile),
    ]
    outputs = [
        ("shear_cl_ee", PNGFile),
        ("shearDensity_cl", PNGFile),
        ("density_cl", PNGFile),
        ("shear_cl_ee_ratio", PNGFile),
    ]

    config_options = {
        "wspace": StageParameter(float, 0.05, msg="Width space between subplots."),
        "hspace": StageParameter(float, 0.05, msg="Height space between subplots."),
    }

    def read_nbin(self, s):
        sources = []
        lenses = []
        for tn, t in s.tracers.items():
            if "source" in tn:
                sources.append(tn)
            if "lens" in tn:
                lenses.append(tn)
        return len(sources), len(lenses)

    def run(self):
        import sacc
        import matplotlib
        import pyccl
        from .plotting import full_3x2pt_plots

        matplotlib.use("agg")
        matplotlib.rcParams["xtick.direction"] = "in"
        matplotlib.rcParams["ytick.direction"] = "in"

        filename = self.get_input("summary_statistics_fourier")
        s = sacc.Sacc.load_fits(filename)
        nbin_source, nbin_lens = self.read_nbin(s)

        filename_theory = self.get_input("twopoint_theory_fourier")

        outputs = {
            "galaxy_density_cl": self.open_output(
                "density_cl", figsize=(3.5 * nbin_lens, 3 * nbin_lens), wrapper=True
            ),
            "galaxy_shearDensity_cl_e": self.open_output(
                "shearDensity_cl",
                figsize=(3.5 * nbin_lens, 3 * nbin_source),
                wrapper=True,
            ),
            "galaxy_shear_cl_ee": self.open_output(
                "shear_cl_ee",
                figsize=(3.5 * nbin_source, 3 * nbin_source),
                wrapper=True,
            ),
        }

        figures = {key: val.file for key, val in outputs.items()}

        full_3x2pt_plots(
            [filename],
            ["summary_statistics_fourier"],
            figures=figures,
            theory_sacc_files=[filename_theory],
            theory_labels=["Fiducial"],
            xi=False,
            xlogscale=True,
        )

        for fig in outputs.values():
            # weirdly this is where matplotlib crashes
            # when some of the plots are empty
            try:
                fig.close()
            except SystemError:
                continue

        # The same but plotting ratios. The key here is not a mistake -
        # it tells the function calle below what the axis labels etc should be
        outputs = {
            "galaxy_shear_cl_ee": self.open_output(
                "shear_cl_ee_ratio",
                figsize=(3.5 * nbin_source, 3 * nbin_source),
                wrapper=True,
            ),
        }

        figures = {key: val.file for key, val in outputs.items()}

        full_3x2pt_plots(
            [filename],
            ["summary_statistics_fourier"],
            figures=figures,
            theory_sacc_files=[filename_theory],
            theory_labels=["Fiducial"],
            xi=False,
            xlogscale=True,
            ratios=True,
        )

        for fig in outputs.values():
            fig.close()


if __name__ == "__main__":
    PipelineStage.main()


##############





########

class TXTwoPointPlotsTheory( TXTwoPointPlots ):
    name = "TXTwoPointPlotsTheory"
    parallel = False
    inputs = [
        ("twopoint_data_real", SACCFile),
        ("twopoint_gamma_x", SACCFile),
        ("twopoint_theory_real", SACCFile),
    ]
    outputs = [
        ("shear_xi_plus", PNGFile),
        ("shear_xi_minus", PNGFile),
        ("shearDensity_xi", PNGFile),
        ("density_xi", PNGFile),
        ("shear_xi_plus_ratio", PNGFile),
       ("shear_xi_minus_ratio", PNGFile),
       ("shearDensity_xi_ratio", PNGFile),
       ("density_xi_ratio", PNGFile),
        ("shearDensity_xi_x", PNGFile),
    ]

    config_options = {
        "wspace": StageParameter(float, 0.05, msg="Width space between subplots."),
        "hspace": StageParameter(float, 0.05, msg="Height space between subplots."),
    }

    def run(self):
        import sacc
        import matplotlib
        import pyccl
        from .plotting import full_3x2pt_plots

        matplotlib.use("agg")
        matplotlib.rcParams["xtick.direction"] = "in"
        matplotlib.rcParams["ytick.direction"] = "in"

        filename = self.get_input("twopoint_data_real")
        s = sacc.Sacc.load_fits(filename)
        nbin_source, nbin_lens = self.read_nbin(s)
        print(nbin_source, nbin_lens)

        filename_theory = self.get_input("twopoint_theory_real")

        outputs = {
            "galaxy_density_xi": self.open_output(
                "density_xi", figsize=(3.5 * nbin_lens, 3 * nbin_lens), wrapper=True
            ),
            "galaxy_shearDensity_xi_t": self.open_output(
                "shearDensity_xi",
                figsize=(3.5 * nbin_lens, 3 * nbin_source),
                wrapper=True,
            ),
            "galaxy_shear_xi_plus": self.open_output(
                "shear_xi_plus",
                figsize=(3.5 * nbin_source, 3 * nbin_source),
                wrapper=True,
            ),
            "galaxy_shear_xi_minus": self.open_output(
                "shear_xi_minus",
                figsize=(3.5 * nbin_source, 3 * nbin_source),
                wrapper=True,
            ),
        }

        figures = {key: val.file for key, val in outputs.items()}

        full_3x2pt_plots(
            [filename],
            ["twopoint_data_real"],
            figures=figures,
            theory_sacc_files=[filename_theory],
            theory_labels=["Fiducial"],
        )

        for fig in outputs.values():
            fig.close()

        outputs = {
            "galaxy_density_xi": self.open_output(
                "density_xi_ratio",
                figsize=(3.5 * nbin_lens, 3 * nbin_lens),
                wrapper=True,
            ),
            "galaxy_shearDensity_xi_t": self.open_output(
                "shearDensity_xi_ratio",
                figsize=(3.5 * nbin_lens, 3 * nbin_source),
                wrapper=True,
            ),
            "galaxy_shear_xi_plus": self.open_output(
                "shear_xi_plus_ratio",
                figsize=(3.5 * nbin_source, 3 * nbin_source),
                wrapper=True,
            ),
            "galaxy_shear_xi_minus": self.open_output(
                "shear_xi_minus_ratio",
                figsize=(3.5 * nbin_source, 3 * nbin_source),
                wrapper=True,
            ),
        }

        figures = {key: val.file for key, val in outputs.items()}

        full_3x2pt_plots(
            [filename],
            ["twopoint_data_real"],
            figures=figures,
            theory_sacc_files=[filename_theory],
            theory_labels=["Fiducial"],
            ratios=True,
        )

        for fig in outputs.values():
            fig.close()


        outputs = {
            "galaxy_shearDensity_xi_x": self.open_output(
                "shearDensity_xi_x",
                figsize=(3.5 * nbin_lens, 3 * nbin_source),
                wrapper=True,
            ),
        }

        figures = {key: val.file for key, val in outputs.items()}

        filename = self.get_input("twopoint_gamma_x")
        if os.path.exists(filename):
            full_3x2pt_plots([filename], ["twopoint_gamma_x"], figures=figures)

        for fig in outputs.values():
            fig.close()


if __name__ == "__main__":
    PipelineStage.main()