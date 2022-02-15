from .base_stage import PipelineStage
from .data_types import FiducialCosmology, SACCFile, PNGFile
import numpy as np


class TXTwoPointPlots(PipelineStage):
    """
    Make plots of the correlation functions and their ratios to
    a fiducial theory prediction.
    """

    name = "TXTwoPointPlots"
    inputs = [
        ("twopoint_data_real", SACCFile),
        ("fiducial_cosmology", FiducialCosmology),  # For example lines
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
        "wspace": 0.05,
        "hspace": 0.05,
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

        filename = self.get_input("twopoint_gamma_x")

        outputs = {
            "galaxy_shearDensity_xi_x": self.open_output(
                "shearDensity_xi_x",
                figsize=(3.5 * nbin_lens, 3 * nbin_source),
                wrapper=True,
            ),
        }

        figures = {key: val.file for key, val in outputs.items()}

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

        return len(source_tracers), len(lens_tracers)

    def read_bins(self, s):
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

        sources = list(sorted(source_tracers))
        lenses = list(sorted(lens_tracers))

        return sources, lenses

    def get_theta_xi_err(self, D):
        """
        For a given datapoint D, returns theta, xi, err,
        after masking for positive errorbars
        (sometimes there are NaNs).
        """
        theta = np.array([d.get_tag("theta") for d in D])
        xi = np.array([d.value for d in D])
        err = np.array([d.get_tag("error") for d in D])
        w = err > 0
        theta = theta[w]
        xi = xi[w]
        err = err[w]

        return theta, xi, err

    def get_theta_xi_err_jk(self, s, dt, src1, src2):
        """
        In this case we want to get the JK errorbars,
        which are stored in the covariance, so we want to
        load a particular covariance block, given a dataype dt.
        Returns theta, xi, err,
        after masking for positive errorbars
        (sometimes there are NaNs).
        """
        theta_jk, xi_jk, cov_jk = s.get_theta_xi(dt, src1, src2, return_cov=True)
        err_jk = np.sqrt(np.diag(cov_jk))
        w_jk = err_jk > 0
        theta_jk = theta_jk[w_jk]
        xi_jk = xi_jk[w_jk]
        err_jk = err_jk[w_jk]

        return theta_jk, xi_jk, err_jk

    def plot_shear_shear(self, s, sources):
        import sacc
        import matplotlib.pyplot as plt

        xip = sacc.standard_types.galaxy_shear_xi_plus
        xim = sacc.standard_types.galaxy_shear_xi_minus
        nsource = len(sources)

        theta = s.get_tag("theta", xip)
        tmin = np.min(theta)
        tmax = np.max(theta)

        coord = (
            lambda dt, i, j: (nsource + 1 - j, i) if dt == xim else (j, nsource - 1 - i)
        )

        for dt in [xip, xim]:
            for i, src1 in enumerate(sources[:]):
                for j, src2 in enumerate(sources[:]):
                    D = s.get_data_points(dt, (src1, src2))

                    if len(D) == 0:
                        continue

                    ax = plt.subplot2grid((nsource + 2, nsource), coord(dt, i, j))

                    scale = 1e-4

                    theta = np.array([d.get_tag("theta") for d in D])
                    xi = np.array([d.value for d in D])
                    err = np.array([d.get_tag("error") for d in D])
                    w = err > 0
                    theta = theta[w]
                    xi = xi[w]
                    err = err[w]

                    plt.errorbar(
                        theta, xi * theta / scale, err * theta / scale, fmt="."
                    )
                    plt.xscale("log")
                    plt.ylim(-1, 1)
                    plt.xlim(tmin, tmax)

                    if dt == xim:
                        if j > 0:
                            ax.set_xticklabels([])
        plots = ["xi", "xi_err"]

        for plot in plots:
            plot_output = self.open_output(
                f"shear_{plot}", wrapper=True, figsize=(2.5 * nsource, 2 * nsource)
            )

            for dt in [xip, xim]:
                for i, src1 in enumerate(sources[:]):
                    for j, src2 in enumerate(sources[:]):
                        D = s.get_data_points(dt, (src1, src2))

                        if len(D) == 0:
                            continue

                        ax = plt.subplot2grid((nsource + 2, nsource), coord(dt, i, j))

                        theta, xi, err = self.get_theta_xi_err(D)
                        if plot == "xi":
                            scale = 1e-4
                            plt.errorbar(
                                theta,
                                xi * theta / scale,
                                err * theta / scale,
                                fmt=".",
                                capsize=1.5,
                                color=self.colors[0],
                            )
                            plt.ylim(-30, 30)
                            ylabel_xim = r"$\theta \cdot \xi_{-} \cdot 10^4$"
                            ylabel_xip = r"$\theta \cdot \xi_{+} \cdot 10^4$"

                        if plot == "xi_err":
                            theta_jk, xi_jk, err_jk = self.get_theta_xi_err_jk(
                                s, dt, src1, src2
                            )
                            plt.plot(
                                theta,
                                err,
                                label="Shape noise",
                                lw=2.0,
                                color=self.colors[0],
                            )
                            plt.plot(
                                theta_jk,
                                err_jk,
                                label="Jackknife",
                                lw=2.0,
                                color=self.colors[1],
                            )
                            ylabel_xim = r"$\sigma\, (\xi_{-})$"
                            ylabel_xip = r"$\sigma\, (\xi_{-})$"

                        plt.xscale("log")
                        plt.xlim(tmin, tmax)

                        if dt == xim:
                            if j > 0:
                                ax.set_xticklabels([])
                            else:
                                plt.xlabel(r"$\theta$ (arcmin)")

                            if i == nsource - 1:
                                ax.yaxis.tick_right()
                                ax.yaxis.set_label_position("right")
                                ax.set_ylabel(ylabel_xim)
                            else:
                                ax.set_yticklabels([])
                        else:
                            ax.set_xticklabels([])
                            if i == nsource - 1:
                                ax.set_ylabel(ylabel_xip)
                            else:
                                ax.set_yticklabels([])

                        # props = dict(boxstyle='square', lw=1.,facecolor='white', alpha=1.)
                        plt.text(
                            0.03,
                            0.93,
                            f"[{i},{j}]",
                            transform=plt.gca().transAxes,
                            fontsize=10,
                            verticalalignment="top",
                        )  # , bbox=props)

            if plot == "xi_err":
                plt.legend()
            plt.tight_layout()
            plt.subplots_adjust(
                hspace=self.config["hspace"], wspace=self.config["wspace"]
            )
            plot_output.close()

    def plot_shear_density(self, s, sources, lenses):
        import sacc
        import matplotlib.pyplot as plt

        gammat = sacc.standard_types.galaxy_shearDensity_xi_t
        nsource = len(sources)
        nlens = len(lenses)

        theta = s.get_tag("theta", gammat)
        tmin = np.min(theta)
        tmax = np.max(theta)

        plots = ["xi", "xi_err"]
        for plot in plots:
            plot_output = self.open_output(
                f"shearDensity_{plot}", wrapper=True, figsize=(3 * nlens, 2 * nsource)
            )

            for i, src1 in enumerate(sources):
                for j, src2 in enumerate(lenses):

                    D = s.get_data_points(gammat, (src1, src2))

                    if len(D) == 0:
                        continue

                    ax = plt.subplot2grid((nsource, nlens), (i, j))

                    if plot == "xi":
                        scale = 1e-2
                        theta, xi, err = self.get_theta_xi_err(D)
                        plt.errorbar(
                            theta,
                            xi * theta / scale,
                            err * theta / scale,
                            fmt=".",
                            capsize=1.5,
                            color=self.colors[0],
                        )
                        plt.ylim(-2, 2)
                        ylabel = r"$\theta \cdot \gamma_t \cdot 10^2$"

                    if plot == "xi_err":
                        theta, xi, err = self.get_theta_xi_err(D)
                        theta_jk, xi_jk, err_jk = self.get_theta_xi_err_jk(
                            s, gammat, src1, src2
                        )
                        plt.plot(
                            theta,
                            err,
                            label="Shape noise",
                            lw=2.0,
                            color=self.colors[0],
                        )
                        plt.plot(
                            theta_jk,
                            err_jk,
                            label="Jackknife",
                            lw=2.0,
                            color=self.colors[1],
                        )
                        ylabel = r"$\sigma\,(\gamma_t)$"

                    plt.xscale("log")
                    plt.xlim(tmin, tmax)

                    if i == nsource - 1:
                        plt.xlabel(r"$\theta$ (arcmin)")
                    else:
                        ax.set_xticklabels([])

                    if j == 0:
                        plt.ylabel(ylabel)
                    else:
                        ax.set_yticklabels([])

                    # props = dict(boxstyle='square', lw=1.,facecolor='white', alpha=1.)
                    plt.text(
                        0.03,
                        0.93,
                        f"[{i},{j}]",
                        transform=plt.gca().transAxes,
                        fontsize=10,
                        verticalalignment="top",
                    )  # , bbox=props)

            if plot == "xi_err":
                plt.legend()
            plt.tight_layout()
            plt.subplots_adjust(
                hspace=self.config["hspace"], wspace=self.config["wspace"]
            )
            plot_output.close()

    def plot_density_density(self, s, lenses):
        import sacc
        import matplotlib.pyplot as plt

        wtheta = sacc.standard_types.galaxy_density_xi
        nlens = len(lenses)

        theta = s.get_tag("theta", wtheta)
        tmin = np.min(theta)
        tmax = np.max(theta)

        plots = ["xi", "xi_err"]
        for plot in plots:
            plot_output = self.open_output(
                f"density_{plot}", wrapper=True, figsize=(3 * nlens, 2 * nlens)
            )

            for i, src1 in enumerate(lenses[:]):
                for j, src2 in enumerate(lenses[:]):

                    D = s.get_data_points(wtheta, (src1, src2))

                    if len(D) == 0:
                        continue

                    ax = plt.subplot2grid((nlens, nlens), (i, j))

                    if plot == "xi":
                        scale = 1
                        theta, xi, err = self.get_theta_xi_err(D)
                        plt.errorbar(
                            theta,
                            xi * theta / scale,
                            err * theta / scale,
                            fmt=".",
                            capsize=1.5,
                            color=self.colors[0],
                        )
                        ylabel = r"$\theta \cdot w$"
                        plt.ylim(-1, 1)

                    if plot == "xi_err":
                        theta, xi, err = self.get_theta_xi_err(D)
                        theta, xi, err = self.get_theta_xi_err(D)
                        theta_jk, xi_jk, err_jk = self.get_theta_xi_err_jk(
                            s, wtheta, src1, src2
                        )
                        plt.plot(
                            theta,
                            err,
                            label="Shape noise",
                            lw=2.0,
                            color=self.colors[0],
                        )
                        plt.plot(
                            theta_jk,
                            err_jk,
                            label="Jackknife",
                            lw=2.0,
                            color=self.colors[1],
                        )
                        ylabel = r"$\sigma\,(w)$"

                    plt.xscale("log")
                    plt.xlim(tmin, tmax)

                    if j > 0:
                        ax.set_xticklabels([])
                    else:
                        plt.xlabel(r"$\theta$ (arcmin)")

                    if i == 0:
                        plt.ylabel(ylabel)
                    else:
                        ax.set_yticklabels([])

                    # props = dict(boxstyle='square', lw=1.,facecolor='white', alpha=1.)
                    plt.text(
                        0.03,
                        0.93,
                        f"[{i},{j}]",
                        transform=plt.gca().transAxes,
                        fontsize=10,
                        verticalalignment="top",
                    )  # , bbox=props)

            if plot == "xi_err":
                plt.legend()
            plt.tight_layout()
            plt.subplots_adjust(
                hspace=self.config["hspace"], wspace=self.config["wspace"]
            )
            plot_output.close()


class TXTwoPointPlotsFourier(PipelineStage):

    name = "TXTwoPointPlotsFourier"
    inputs = [
        ("summary_statistics_fourier", SACCFile),
        ("fiducial_cosmology", FiducialCosmology),  # For example lines
        ("twopoint_theory_fourier", SACCFile),
    ]
    outputs = [
        ("shear_cl_ee", PNGFile),
        ("shearDensity_cl", PNGFile),
        ("density_cl", PNGFile),
        ("shear_cl_ee_ratio", PNGFile),
    ]

    config_options = {
        "wspace": 0.05,
        "hspace": 0.05,
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
            fig.close()

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
