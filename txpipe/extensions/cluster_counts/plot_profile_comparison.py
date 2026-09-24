import re
import pickle
import numpy as np
from ...base_stage import PipelineStage
from ...data_types import PickleFile, PNGFile

# Fixed input tag -> (display label, color, marker)
VARIANTS = [
    ("cluster_profiles_fid", "Omega_m fid", "C0", "o"),
    ("cluster_profiles_lowOm", "Omega_m -20%", "C1", "s"),
    ("cluster_profiles_highOm", "Omega_m +20%", "C2", "^"),
]


class CLClusterProfileComparisonPlots(PipelineStage):
    """
    Overlay stacked tangential Delta-Sigma(R) profiles from three
    CLClusterEnsembleProfiles runs (radius-based selection under three
    Omega_m cosmologies) on one set of z x richness bin panels, for
    direct visual comparison. Below each main panel, a residual panel
    shows 1 - Delta-Sigma / Delta-Sigma_fid for the Omega_m-varied
    cosmologies.
    """

    name = "CLClusterProfileComparisonPlots"
    parallel = False

    inputs = [(tag, PickleFile) for tag, *_ in VARIANTS]

    outputs = [
        ("cluster_profiles_comparison_plot", PNGFile),
    ]

    config_options = {
        # Must match the cov_type used upstream in CLClusterEnsembleProfiles:
        # sample_covariance -> tan_sc, jackknife_covariance -> tan_jk, bootstrap_covariance -> tan_bs
        "cov_component": "tan_jk",
        "x_jitter_frac": 0.02,  # per-variant log-x offset so overlapping error bars stay legible
    }

    def run(self):
        import matplotlib

        matplotlib.use("agg")

        cov_component = self.config["cov_component"]
        if not cov_component.startswith("tan_"):
            raise ValueError("cov_component config option must start with 'tan_', e.g. 'tan_jk'")
        jitter = self.config["x_jitter_frac"]

        data = {}
        for tag, *_ in VARIANTS:
            with open(self.get_input(tag), "rb") as f:
                data[tag] = pickle.load(f)

        ref_tag = VARIANTS[0][0]
        bin_info = {}
        for key in data[ref_tag]:
            m = re.match(r"(?:bin_)?zbin_(\d+)_richbin_(\d+)$", key)
            if m:
                bin_info[key] = (int(m.group(1)), int(m.group(2)))

        if not bin_info:
            raise ValueError(
                f"No bins matching 'zbin_<i>_richbin_<j>' found in {ref_tag} keys: {list(data[ref_tag].keys())}"
            )

        for tag, *_ in VARIANTS[1:]:
            if set(data[tag].keys()) != set(data[ref_tag].keys()):
                raise ValueError(f"{tag} has a different set of z/richness bins than {ref_tag}")

        n_zbin = max(i for i, j in bin_info.values()) + 1
        n_richbin = max(j for i, j in bin_info.values()) + 1

        # Variants whose deviation from the fid baseline is a cosmology
        # effect (Omega_m varied) rather than a change of selection method.
        cosmology_tags = {tag for tag, label, *_ in VARIANTS if "Omega_m" in label and tag != ref_tag}

        fig = self.open_output(
            "cluster_profiles_comparison_plot",
            figsize=(4.5 * n_zbin, 4.5 * n_richbin),
            wrapper=True,
        )
        gs = fig.file.add_gridspec(2 * n_richbin, n_zbin, height_ratios=[3, 1] * n_richbin, hspace=0.4)

        axes_main = np.empty((n_richbin, n_zbin), dtype=object)
        axes_resid = np.empty((n_richbin, n_zbin), dtype=object)
        for j in range(n_richbin):
            for i in range(n_zbin):
                ax_m = fig.file.add_subplot(gs[2 * j, i], sharex=axes_main[0, 0], sharey=axes_main[0, 0])
                ax_r = fig.file.add_subplot(gs[2 * j + 1, i], sharex=ax_m, sharey=axes_resid[0, 0])
                ax_m.tick_params(labelbottom=False)
                axes_main[j, i] = ax_m
                axes_resid[j, i] = ax_r

        y_min, y_max = np.inf, -np.inf
        resid_absmax = 0.0

        for key, (i, j) in bin_info.items():
            ax = axes_main[j, i]
            ax_r = axes_resid[j, i]
            edges = data[ref_tag][key].get("cluster_bin_edges", {})
            title = (
                f"z=[{edges.get('z_min', '?'):.2f},{edges.get('z_max', '?'):.2f}) "
                f"rich=[{edges.get('rich_min', '?'):.1f},{edges.get('rich_max', '?'):.1f})"
                if edges
                else key
            )
            ax.set_title(title, fontsize=8)

            fid_radius, fid_tan = None, None

            for vi, (tag, label, color, marker) in enumerate(VARIANTS):
                entry = data[tag][key]
                n_cl = entry["n_cl"]
                ensemble = entry["clmm_cluster_ensemble"]
                if ensemble is None or n_cl < 2:
                    continue

                radius = ensemble.stacked_data["radius"]
                tan = np.asarray(ensemble.stacked_data["tangential_comp"])
                tan_err = (
                    np.sqrt(np.diag(ensemble.cov[cov_component]))
                    if cov_component in ensemble.cov
                    else None
                )

                # A negative stacked Delta-Sigma point (shot noise flipping
                # sign in a low-S/N bin, typically the innermost one) can't
                # be shown at all on a log y-axis -- matplotlib renders it as
                # a spike spanning the whole axis rather than clipping
                # cleanly. Drop those points rather than distorting the plot.
                positive = tan > 0
                radius, tan = radius[positive], tan[positive]
                if tan_err is not None:
                    tan_err = tan_err[positive]
                    # Also clip the lower whisker just short of zero for
                    # bins where the error still exceeds the central value.
                    lower = np.minimum(tan_err, tan * (1 - 1e-3))
                    yerr = [lower, tan_err]
                else:
                    yerr = None

                x = radius * (1 + jitter * (vi - 1.5))
                ax.errorbar(
                    x, tan, yerr=yerr, fmt=marker + "-", color=color,
                    label=label, markersize=4, capsize=2,
                )

                if len(tan):
                    y_min = min(y_min, tan.min())
                    y_max = max(y_max, tan.max())

                if tag == ref_tag:
                    fid_radius, fid_tan = radius, tan
                elif tag in cosmology_tags and fid_radius is not None and len(radius) >= 2:
                    # Interpolate onto the fid radius grid (in log-log space,
                    # since Delta-Sigma(R) is close to a power law) so the
                    # residual can be evaluated at the same points as fid
                    # even though each variant's positive-value filtering
                    # above may have dropped different radial bins.
                    lo, hi = radius.min(), radius.max()
                    mask = (fid_radius >= lo) & (fid_radius <= hi)
                    if mask.any():
                        interp_tan = np.exp(
                            np.interp(np.log(fid_radius[mask]), np.log(radius), np.log(tan))
                        )
                        resid = 1 - interp_tan / fid_tan[mask]
                        ax_r.plot(
                            fid_radius[mask], resid, marker + "-", color=color,
                            markersize=4, linewidth=1,
                        )
                        if len(resid):
                            resid_absmax = max(resid_absmax, np.abs(resid).max())

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax_r.set_xscale("log")
            ax_r.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)

        # Setting explicit y-limits from the plotted central values (rather
        # than relying on matplotlib's autoscale) sidesteps a real
        # matplotlib quirk: errorbar() whisker extents on axes sharing a log
        # y-axis (sharey=True) can corrupt the shared autoscaled range by
        # many orders of magnitude even when every plotted value is sane.
        # Error bars that exceed this range are simply clipped at the frame,
        # which is the desired behaviour for occasional huge-uncertainty bins.
        if np.isfinite(y_min) and np.isfinite(y_max):
            axes_main[0, 0].set_ylim(y_min / 5, y_max * 5)

        resid_pad = max(resid_absmax, 0.05) * 1.2
        axes_resid[0, 0].set_ylim(-resid_pad, resid_pad)

        handles, labels = axes_main[0, 0].get_legend_handles_labels()
        fig.file.legend(handles, labels, loc="upper right", fontsize=8, ncol=len(VARIANTS))

        for ax in axes_resid[-1, :]:
            ax.set_xlabel("R [Mpc]")
        for ax in axes_main[:, 0]:
            ax.set_ylabel("delta_sigma(R)")
        for ax in axes_resid[:, 0]:
            ax.set_ylabel("1 - delta_sigma/delta_sigma_fid", fontsize=7)

        fig.file.tight_layout()
        fig.close()
