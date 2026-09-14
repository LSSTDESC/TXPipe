import re
import pickle
import numpy as np
from ...base_stage import PipelineStage
from ...data_types import PickleFile, PNGFile

# Fixed input tag -> (display label, color, marker)
VARIANTS = [
    ("cluster_profiles_radius_fid", "radius, Ωm fid", "C0", "o"),
    ("cluster_profiles_radius_lowOm", "radius, Ωm -20%", "C1", "s"),
    ("cluster_profiles_radius_highOm", "radius, Ωm +20%", "C2", "^"),
    ("cluster_profiles_angle", "fixed angle", "C3", "D"),
]


class CLClusterProfileComparisonPlots(PipelineStage):
    """
    Overlay stacked tangential Delta-Sigma(R) profiles from four
    CLClusterEnsembleProfiles runs (radius-based selection under three
    cosmologies, plus fixed-angle selection) on one set of z x richness
    bin panels, for direct visual comparison.
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

        fig = self.open_output(
            "cluster_profiles_comparison_plot",
            figsize=(4.5 * n_zbin, 3.5 * n_richbin),
            wrapper=True,
        )
        axes = fig.file.subplots(n_richbin, n_zbin, sharex=True, sharey=True, squeeze=False)

        y_min, y_max = np.inf, -np.inf

        for key, (i, j) in bin_info.items():
            ax = axes[j, i]
            edges = data[ref_tag][key].get("cluster_bin_edges", {})
            title = (
                f"z=[{edges.get('z_min', '?'):.2f},{edges.get('z_max', '?'):.2f}) "
                f"rich=[{edges.get('rich_min', '?'):.1f},{edges.get('rich_max', '?'):.1f})"
                if edges
                else key
            )
            ax.set_title(title, fontsize=8)

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

            ax.set_xscale("log")
            ax.set_yscale("log")

        # Setting explicit y-limits from the plotted central values (rather
        # than relying on matplotlib's autoscale) sidesteps a real
        # matplotlib quirk: errorbar() whisker extents on axes sharing a log
        # y-axis (sharey=True) can corrupt the shared autoscaled range by
        # many orders of magnitude even when every plotted value is sane.
        # Error bars that exceed this range are simply clipped at the frame,
        # which is the desired behaviour for occasional huge-uncertainty bins.
        if np.isfinite(y_min) and np.isfinite(y_max):
            axes[0, 0].set_ylim(y_min / 5, y_max * 5)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.file.legend(handles, labels, loc="upper right", fontsize=8, ncol=len(VARIANTS))

        for ax in axes[-1, :]:
            ax.set_xlabel("R [Mpc]")
        for ax in axes[:, 0]:
            ax.set_ylabel("delta_sigma(R)")

        fig.file.tight_layout()
        fig.close()
