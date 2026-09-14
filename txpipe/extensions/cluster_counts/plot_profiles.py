import re
import pickle
import numpy as np
from ...base_stage import PipelineStage
from ...data_types import PickleFile, PNGFile


class CLClusterProfilePlots(PipelineStage):
    """
    Plot the stacked cluster weak-lensing profiles produced by
    CLClusterEnsembleProfiles: tangential and cross component vs radius,
    with error bars from the ensemble covariance, one panel per
    redshift x richness bin.
    """

    name = "CLClusterProfilePlots"
    parallel = False

    inputs = [
        ("cluster_profiles", PickleFile),
    ]

    outputs = [
        ("cluster_profiles_plot", PNGFile),
    ]

    config_options = {
        # Must match the cov_type used upstream in CLClusterEnsembleProfiles:
        # sample_covariance -> tan_sc, jackknife_covariance -> tan_jk, bootstrap_covariance -> tan_bs
        "cov_component": "tan_jk",
        "show_individual_clusters": True,
    }

    def run(self):
        import matplotlib
        # might crash if we dont first specify this 'headless' backend
        matplotlib.use("agg")

        cov_component = self.config["cov_component"]
        if not cov_component.startswith("tan_"):
            raise ValueError("cov_component config option must start with 'tan_', e.g. 'tan_jk'")
        cross_component = "cross_" + cov_component[len("tan_"):]
        show_individual = self.config["show_individual_clusters"]

        with open(self.get_input("cluster_profiles"), "rb") as f:
            data = pickle.load(f)

        bin_info = {}
        # a lot of variance in bin naming so im trying to handle that here
        for key in data:
            m = re.match(r"(?:bin_)?zbin_(\d+)_richbin_(\d+)$", key)
            if m:
                bin_info[key] = (int(m.group(1)), int(m.group(2)))

        if not bin_info:
            raise ValueError(
                f"No bins matching 'zbin_<i>_richbin_<j>' found in cluster_profiles keys: {list(data.keys())}"
            )

        n_zbin = max(i for i, j in bin_info.values()) + 1
        n_richbin = max(j for i, j in bin_info.values()) + 1

        fig = self.open_output(
            "cluster_profiles_plot",
            figsize=(4 * n_zbin, 3 * n_richbin),
            wrapper=True,
        )
        axes = fig.file.subplots(n_richbin, n_zbin, sharex=True, sharey=True, squeeze=False)
        # quick way to shut off auto range
        y_min, y_max = np.inf, -np.inf

        for key, (i, j) in bin_info.items():
            ax = axes[j, i]
            entry = data[key]
            n_cl = entry["n_cl"]
            ensemble = entry["clmm_cluster_ensemble"]
            edges = entry.get("cluster_bin_edges", {})
            title = (
                f"z=[{edges.get('z_min', '?'):.2f},{edges.get('z_max', '?'):.2f}) "
                f"rich=[{edges.get('rich_min', '?'):.1f},{edges.get('rich_max', '?'):.1f})\n"
                f"N_cl={n_cl}"
                if edges
                else f"{key}\nN_cl={n_cl}"
            )

            if ensemble is None or n_cl < 2:
                ax.set_title(title, fontsize=8)
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                continue

            radius = ensemble.stacked_data["radius"]
            tan = np.asarray(ensemble.stacked_data["tangential_comp"])
            cross = np.asarray(ensemble.stacked_data["cross_comp"])

            tan_err = np.sqrt(np.diag(ensemble.cov[cov_component])) if cov_component in ensemble.cov else None
            cross_err = np.sqrt(np.diag(ensemble.cov[cross_component])) if cross_component in ensemble.cov else None
            if tan_err is None:
                print(f"[{key}] covariance component '{cov_component}' not found, plotting without error bars")

            # A non-positive stacked point (shot noise flipping sign in a
            # low-S/N bin, typically the innermost one) can't be shown at all
            # on a log y-axis. Matplotlib renders it as a spike spanning
            # the whole axis rather than clipping cleanly. Drop those points,
            # and clip the remaining lower whiskers just short of zero.
            tan_positive = tan > 0
            radius_tan, tan = radius[tan_positive], tan[tan_positive]
            if tan_err is not None:
                tan_err = tan_err[tan_positive]
                tan_err = [np.minimum(tan_err, tan * (1 - 1e-3)), tan_err]

            cross_positive = cross > 0
            radius_cross, cross = radius[cross_positive], cross[cross_positive]
            if cross_err is not None:
                cross_err = cross_err[cross_positive]
                cross_err = [np.minimum(cross_err, cross * (1 - 1e-3)), cross_err]

            if show_individual:
                for k in range(len(ensemble)):
                    ax.plot(
                        ensemble.data["radius"][k],
                        ensemble.data["tangential_comp"][k],
                        color="C0",
                        alpha=0.15,
                        linewidth=1,
                    )

            ax.errorbar(radius_tan, tan, yerr=tan_err, fmt="o-", color="C0", label="tangential")
            ax.errorbar(radius_cross, cross, yerr=cross_err, fmt="s-", color="C1", label="cross")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(title, fontsize=8)

            for arr in (tan, cross):
                if len(arr):
                    y_min = min(y_min, arr.min())
                    y_max = max(y_max, arr.max())

        if np.isfinite(y_min) and np.isfinite(y_max):
            axes[0, 0].set_ylim(y_min / 5, y_max * 5)

        axes[0, 0].legend(fontsize=8)
        for ax in axes[-1, :]:
            ax.set_xlabel("radius")
        for ax in axes[:, 0]:
            ax.set_ylabel("delta_sigma / g_t")

        fig.file.tight_layout()
        fig.close()
