import re
import pickle
import numpy as np
from ...base_stage import PipelineStage
from ...data_types import PickleFile, PNGFile


class CLClusterJackknifeComparisonPlots(PipelineStage):
    """
    Compare two covariance estimates stored on the same CLMM cluster
    ensembles -- by default CLMM's healpix jackknife (tan_jk) and the
    kmeans jackknife (tan_kmjk) from CLClusterEnsembleProfiles with
    kmeans_jackknife_njk != 0.

    Produces a z x richness grid of stacked tangential profiles with both
    sets of error bars and a sigma_2/sigma_1 residual panel, plus a single
    scatter plot of sigma_2 vs sigma_1 over all bins and radii.
    """

    name = "CLClusterJackknifeComparisonPlots"
    parallel = False

    inputs = [
        ("cluster_profiles", PickleFile),
    ]

    outputs = [
        ("cluster_jk_comparison_plot", PNGFile),
        ("cluster_jk_error_scatter_plot", PNGFile),
    ]

    config_options = {
        # [reference, comparison]; the residual panel shows sigma_comparison / sigma_reference
        "cov_components": ["tan_jk", "tan_kmjk"],
        "labels": ["healpix JK", "kmeans JK"],
        "x_label": "theta [arcmin]",  # should match bin_units upstream
        "y_label": "g_t",
        "x_jitter_frac": 0.03,
    }

    def run(self):
        import matplotlib

        matplotlib.use("agg")

        cov_components = self.config["cov_components"]
        labels = self.config["labels"]
        if len(cov_components) != 2 or len(labels) != 2:
            raise ValueError("cov_components and labels must each have exactly two entries")
        jitter = self.config["x_jitter_frac"]
        colors = ["C0", "C3"]
        markers = ["o", "s"]
        n_jk_keys = {"tan_jk": "n_jk_healpix", "tan_kmjk": "n_jk_kmeans"}

        with open(self.get_input("cluster_profiles"), "rb") as f:
            data = pickle.load(f)

        bin_info = {}
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
            "cluster_jk_comparison_plot",
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
        ratio_min, ratio_max = np.inf, -np.inf
        # (z-bin index, sigma_ref, sigma_cmp) for the scatter plot
        scatter_points = []

        for key, (i, j) in bin_info.items():
            ax = axes_main[j, i]
            ax_r = axes_resid[j, i]
            entry = data[key]
            edges = entry.get("cluster_bin_edges", {})
            title = (
                f"z=[{edges.get('z_min', '?'):.2f},{edges.get('z_max', '?'):.2f}) "
                f"rich=[{edges.get('rich_min', '?'):.1f},{edges.get('rich_max', '?'):.1f})"
                if edges
                else key
            )
            n_jk = [entry.get(n_jk_keys.get(c, ""), None) for c in cov_components]
            title += f"\nN_cl={entry['n_cl']}  N_jk: " + " / ".join(
                f"{lab}={n if n is not None else '?'}" for lab, n in zip(labels, n_jk)
            )
            ax.set_title(title, fontsize=8)

            ensemble = entry["clmm_cluster_ensemble"]
            if ensemble is None or entry["n_cl"] < 2:
                continue
            missing = [c for c in cov_components if c not in ensemble.cov]
            if missing:
                raise ValueError(f"{key}: covariance(s) {missing} not found; available: {list(ensemble.cov)}")

            radius = np.asarray(ensemble.stacked_data["radius"])
            tan = np.asarray(ensemble.stacked_data["tangential_comp"])
            sigmas = [np.sqrt(np.diag(ensemble.cov[c])) for c in cov_components]

            # Negative stacked points can't be drawn on a log y-axis; drop them
            # (same treatment as CLClusterProfileComparisonPlots).
            positive = tan > 0
            for vi, (sig, label, color, marker) in enumerate(zip(sigmas, labels, colors, markers)):
                r, t, s = radius[positive], tan[positive], sig[positive]
                lower = np.minimum(s, t * (1 - 1e-3))
                x = r * (1 + jitter * (vi - 0.5))
                ax.errorbar(x, t, yerr=[lower, s], fmt=marker, color=color, label=label, markersize=4, capsize=2)
                if len(t):
                    y_min = min(y_min, t.min())
                    y_max = max(y_max, t.max())

            ratio = sigmas[1] / sigmas[0]
            ax_r.plot(radius, ratio, "o-", color="k", markersize=4, linewidth=1)
            finite = np.isfinite(ratio)
            if finite.any():
                ratio_min = min(ratio_min, ratio[finite].min())
                ratio_max = max(ratio_max, ratio[finite].max())

            scatter_points += [(i, a, b) for a, b in zip(sigmas[0], sigmas[1])]

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax_r.set_xscale("log")
            ax_r.axhline(1.0, color="k", linewidth=0.8, alpha=0.5)

        # explicit y-limits: see CLClusterProfileComparisonPlots for the shared log-axis autoscale quirk
        if np.isfinite(y_min) and np.isfinite(y_max):
            axes_main[0, 0].set_ylim(y_min / 5, y_max * 5)
        if np.isfinite(ratio_min) and np.isfinite(ratio_max):
            pad = max(abs(ratio_max - 1), abs(1 - ratio_min), 0.1) * 1.2
            axes_resid[0, 0].set_ylim(1 - pad, 1 + pad)

        handles, leg_labels = axes_main[0, 0].get_legend_handles_labels()
        fig.file.legend(handles, leg_labels, loc="upper right", fontsize=8, ncol=2)

        for ax in axes_resid[-1, :]:
            ax.set_xlabel(self.config["x_label"])
        for ax in axes_main[:, 0]:
            ax.set_ylabel(self.config["y_label"])
        for ax in axes_resid[:, 0]:
            ax.set_ylabel(f"sigma {labels[1]} / {labels[0]}", fontsize=7)

        fig.file.tight_layout()
        fig.close()

        self.plot_error_scatter(scatter_points, labels, n_zbin)

    def plot_error_scatter(self, scatter_points, labels, n_zbin):
        fig = self.open_output("cluster_jk_error_scatter_plot", figsize=(6, 6), wrapper=True)
        ax = fig.file.add_subplot(111)
        if scatter_points:
            zbin, s_ref, s_cmp = (np.array(a) for a in zip(*scatter_points))
            for i in range(n_zbin):
                sel = zbin == i
                if sel.any():
                    ax.scatter(s_ref[sel], s_cmp[sel], s=15, label=f"zbin {i}")
            good = (s_ref > 0) & (s_cmp > 0)
            lo = min(s_ref[good].min(), s_cmp[good].min()) / 1.5
            hi = max(s_ref[good].max(), s_cmp[good].max()) * 1.5
            ax.plot([lo, hi], [lo, hi], "k:")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"sigma ({labels[0]})")
        ax.set_ylabel(f"sigma ({labels[1]})")
        ax.legend(fontsize=8)
        fig.file.tight_layout()
        fig.close()
