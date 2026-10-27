"""
Compare TXPipe theory predictions made with two firecrown versions.

    python plot_comparison.py <old_dir> <new_dir> <old_label> <new_label> <plot_dir>

The directories are outputs of compute_theory.py.
"""
import glob
import os
import sys
import numpy as np
import sacc
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

old_dir, new_dir, old_label, new_label, plot_dir = sys.argv[1:6]
os.makedirs(plot_dir, exist_ok=True)

OLD_COLOR = "#2a78d6"
NEW_COLOR = "#eb6834"
INK = "#3d3d3a"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e4e4e0",
        "grid.linewidth": 0.6,
    }
)

PANELS = {
    "fourier": [
        ("galaxy_shear_cl_ee", r"$C_\ell^{\gamma\gamma}$"),
        ("galaxy_shearDensity_cl_e", r"$C_\ell^{\gamma\delta}$"),
        ("galaxy_density_cl", r"$C_\ell^{\delta\delta}$"),
    ],
    "real": [
        ("galaxy_shear_xi_plus", r"$\xi_+$"),
        ("galaxy_shear_xi_minus", r"$\xi_-$"),
        ("galaxy_shearDensity_xi_t", r"$\gamma_t$"),
        ("galaxy_density_xi", r"$w(\theta)$"),
    ],
}


def load(path):
    return sacc.Sacc.load_fits(path)


def rel_diff(old, new):
    nz = old != 0
    out = np.zeros_like(old)
    out[nz] = new[nz] / old[nz] - 1
    return out


def plot_space(space, bias_case):
    s_old = load(f"{old_dir}/{space}_{bias_case}.sacc")
    s_new = load(f"{new_dir}/{space}_{bias_case}.sacc")
    tag = "ell" if space == "fourier" else "theta"

    # residuals pooled over every bias input form
    cases = sorted(
        os.path.basename(p)[len(space) + 1 : -5] for p in glob.glob(f"{old_dir}/{space}_*.sacc")
    )

    panels = PANELS[space]
    fig, axes = plt.subplots(
        2, len(panels), figsize=(3.4 * len(panels), 5.6), sharex="col",
        gridspec_kw={"height_ratios": [3, 1.2]},
    )
    worst = 0.0
    for j, (dt, label) in enumerate(panels):
        ax, rax = axes[0, j], axes[1, j]
        for k, (t1, t2) in enumerate(s_old.get_tracer_combinations(dt)):
            x, y_old = s_old.get_tag(tag, dt, (t1, t2)), s_old.get_mean(dt, (t1, t2))
            y_new = s_new.get_mean(dt, (t1, t2))
            x = np.array(x)
            if space == "fourier":
                f = x * (x + 1) / (2 * np.pi)
            else:
                f = x
            ax.plot(x, f * y_old, color=OLD_COLOR, lw=2, label=old_label if k == 0 else None)
            ax.plot(x, f * y_new, "o", ms=4, mfc="none", mew=1.0, color=NEW_COLOR,
                    label=new_label if k == 0 else None)

        for case in cases:
            a = load(f"{old_dir}/{space}_{case}.sacc")
            b = load(f"{new_dir}/{space}_{case}.sacc")
            for t1, t2 in a.get_tracer_combinations(dt):
                x = np.array(a.get_tag(tag, dt, (t1, t2)))
                r = rel_diff(a.get_mean(dt, (t1, t2)), b.get_mean(dt, (t1, t2)))
                worst = max(worst, np.abs(r).max())
                rax.plot(x, r, "o", ms=3, color=NEW_COLOR, alpha=0.6)

        ax.set_xscale("log")
        rax.set_xscale("log")
        if space == "fourier":
            ax.set_yscale("log")
            ax.set_ylabel(r"$\ell(\ell+1)$" + label + r"$/2\pi$")
            rax.set_xlabel(r"$\ell$")
        else:
            ax.set_ylabel(r"$\theta\,$" + label + " [arcmin]")
            rax.set_xlabel(r"$\theta$ [arcmin]")
        ax.set_title(dt, fontsize=9, color=INK)
        rax.axhline(0, color=INK, lw=0.8)
        rax.set_ylim(-1e-10, 1e-10)
        rax.set_ylabel("rel. diff." if j == 0 else "")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f"TXPipe theory_3x2pt, {space} space: {old_label} vs {new_label}\n"
        f"top: b0={bias_case}, all bin pairs; bottom: all bias input forms "
        f"({', '.join(cases)}), max |rel. diff.| = {worst:.1e}",
        fontsize=10, color=INK,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    out = f"{plot_dir}/theory_comparison_{space}.png"
    fig.savefig(out, dpi=150)
    print(f"{out}: max |rel diff| = {worst:.3e}")


def plot_bmode():
    a = load(f"{old_dir}/bmode.sacc")
    b = load(f"{new_dir}/bmode.sacc")
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for dt, marker in [("galaxy_shear_cl_ee", "o"), ("galaxy_shear_cl_bb", "s")]:
        x = np.array(a.get_tag("ell", dt, ("source_0", "source_0")))
        f = x * (x + 1) / (2 * np.pi)
        ya, yb = a.get_mean(dt, ("source_0", "source_0")), b.get_mean(dt, ("source_0", "source_0"))
        ax.plot(x, f * ya, "-", color=OLD_COLOR, lw=2, label=f"{dt} ({old_label})")
        ax.plot(x, f * yb, marker, ms=4, mfc="none", color=NEW_COLOR, label=f"{dt} ({new_label})")
        print(f"bmode {dt}: identical = {np.array_equal(ya, yb)}, all zero = {not ya.any()}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$")
    ax.set_title("Unsupported data types (BB) are set to zero", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/theory_comparison_bmode.png", dpi=150)


plot_space("fourier", "float")
plot_space("real", "float")
plot_bmode()
