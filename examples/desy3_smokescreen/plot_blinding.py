"""
Plot the shift that Smokescreen applied to the DES Y3 data vector.

    python examples/desy3_smokescreen/plot_blinding.py [--raw FILE] [--blinded FILE] [--output FILE]

For each statistic, the top panel shows the raw and blinded data for one
tomographic bin pair, and the bottom panel the shift of every point in units
of its error bar, with one line per bin pair.

The shift depends on the drawn parameter values, so this plot is for checking
the example only: in a real blinded analysis you would not look at it.
"""
import argparse

import numpy as np
import sacc
import matplotlib
import matplotlib.ticker

matplotlib.use("agg")
import matplotlib.pyplot as plt

# Statistic, label, and the bin pair shown in the top panel
PANELS = [
    ("galaxy_shear_xi_plus", r"$\xi_+$", ("source_3", "source_3")),
    ("galaxy_shear_xi_minus", r"$\xi_-$", ("source_3", "source_3")),
    ("galaxy_shearDensity_xi_t", r"$\gamma_t$", ("source_3", "lens_1")),
    ("galaxy_density_xi", r"$w(\theta)$", ("lens_1", "lens_1")),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default="data/desy3/inputs/des_y3_3x2pt_maglim.sacc")
    parser.add_argument("--blinded", default="data/desy3/outputs_smokescreen/twopoint_data_real.sacc")
    parser.add_argument("--output", default="data/desy3/outputs_smokescreen/blinding_shift.png")
    args = parser.parse_args()

    raw = sacc.Sacc.load_fits(args.raw)
    blinded = sacc.Sacc.load_fits(args.blinded)
    delta = blinded.mean - raw.mean
    sigma = np.sqrt(np.diag(raw.covariance.dense))

    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex="col")
    for j, (data_type, label, pair) in enumerate(PANELS):
        top, bottom = axes[0, j], axes[1, j]

        idx = raw.indices(data_type, pair)
        theta = np.array(raw.get_tag("theta", data_type, pair))
        top.errorbar(theta, theta * raw.mean[idx], theta * sigma[idx], fmt="o", ms=4, color="0.4", label="raw")
        top.plot(theta, theta * blinded.mean[idx], "s", ms=4, mfc="none", color="C3", label="blinded")
        top.set_title(f"{label}  {pair[0]}, {pair[1]}", fontsize=10)
        top.set_ylabel(r"$\theta\,$" + label)

        for t1, t2 in raw.get_tracer_combinations(data_type):
            idx = raw.indices(data_type, (t1, t2))
            theta = np.array(raw.get_tag("theta", data_type, (t1, t2)))
            bottom.plot(theta, delta[idx] / sigma[idx], color="C0", alpha=0.6, lw=1)
        bottom.axhline(0, color="k", lw=0.5)
        bottom.set_xscale("log")
        bottom.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        bottom.set_xlabel(r"$\theta$ [arcmin]")
    axes[0, 0].legend(frameon=False)
    axes[1, 0].set_ylabel(r"$\Delta d / \sigma$")
    fig.suptitle("Smokescreen blinding shift on the DES Y3 MagLim 3x2pt data vector")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
