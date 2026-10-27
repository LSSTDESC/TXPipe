"""
Compute TXPipe 3x2pt theory predictions on a synthetic sacc file, so that
results from different firecrown versions can be compared.

Run this once per environment / TXPipe checkout, e.g.:

    python compute_theory.py <txpipe_repo> <output_dir>

It writes one sacc file per (space, bias option) case into output_dir.
"""
import os
import sys
import numpy as np
import sacc

repo, outdir = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
from txpipe.utils.theory import theory_3x2pt
from txpipe.data_types import FiducialCosmology

import firecrown

print("Using firecrown", firecrown.__version__)


def make_tracers():
    s = sacc.Sacc()
    z = np.linspace(0.0, 3.0, 300)
    for i, zc in enumerate([0.4, 0.7, 1.0, 1.3]):
        s.add_tracer("NZ", f"source_{i}", z, np.exp(-0.5 * ((z - zc) / 0.15) ** 2))
    for i, zc in enumerate([0.3, 0.5, 0.7]):
        s.add_tracer("NZ", f"lens_{i}", z, np.exp(-0.5 * ((z - zc) / 0.05) ** 2))
    return s


cosmo = FiducialCosmology(f"{repo}/data/fiducial_cosmology.yml", "r").to_ccl()
ell = np.unique(np.geomspace(20, 2000, 30).astype(int))
theta = np.geomspace(2.5, 250, 20)  # arcmin

# All the forms of bias input that make_bias_parameters accepts
bias_options = {
    "none": None,
    "float": 1.5,
    "array": np.array([1.2, 1.4, 1.6]),
    "dict": {"lens_0": 1, "lens_1": 2, "lens_2": 2},
    "string": "1.1,1.3,1.5",
}

os.makedirs(outdir, exist_ok=True)

for space in ["fourier", "real"]:
    for name, bias in bias_options.items():
        kw = {"ell_values": ell} if space == "fourier" else {"theta_values": theta}
        theory = theory_3x2pt(cosmo, make_tracers(), bias=bias, **kw)
        theory.save_fits(f"{outdir}/{space}_{name}.sacc", overwrite=True)

# Unsupported data types (B-modes) should be skipped and set to zero
s = make_tracers()
for e in ell:
    s.add_data_point("galaxy_shear_cl_ee", ("source_0", "source_0"), 0.0, ell=e)
    s.add_data_point("galaxy_shear_cl_bb", ("source_0", "source_0"), 0.0, ell=e)
theory = theory_3x2pt(cosmo, s, bias=None)
theory.save_fits(f"{outdir}/bmode.sacc", overwrite=True)
