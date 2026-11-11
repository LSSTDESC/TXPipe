"""
Download the public DES Y3 MagLim 3x2pt data vector and convert it to a
TXPipe-style sacc file, ready to be blinded by TXSmokescreenBlinding.

    python examples/desy3_smokescreen/prepare_data.py [--fits FILE] [--output FILE]

The data is the official DES Y3 two-point file in the CosmoSIS "twopoint"
FITS format, as distributed with the CosmoSIS standard library. We:

 - convert the n(z) to sacc NZ tracers named source_0..3 and lens_0..3,
   following TXPipe conventions;
 - keep only the first four MagLim lens bins and the w(theta) auto-correlations,
   as in the DES Y3 3x2pt analysis;
 - apply the DES Y3 MagLim 3x2pt angular scale cuts.
"""
import argparse
import os
import urllib.request

import numpy as np
import sacc
from astropy.io import fits

URL = (
    "https://raw.githubusercontent.com/joezuntz/cosmosis-standard-library/main/"
    "likelihood/des-y3/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits"
)
DEFAULT_FITS = "data/desy3/inputs/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits"
DEFAULT_OUTPUT = "data/desy3/inputs/des_y3_3x2pt_maglim.sacc"

N_SOURCE = 4
N_LENS = 4

# Minimum angular scales in arcmin, from the CosmoSIS standard library file
# examples/des-y3-maglim-scale-cuts.ini. Keys use the FITS (1-based) bin numbers;
# for gammat the first bin is the lens. There are no maximum scale cuts.
SCALE_CUTS = {
    "xip": {
        (1, 1): 2.475, (1, 2): 6.21691892, (1, 3): 6.21691892, (1, 4): 4.93827423,
        (2, 2): 6.21691892, (2, 3): 6.21691892, (2, 4): 6.21691892,
        (3, 3): 6.21691892, (3, 4): 6.21691892, (4, 4): 4.93827423,
    },
    "xim": {
        (1, 1): 24.75, (1, 2): 62.16918918, (1, 3): 62.16918918, (1, 4): 49.3827423,
        (2, 2): 62.16918918, (2, 3): 78.26637209, (2, 4): 78.26637209,
        (3, 3): 78.26637209, (3, 4): 78.26637209, (4, 4): 62.16918918,
    },
    "gammat": {
        (1, s): 25.40688358 for s in range(1, 5)
    } | {
        (2, s): 18.26006735 for s in range(1, 5)
    } | {
        (3, s): 13.05578642 for s in range(1, 5)
    } | {
        (4, s): 10.86591189 for s in range(1, 5)
    },
    "wtheta": {(1, 1): 33.8758448, (2, 2): 24.3467565, (3, 3): 17.40771522, (4, 4): 14.48788252},
}

# FITS extension -> sacc data type, and a function giving the TXPipe tracer
# pair for FITS bins (b1, b2). TXPipe stores gammat as (source, lens).
DATA_TYPES = {
    "xip": (sacc.standard_types.galaxy_shear_xi_plus, lambda b1, b2: (f"source_{b1-1}", f"source_{b2-1}")),
    "xim": (sacc.standard_types.galaxy_shear_xi_minus, lambda b1, b2: (f"source_{b1-1}", f"source_{b2-1}")),
    "gammat": (sacc.standard_types.galaxy_shearDensity_xi_t, lambda b1, b2: (f"source_{b2-1}", f"lens_{b1-1}")),
    "wtheta": (sacc.standard_types.galaxy_density_xi, lambda b1, b2: (f"lens_{b1-1}", f"lens_{b2-1}")),
}


def download(fits_file):
    if os.path.exists(fits_file):
        print(f"Using existing {fits_file}")
        return
    os.makedirs(os.path.dirname(fits_file) or ".", exist_ok=True)
    print(f"Downloading {URL} to {fits_file}")
    urllib.request.urlretrieve(URL, fits_file)


def add_tracers(S, hdus):
    for name, ext, n in [("source", "nz_source", N_SOURCE), ("lens", "nz_lens", N_LENS)]:
        nz = hdus[ext].data
        quantity = "galaxy_shear" if name == "source" else "galaxy_density"
        for i in range(n):
            S.add_tracer("NZ", f"{name}_{i}", nz["Z_MID"], nz[f"BIN{i+1}"], quantity=quantity)


def convert(fits_file):
    """
    Build a sacc file with the data points that pass the selection and
    scale cuts, and the matching block of the covariance.
    """
    hdus = fits.open(fits_file)
    S = sacc.Sacc()
    add_tracers(S, hdus)

    # The covariance blocks follow the order of the extensions in its header
    cov_header = hdus["COVMAT"].header
    n_blocks = len([k for k in cov_header if k.startswith("NAME_")])
    order = [cov_header[f"NAME_{i}"] for i in range(n_blocks)]

    kept = []
    index = 0
    for ext in order:
        data_type, tracers = DATA_TYPES[ext]
        for row in hdus[ext].data:
            b1, b2 = int(row["BIN1"]), int(row["BIN2"])
            theta = float(row["ANG"])
            # Points with no scale cut entry are outside the analysis:
            # lens bins 5 and 6, and w(theta) cross-correlations.
            min_theta = SCALE_CUTS[ext].get((b1, b2))
            if min_theta is not None and theta >= min_theta:
                S.add_data_point(
                    data_type,
                    tracers(b1, b2),
                    float(row["VALUE"]),
                    theta=theta,
                    theta_min=float(row["ANGLEMIN"]),
                    theta_max=float(row["ANGLEMAX"]),
                )
                kept.append(index)
            index += 1

    cov = np.array(hdus["COVMAT"].data)
    S.add_covariance(cov[np.ix_(kept, kept)])
    return S


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fits", default=DEFAULT_FITS, help="DES Y3 twopoint FITS file (downloaded if missing)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output sacc file")
    args = parser.parse_args()

    download(args.fits)
    S = convert(args.fits)
    S.metadata["source"] = "DES Y3 MagLim 3x2pt, " + os.path.basename(args.fits)
    S.metadata["scale_cuts"] = "DES Y3 MagLim 3x2pt (CosmoSIS standard library)"
    S.save_fits(args.output, overwrite=True)

    print(f"Saved {len(S)} data points to {args.output}")
    for data_type in S.get_data_types():
        print(f"  {data_type}: {len(S.indices(data_type))}")


if __name__ == "__main__":
    main()
