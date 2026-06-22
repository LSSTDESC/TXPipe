"""
Convert Roman HLIS parquet shear catalogs to TXPipe HDF5 format.

Produces:
  - roman_hlis_shear_catalog.hdf5   (metadetect-style shear catalog)
  - roman_hlis_photometry_catalog.hdf5

Column conventions
------------------
Magnitudes:  mag = 22.5 - 2.5 * log10(flux)   [flux from flux_pgauss_*]
Mag errors:  mag_err = 2.5 / ln(10) * flux_err / flux
T (size):    T = reff
All missing fields (flags, weight, psf_*, true_g*, extendedness) are set to
dummy values (0 or 1 as appropriate).
"""

import numpy as np
import pandas as pd
import h5py

# ---------------------------------------------------------------------------
# Input / output paths
# ---------------------------------------------------------------------------
INPUT_DIR = "."
OUTPUT_SHEAR = "roman_hlis_shear_catalog.hdf5"
OUTPUT_PHOT = "roman_hlis_photometry_catalog.hdf5"

PARQUET_FILES = {
    "00": "example_cat_scm_noshear.parquet",
    "1m": "example_cat_scm_1m.parquet",
    "1p": "example_cat_scm_1p.parquet",
    "2m": "example_cat_scm_2m.parquet",
    "2p": "example_cat_scm_2p.parquet",
}

# LSST bands available in the parquet files
LSST_BANDS = ["u", "g", "r", "i", "z", "y"]

# Roman bands available in the parquet files
ROMAN_BANDS = ["Y", "J", "H"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
LOG10 = np.log(10)


def flux_to_mag(flux):
    """22.5-based AB magnitude; non-positive flux → NaN."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mag = np.where(flux > 0, 22.5 - 2.5 * np.log10(flux), np.nan)
    return mag


def flux_to_mag_err(flux, flux_err):
    """Magnitude error from flux and flux error."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mag_err = np.where(flux > 0, 2.5 / LOG10 * np.abs(flux_err / flux), np.nan)
    return mag_err


def flux_to_snr(flux, flux_err):
    """Per-band S/N."""
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = np.where(flux_err > 0, flux / flux_err, np.nan)
    return snr


# ---------------------------------------------------------------------------
# Shear catalog
# ---------------------------------------------------------------------------
def make_shear_catalog():
    print("Building shear catalog …")
    with h5py.File(OUTPUT_SHEAR, "w") as f:
        shear_grp = f.create_group("shear")

        for variant, filename in PARQUET_FILES.items():
            path = f"{INPUT_DIR}/{filename}"
            print(f"  reading {filename} → shear/{variant}")
            df = pd.read_parquet(path)
            n = len(df)

            grp = shear_grp.create_group(variant)

            # --- position / identity ---
            grp.create_dataset("id",            data=df["objectid"].values.astype(np.int64))
            grp.create_dataset("ra",            data=df["ra"].values)
            grp.create_dataset("dec",           data=df["dec"].values)

            # --- redshift ---
            grp.create_dataset("redshift_true", data=df["z"].values)

            # --- shear ---
            grp.create_dataset("g1",            data=df["g1"].values)
            grp.create_dataset("g2",            data=df["g2"].values)

            # --- size ---
            grp.create_dataset("T",             data=df["reff"].values)
            grp.create_dataset("T_err",         data=np.zeros(n))   # dummy

            # --- S/N ---
            grp.create_dataset("s2n",           data=df["snr"].values)
            grp.create_dataset("pgauss_s2n",    data=df["pgauss_s2n"].values)

            # --- magnitudes (r, i, z) ---
            for band in ["r", "i", "z"]:
                flux     = df[f"flux_pgauss_LSST_{band}"].values
                flux_err = df[f"flux_err_pgauss_LSST_{band}"].values
                grp.create_dataset(f"mag_{band}",     data=flux_to_mag(flux))
                grp.create_dataset(f"mag_err_{band}", data=flux_to_mag_err(flux, flux_err))

            # --- PSF (dummy) ---
            grp.create_dataset("psf_T_mean",    data=np.zeros(n))
            grp.create_dataset("psf_g1",        data=np.zeros(n))
            grp.create_dataset("psf_g2",        data=np.zeros(n))

            # --- true shear (dummy) ---
            grp.create_dataset("true_g1",       data=np.zeros(n))
            grp.create_dataset("true_g2",       data=np.zeros(n))

            # --- flags / weight ---
            grp.create_dataset("flags",         data=np.zeros(n, dtype=np.int64))
            grp.create_dataset("weight",        data=np.ones(n))

    print(f"  → wrote {OUTPUT_SHEAR}\n")


# ---------------------------------------------------------------------------
# Photometry catalog  (uses the noshear / "00" parquet)
# ---------------------------------------------------------------------------
def make_photometry_catalog():
    print("Building photometry catalog …")
    filename = PARQUET_FILES["00"]
    path = f"{INPUT_DIR}/{filename}"
    print(f"  reading {filename}")
    df = pd.read_parquet(path)
    n = len(df)

    with h5py.File(OUTPUT_PHOT, "w") as f:
        grp = f.create_group("photometry")

        # --- position / identity ---
        grp.create_dataset("id",            data=df["objectid"].values.astype(np.int64))
        grp.create_dataset("ra",            data=df["ra"].values)
        grp.create_dataset("dec",           data=df["dec"].values)

        # --- redshift ---
        grp.create_dataset("redshift_true", data=df["z"].values)

        # --- true shear & size ---
        grp.create_dataset("shear_1",       data=df["g1"].values)
        grp.create_dataset("shear_2",       data=df["g2"].values)
        grp.create_dataset("size_true",     data=df["reff"].values)

        # --- LSST-band magnitudes, errors, and S/N ---
        for band in LSST_BANDS:
            flux     = df[f"flux_pgauss_LSST_{band}"].values
            flux_err = df[f"flux_err_pgauss_LSST_{band}"].values
            grp.create_dataset(f"mag_{band}",     data=flux_to_mag(flux))
            grp.create_dataset(f"mag_err_{band}", data=flux_to_mag_err(flux, flux_err))
            grp.create_dataset(f"snr_{band}",     data=flux_to_snr(flux, flux_err))

        # --- Roman-band magnitudes, errors, and S/N ---
        for band in ROMAN_BANDS:
            flux     = df[f"flux_pgauss_{band}"].values
            flux_err = df[f"flux_err_pgauss_{band}"].values
            grp.create_dataset(f"mag_{band}",     data=flux_to_mag(flux))
            grp.create_dataset(f"mag_err_{band}", data=flux_to_mag_err(flux, flux_err))
            grp.create_dataset(f"snr_{band}",     data=flux_to_snr(flux, flux_err))

        # --- extendedness (all galaxies → 1) ---
        grp.create_dataset("extendedness",  data=np.ones(n))

    print(f"  → wrote {OUTPUT_PHOT}\n")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    make_shear_catalog()
    make_photometry_catalog()
    print("Done.")
