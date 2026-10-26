"""
Convert the concatenated anacal2 Gaia star parquet dump into the star_catalog
layout TXStarCatalogSplitter expects (a "stars" group with ra, dec, mag_r).

The Gaia table has no r-band magnitude, so the broad Gaia G band stands in for
it. G is not r: for these stars it runs a few tenths of a magnitude brighter,
so the 14 / 18.2 / 22 split points that TXStarCatalogSplitter uses to divide
bright from dim stars are approximate.

The dump is concatenated from per-cell files, and cells overlap, so the same
star appears up to 8 times. Duplicates are dropped using gaia_source_id.
"""

import h5py
import numpy as np

IN = "data/txpipe_dp1_all/download/anacal2_gaia_stars/combined.hdf5"
OUT = "data/txpipe_dp1_all/download/anacal2_gaia_stars/star_catalog.hdf5"


def main():
    with h5py.File(IN, "r") as f:
        names = [x.decode() for x in f["df/block0_items"][:]]
        values = f["df/block0_values"][:]
        # pandas stores the integer columns in a separate block
        assert f["df/block1_items"][0].decode() == "gaia_source_id"
        source_id = f["df/block1_values"][:, 0]

    cols = {name: values[:, i] for i, name in enumerate(names)}

    _, first = np.unique(source_id, return_index=True)
    print(f"keeping {first.size:,} unique stars from {source_id.size:,} rows")
    first.sort()
    source_id = source_id[first]
    ra = cols["ra"][first]
    dec = cols["dec"][first]
    mag_r = cols["gaia_g_mag"][first]

    # A handful of Gaia rows have no G magnitude and cannot be binned.
    good = np.isfinite(mag_r)
    print(f"dropping {(~good).sum()} stars with no Gaia G magnitude")
    source_id, ra, dec, mag_r = source_id[good], ra[good], dec[good], mag_r[good]

    with h5py.File(OUT, "w") as f:
        group = f.create_group("stars")
        group.create_dataset("id", data=source_id)
        group.create_dataset("ra", data=ra)
        group.create_dataset("dec", data=dec)
        group.create_dataset("mag_r", data=mag_r)

    bright = ((mag_r > 14.0) & (mag_r < 18.2)).sum()
    dim = ((mag_r > 18.2) & (mag_r < 22.0)).sum()
    print(f"wrote {OUT}: {ra.size:,} stars ({bright:,} bright, {dim:,} dim)")
    print(f"  ra {ra.min():.2f}..{ra.max():.2f}  dec {dec.min():.2f}..{dec.max():.2f}")


main()
