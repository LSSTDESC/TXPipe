"""
Build a source n(z) for the DP1 metadetect catalog from the SITCOMTN-154 gold
photo-z table.

The gold object catalog and the photo-z table are row-for-row aligned (same
objectId order), so no join is needed between them. Neither one shares an ID
with the metadetect catalog, though, since metadetect runs its own detection,
so we match on the sky instead.

Each object's p(z) is reconstructed from the five quantiles the table provides
(2.5 / 16 / 50 / 84 / 97.5) and the reconstructed PDFs are stacked with the
shear weights to give the n(z). Objects with no gold match are dropped, which
biases the result: they are ~0.5 mag fainter in i, so the stack is missing part
of the high-z tail.

Writes a QPNOfZFile with two rows: bin 0 and the 2D bin.
"""

import h5py
import numpy as np
import qp
from scipy.spatial import cKDTree

GOLD = "data/txpipe_dp1_all/download/80_dp1_gold_all_sitcomtn154_509bdd8359576489cedf762800f8570f/dp1_v29.0.0_gold_all.hdf5"
PZ = "data/txpipe_dp1_all/download/128_pz_table_dp1_all_gold_baseline_sitcomtn154_99d6722de1645ee6d92a49593835ece2/dp1_v29.0.0_gold_photoz_cat.parquet"
SHEAR = "data/txpipe_dp1_all/download/ingested_shear_DP1_precut.hdf5"
TOMO = "data/example/outputs_metadetect_dp1_sel/shear_tomography_catalog.hdf5"
OUT = "data/txpipe_dp1_all/download/shear_photoz_stack_fzboost.hdf5"

CODE = "fzboost"
MATCH_RADIUS = 0.5  # arcsec
QUANTS = np.array([0.025, 0.16, 0.5, 0.84, 0.975])
ZGRID = np.linspace(0.0, 3.0, 301)


def unit_vectors(ra, dec):
    r = np.radians(ra)
    d = np.radians(dec)
    return np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])


def match_to_gold():
    """Nearest-neighbour match of the metadetect 'ns' rows onto the gold catalog."""
    with h5py.File(GOLD, "r") as f:
        tree = cKDTree(unit_vectors(f["coord_ra"][:], f["coord_dec"][:]))

    with h5py.File(SHEAR, "r") as f:
        ra = f["shear/ns/ra"][:]
        dec = f["shear/ns/dec"][:]

    chord, index = tree.query(unit_vectors(ra, dec), k=1)
    # chord length on the unit sphere -> angular separation
    sep = np.degrees(2 * np.arcsin(chord / 2)) * 3600.0
    matched = sep < MATCH_RADIUS
    print(f"matched {matched.sum():,} / {matched.size:,} detections "
          f"({100 * matched.mean():.1f}%) within {MATCH_RADIUS}\"")
    return index, matched


def read_quantiles(index):
    import pyarrow.parquet as pq

    cols = [f"{CODE}_z_err95_low", f"{CODE}_z_err68_low", f"{CODE}_z_median",
            f"{CODE}_z_err68_high", f"{CODE}_z_err95_high"]
    table = pq.read_table(PZ, columns=cols)
    locs = np.column_stack([table[c].to_numpy() for c in cols])
    return locs[index]


def main():
    index, matched = match_to_gold()

    with h5py.File(TOMO, "r") as f:
        zbin = f["tomography/bin"][:]
        nbin = f["tomography"].attrs["nbin"]

    with h5py.File(SHEAR, "r") as f:
        weight = f["shear/ns/weight"][:]

    # The tomography catalog is index-aligned with the 'ns' group.
    assert zbin.size == matched.size, "tomography catalog is not aligned with shear/ns"

    locs = read_quantiles(index)

    # nbin rows for the tomographic bins, plus a final row for the 2D bin.
    pdfs = np.zeros((nbin + 1, ZGRID.size))

    for i in range(nbin):
        use = (zbin == i) & matched
        n_dropped = ((zbin == i) & ~matched).sum()
        print(f"bin {i}: stacking {use.sum():,} objects, dropping {n_dropped:,} unmatched")

        # Reconstruct each p(z) from its quantiles, then evaluate on our grid.
        ensemble = qp.Ensemble(qp.quant, data={"quants": QUANTS, "locs": locs[use]})
        pz = ensemble.pdf(ZGRID)

        w = weight[use]
        stacked = w @ pz
        pdfs[i] = stacked / np.trapezoid(stacked, ZGRID)
        pdfs[-1] += stacked

    pdfs[-1] /= np.trapezoid(pdfs[-1], ZGRID)

    stack = qp.Ensemble(qp.interp, data={"xvals": ZGRID, "yvals": pdfs})
    with h5py.File(OUT, "w") as f:
        group = f.create_group("qp")
        for name, subtables in stack.build_tables().items():
            subgroup = group.create_group(name)
            for key, val in subtables.items():
                subgroup.create_dataset(key, dtype=val.dtype, data=val.data)

    mean_z = np.trapezoid(ZGRID * pdfs[-1], ZGRID)
    print(f"wrote {OUT} ({nbin + 1} rows), 2D mean z = {mean_z:.3f}")


main()
