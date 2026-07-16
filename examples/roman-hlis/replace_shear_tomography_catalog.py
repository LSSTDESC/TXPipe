import h5py
import shutil
import numpy as np
import matplotlib.pyplot as plt

output_dir = '/pscratch/sd/c/chihway/TXPipe/data/example/outputs_roman_hlis/'

new_file = output_dir+"e2e/e2e_Jun292026_casphotoz.h5"
old_file = output_dir+"shear_tomography_catalog.hdf5"
output_file = output_dir+"e2e/shear_tomography_catalog_new.hdf5"

# Make a safe copy first
shutil.copy(old_file, output_file)

mapping = {
    "tomography/bin": "tomographic_bin_assignment_noshear",
    "tomography/bin_1m": "tomographic_bin_assignment_1m",
    "tomography/bin_1p": "tomographic_bin_assignment_1p",
    "tomography/bin_2m": "tomographic_bin_assignment_2m",
    "tomography/bin_2p": "tomographic_bin_assignment_2p",
}

# replace the tomography
with h5py.File(new_file, "r") as fb, h5py.File(output_file, "r+") as fout:

    for target_path, new_path in mapping.items():
        print(f"Replacing {target_path} with {new_path}")

        old = fout[target_path]
        new = fb[new_path][:].astype(old.dtype)

        if old.shape != new.shape:
            raise ValueError(
                f"Shape mismatch for {target_path}: "
                f"old shape {old.shape}, new shape {new.shape}"
            )

        old[...] = new


# rewrite the metadata

META_VARIANTS = ["00", "1p", "1m", "2p", "2m"]


def compute_and_write_metadetect_metadata(tomo_path, shear_path, delta_gamma=0.01):
    """
    Recompute all counts/metadata and response matrices for a metadetect
    tomography catalog using the bin assignments already stored in tomo_path.

    Quantities written:
        counts/counts, counts/counts_2d
        counts/N_eff, counts/N_eff_2d
        counts/mean_e1, counts/mean_e1_2d
        counts/mean_e2, counts/mean_e2_2d
        counts/sigma_e, counts/sigma_e_2d
        response/R, response/R_2d

    Parameters
    ----------
    tomo_path : str
        Path to shear_tomography_catalog.hdf5. Bin assignments must already be
        written (tomography/bin_00, tomography/bin_1p, etc.).
    shear_path : str
        Path to shear_catalog.hdf5. Expected layout: shear/{variant}/g1,
        shear/{variant}/g2, shear/{variant}/weight for each META_VARIANT.
    delta_gamma : float
        Step size used for the metadetect shear variants (default 0.01).
    """
    print(f"Reading bin assignments from {tomo_path}")
    with h5py.File(tomo_path, "r") as tf:
        nbin = int(tf["tomography"].attrs["nbin"])
        bins = {v: tf[f"tomography/bin_{v}"][:] for v in META_VARIANTS}

    print(f"Reading shear data from {shear_path}")
    with h5py.File(shear_path, "r") as sf:
        g1     = {v: sf[f"shear/{v}/g1"][:]     for v in META_VARIANTS}
        g2     = {v: sf[f"shear/{v}/g2"][:]     for v in META_VARIANTS}
        weight = {v: sf[f"shear/{v}/weight"][:] for v in META_VARIANTS}

    def weighted_mean(x, w):
        return np.dot(w, x) / np.sum(w)

    def weighted_mean_and_var(x, w):
        sw = np.sum(w)
        mean = np.dot(w, x) / sw
        var  = np.dot(w, (x - mean) ** 2) / sw
        return mean, var

    def compute_stats(sel_00, sel_1p, sel_1m, sel_2p, sel_2m):
        """Compute all statistics for one bin selection."""
        w = weight["00"][sel_00]
        counts = int(sel_00.sum())
        if counts == 0:
            return None

        N_eff = np.sum(w) ** 2 / np.sum(w ** 2)

        mean_g1, var_g1 = weighted_mean_and_var(g1["00"][sel_00], w)
        mean_g2, var_g2 = weighted_mean_and_var(g2["00"][sel_00], w)

        # Response matrix: R[i,j] = (mean_gi_jp - mean_gi_jm) / delta_gamma
        # where i indexes ellipticity component and j indexes shear direction
        R = np.zeros((2, 2))
        for col, component in enumerate([g1, g2]):
            w_1p = weight["1p"][sel_1p];  R[col, 0]  = weighted_mean(component["1p"][sel_1p], w_1p)
            w_1m = weight["1m"][sel_1m];  R[col, 0] -= weighted_mean(component["1m"][sel_1m], w_1m)
            w_2p = weight["2p"][sel_2p];  R[col, 1]  = weighted_mean(component["2p"][sel_2p], w_2p)
            w_2m = weight["2m"][sel_2m];  R[col, 1] -= weighted_mean(component["2m"][sel_2m], w_2m)
        R /= delta_gamma

        Rinv = np.linalg.inv(R)
        mean_e1, mean_e2 = Rinv @ np.array([mean_g1, mean_g2])

        # sigma_e: propagate raw ellipticity variance through R^{-1}
        # P = diag(inv(R @ R)), sigma_e = sqrt(0.5 * P . [var_g1, var_g2])
        P = np.diag(np.linalg.inv(R @ R))
        sigma_e = float(np.sqrt(0.5 * P @ np.array([var_g1, var_g2])))

        return dict(counts=counts, N_eff=N_eff,
                    mean_e1=float(mean_e1), mean_e2=float(mean_e2),
                    sigma_e=sigma_e, R=R)

    # Per-bin
    results = []
    for b in range(nbin):
        sels = {v: (bins[v] == b) for v in META_VARIANTS}
        stats = compute_stats(sels["00"], sels["1p"], sels["1m"], sels["2p"], sels["2m"])
        results.append(stats)
        if stats:
            print(f"  bin {b}: counts={stats['counts']:,}, N_eff={stats['N_eff']:.1f}, "
                  f"mean_e=({stats['mean_e1']:.4f}, {stats['mean_e2']:.4f}), "
                  f"sigma_e={stats['sigma_e']:.4f}")
        else:
            print(f"  bin {b}: empty")

    # 2D (all selected objects)
    sels_2d = {v: (bins[v] >= 0) for v in META_VARIANTS}
    stats_2d = compute_stats(sels_2d["00"], sels_2d["1p"], sels_2d["1m"],
                              sels_2d["2p"], sels_2d["2m"])
    if stats_2d:
        print(f"  2d:  counts={stats_2d['counts']:,}, N_eff={stats_2d['N_eff']:.1f}, "
              f"mean_e=({stats_2d['mean_e1']:.4f}, {stats_2d['mean_e2']:.4f}), "
              f"sigma_e={stats_2d['sigma_e']:.4f}")

    # Write everything back
    print(f"Writing metadata to {tomo_path}")
    with h5py.File(tomo_path, "r+") as tf:
        for b, stats in enumerate(results):
            if stats is None:
                continue
            tf["counts/counts"][b]  = stats["counts"]
            tf["counts/N_eff"][b]   = stats["N_eff"]
            tf["counts/mean_e1"][b] = stats["mean_e1"]
            tf["counts/mean_e2"][b] = stats["mean_e2"]
            tf["counts/sigma_e"][b] = stats["sigma_e"]
            tf["response/R"][b]     = stats["R"]

        if stats_2d:
            tf["counts/counts_2d"][:]  = stats_2d["counts"]
            tf["counts/N_eff_2d"][:]   = stats_2d["N_eff"]
            tf["counts/mean_e1_2d"][:] = stats_2d["mean_e1"]
            tf["counts/mean_e2_2d"][:] = stats_2d["mean_e2"]
            tf["counts/sigma_e_2d"][:] = stats_2d["sigma_e"]
            tf["response/R_2d"][:]     = stats_2d["R"]

    print("Metadata updated successfully.")


# ---------------------------------------------------------------------------
# CLI: structure-fix pass (handles mismatched nbin)
# ---------------------------------------------------------------------------


tomo_path  = output_dir+"e2e/shear_tomography_catalog_new.hdf5"
shear_path = None
delta_gamma = 0.01

with h5py.File(tomo_path, "r+") as f:
    bins = f["tomography/bin"][:]
    cat_type = f["tomography"].attrs["catalog_type"]

    nbin_actual = int(bins.max()) + 1 if (bins >= 0).any() else 0
    nbin_stored = int(f["counts/counts"].shape[0])

    print(f"catalog_type: {cat_type}")
    print(f"nbin stored in counts: {nbin_stored}, nbin from bin array: {nbin_actual}")

    scalar_datasets = ["counts", "sigma_e", "mean_e1", "mean_e2", "N_eff"]
    if nbin_actual != nbin_stored:
        print(f"Recreating counts datasets with nbin={nbin_actual}")
        for name in scalar_datasets:
            key = f"counts/{name}"
            old_data = f[key][:]
            del f[key]
            new_data = np.zeros(nbin_actual, dtype=old_data.dtype)
            if name == "counts":
                new_data[:] = np.array([int(np.sum(bins == b)) for b in range(nbin_actual)])
            else:
                n_copy = min(nbin_actual, len(old_data))
                new_data[:n_copy] = old_data[:n_copy]
            f["counts"].create_dataset(name, data=new_data)
            print(f"  counts/{name}: shape {old_data.shape} -> {new_data.shape}")
        f["tomography"].attrs["nbin"] = nbin_actual
        print(f"  tomography.nbin attr: {nbin_stored} -> {nbin_actual}")
    else:
        print("nbin matches; updating counts values only")
        counts_actual = np.array([int(np.sum(bins == b)) for b in range(nbin_actual)])
        for b in range(nbin_actual):
            print(f"  bin {b}: {int(f['counts/counts'][b])} -> {counts_actual[b]}")
            f["counts/counts"][b] = counts_actual[b]

    count2d = int((bins >= 0).sum())
    print(f"  counts_2d: {int(f['counts/counts_2d'][0])} -> {count2d}")
    f["counts/counts_2d"][:] = count2d

    if "response" in f and cat_type == "metadetect" and "response/R" in f:
        R_old = f["response/R"][:]
        if len(R_old) != nbin_actual:
            print(f"Recreating response/R: {R_old.shape} -> ({nbin_actual}, 2, 2)")
            del f["response/R"]
            R_new = np.zeros((nbin_actual, 2, 2), dtype=R_old.dtype)
            n_copy = min(nbin_actual, len(R_old))
            R_new[:n_copy] = R_old[:n_copy]
            for i in range(n_copy, nbin_actual):
                R_new[i] = np.eye(2)
            f["response"].create_dataset("R", data=R_new)

if shear_path is not None and cat_type == "metadetect":
    compute_and_write_metadetect_metadata(tomo_path, shear_path, delta_gamma)
elif shear_path is not None:
    print(f"Warning: shear catalog supplied but catalog_type is '{cat_type}', skipping metadata recomputation.")

shutil.copy(output_file, old_file)

print("Done.")