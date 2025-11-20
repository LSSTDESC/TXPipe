#Do a matching of the metadetect catalogs
import parallel_statstics
import h5py
import os

params = [
    "gauss_g_1",
    "gauss_g_2",
    "gauss_s2n",
    "pgauss_band_flux_g_nodered",
    "pgauss_band_flux_r_nodered",
    "pgauss_band_flux_i_nodered",
    "pgauss_band_flux_z_nodered",
    "pgauss_T",
]

nparam = len(params)
variants = [
    "noshear",
    "1p",
    "1m",
    "2p",
    "2m",
]

stats = {
    v: parallel_statstics.ParallelMean(nparam) for v in variants
}

results = {}
delta_gamma = 0.02
chunk_size = 1_000_000

cfs_dir = os.environ["CFS"]
filename = os.path.join(cfs_dir, "des", "y6kp-cats", "final_version", "metadetect_notomo2024-11-07.hdf5")

with h5py.File(filename) as f:
    for v in variants:
        stat = stats[v]
        s = 0
        e = 0
        ntot = f[f"metadetect/{v}/ra"].shape[0]
        while e < ntot:
            s = e
            e = min(s + chunk_size, ntot)
            print(f"Processing {v} {s}:{e} / {ntot}")
            data = {p: f[f"{v}/{p}"][s:e] for p in params}
            for i, p in enumerate(params):
                stat.add_data(i, data[p])
        _, mu = stat.collect()
        print(f"Variant {v}:")
        for i, p in enumerate(params):
            print(f"  {p}: {mu[i]}")
        results[v] = mu

print("")
for i, p in enumerate(params):
    dp_dg1 = (results["1p"][i] - results["1m"][i]) / delta_gamma
    dp_dg2 = (results["2p"][i] - results["2m"][i]) / delta_gamma
    print(f"d({p}) / dg1: {dp_dg1}")
    print(f"d({p}) / dg2: {dp_dg2}")
