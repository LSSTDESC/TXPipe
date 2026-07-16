import h5py
import numpy as np

output_dir = '/pscratch/sd/c/chihway/TXPipe/data/example/outputs_roman_hlis/e2e/'

src = output_dir+"binned_random_catalog_new.hdf5"
dst = output_dir+"binned_random_catalog_sub.hdf5"
sample_rate = 0.5

np.random.seed(42)

with h5py.File(src, "r") as f_in, h5py.File(dst, "w") as f_out:
    randoms_group = f_out.create_group("randoms")
    bins = sorted(f_in["randoms"].keys())
    randoms_group.attrs["nbin"] = len(bins)

    for bin_name in bins:
        ra = f_in[f"randoms/{bin_name}/ra"][:]
        dec = f_in[f"randoms/{bin_name}/dec"][:]
        ntotal = len(ra)
        nsub = int(sample_rate * ntotal)
        idx = np.random.choice(ntotal, size=nsub, replace=False)

        grp = randoms_group.create_group(bin_name)
        grp.create_dataset("ra", data=ra[idx])
        grp.create_dataset("dec", data=dec[idx])
        print(f"{bin_name}: {ntotal} -> {nsub}")

print("Done.")
