import h5py

input_file = "data/example/inputs/metadetect_shear_catalog.hdf5"
output_file = "data/example/inputs/metadetect_shear_catalog_different_lengths.hdf5"


skips = {
    "00": 1,
    "1p": 100,
    "1m": 120,
    "2p": 130,
    "2m": 140,
}

with h5py.File(input_file, "r") as f:
    with h5py.File(output_file, "w") as f_out:
        g = f["shear"]
        g_out = f_out.create_group("shear")
        f_out['shear']
        for attr in f['shear'].attrs:
            f_out['shear'].attrs[attr] = f['shear'].attrs[attr]
        # loop through 1p, 1m, 2p, 2m
        for subgroup in g.keys():
            skip = skips[subgroup]

            subg = g[subgroup]
            subf = f_out['shear'].create_group(subgroup)

            for key in subg.keys():
                # skip final entries to make the lengths all different
                data = subg[key][:-skip]
                subf.create_dataset(key, data=data)
