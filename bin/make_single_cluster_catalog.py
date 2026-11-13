import numpy as np 
import h5py
ra = np.array([0.0])
dec = np.array([0.0])
z = np.array([0.22])
z_err = np.array([0.01])
ids = np.array([0])
richness = np.array([10.0])
richness_err = np.array([1.0])
scale = np.array([1.0])

with h5py.File("mock_single_cluster_catalog.hdf5", "w") as f:
    g = f.create_group("clusters")
    g
    g.create_dataset("cluster_id", data=ids)
    g.create_dataset("ra", data=ra)
    g.create_dataset("dec", data=dec)
    g.create_dataset("redshift", data=z)
    g.create_dataset("redshift_err", data=z_err)
    g.create_dataset("richness", data=richness)
    g.create_dataset("richness_err", data=richness_err)
    g.create_dataset("scaleval", data=scale)

    g.attrs["n_clusters"] = 1
