import h5py
import numpy as np


z_name = 'Z.npy'
tomo_name = "./outputs/tomography_catalog.hdf5"
nbin_source = 4
nbin_lens = 1

z = np.load(z_name)
f=h5py.File(tomo_name)

source = f['tomography/source_bin'][:]
lens = f['tomography/lens_bin'][:]

z_grid = np.arange(0., 5.0, 0.01)

for i in range(nbin_source):
    w = np.where(source==i)
    zw = z[w]

    n = len(zw)
    print(f"n = {n} for {i}")
    if n==0:
        continue
    print(i, zw.min(), zw.max(), zw.mean())
    n_of_z,edges = np.histogram(zw, bins=z_grid)
    output = np.array([edges[:-1], n_of_z]).T
    np.savetxt(f"source_{i}.txt", output)


for i in range(nbin_lens=1):
    w = np.where(lens==i)
    zw = z[w]

    n = len(zw)
    print(f"n = {n} for {i}")
    if n==0:
        continue
    print(i, zw.min(), zw.max(), zw.mean())
    n_of_z,edges = np.histogram(zw, bins=z_grid)
    output = np.array([edges[:-1], n_of_z]).T
    np.savetxt(f"lens_{i}.txt", output)
