import h5py
import sys
filename = sys.argv[1]
with h5py.File(filename, "r+") as f:
    f["shear/ns"] = f["shear/00"]
