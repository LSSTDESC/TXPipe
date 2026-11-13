import h5py
import sys
import argparse
import glob
import os

def printer(name, obj, prov=False):
    indent = "    " * name.count("/")
    bits = name.split("/")
    if (not prov) and (bits[0] == "provenance"):
        return
    name = bits[-1]
    if isinstance(obj, h5py.File):
        print(f"{indent}[{name}]")
    elif isinstance(obj, h5py.Group):
        print(f"{indent}[{name}]")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}- {name} {obj.dtype} {obj.shape}")

    if hasattr(obj, "attrs"):
        d = dict(obj.attrs)
        for k, v in d.items():
            print(f"{indent}    * {k} = {v}")

def printer_prov(name, obj):
    return printer(name, obj, prov=True)

def main(path, prov):
    if os.path.isdir(path):
        files = glob.glob(f"{path}/*.hdf") + glob.glob(f"{path}/*.hdf5")
    else:
        files = [path]
    for infile in files:
        print("-"*80)
        print(infile)
        print("")
        p = printer_prov if prov else printer
        with h5py.File(infile, "r") as f:
            f.visititems(p)
        print("")
        print("")






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Print contents of HDF5 file or directory of files")
    parser.add_argument("path", help="Name of file or directory")
    parser.add_argument("--prov", action="store_true", help="Include the provenance section")
    args = parser.parse_args()
    main(args.path, args.prov)