import h5py
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='Cut out a subset')

ra_min = 60
ra_max = 61
dec_min = -31
dec_max = -30


def trim_file(input_file, output_file, group_name, thin):
    if isinstance(group_name, str):
        group_name = [group_name]
    file_in = h5py.File(input_file, 'r')
    file_out = h5py.File(output_file, 'w')


    file_in.copy('provenance', file_out)

    for group_name in group_name:
        group_in = file_in[group_name]
        group_out = file_out.create_group(group_name)

        ra = group_in['ra'][:]
        print("read group", group_name, "size:", ra.size)
        dec = group_in['dec'][:]
        w  = ra > ra_min
        w &= ra < ra_max
        w &= dec > dec_min
        w &= dec < dec_max
        del ra, dec

        if thin > 1:
            w[np.arange(w.size) % thin > 0] = False

        print('Output count:', w.sum())

        for col in group_in.keys():
            print("Copying", col)
            d = group_in[col][w]
            group_out.create_dataset(col, data=d)


if __name__ == '__main__':
    trim_file('photometry_catalog.hdf5', 'example_photometry_catalog.hdf5', 'photometry', 1)

    shear_groups = ["shear/00", "shear/1p", "shear/1m", "shear/2p", "shear/2m"]
    trim_file('shear_catalog.hdf5', 'example_shear_catalog.hdf5', shear_groups, 1)
    # trim_file('star_catalog.hdf5', 'example_star_catalog.hdf5', 'stars', 5)
