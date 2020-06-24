import h5py
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='Cut out a subset')

ra_min = 60
ra_max = 61
dec_min = -31
dec_max = -30


def trim_file(input_file, output_file, group_name, thin):
    file_in = h5py.File(input_file, 'r')
    file_out = h5py.File(output_file, 'w')

    group_in = file_in[group_name]
    group_out = file_out.create_group(group_name)

    file_in.copy('provenance', file_out)

    ra = group_in['ra'][:]
    print("read RA", ra.size)
    dec = group_in['dec'][:]
    print("read dec")
    w  = ra > ra_min
    w &= ra < ra_max
    w &= dec > dec_min
    w &= dec < dec_max
    del ra, dec
    w[np.arange(w.size) % thin > 0] = False

    print('Output count:', w.sum())

    for col in group_in.keys():
        print("Copying", col)
        d = group_in[col][w]
        group_out.create_dataset(col, data=d)


if __name__ == '__main__':
    trim_file('photometry_catalog.hdf5', 'example_photometry_catalog.hdf5', 'photometry', 5)
    trim_file('shear_catalog.hdf5', 'example_shear_catalog.hdf5', 'photometry', 5)
    trim_file('star_catalog.hdf5', 'example_star_catalog.hdf5', 'stars', 5)
