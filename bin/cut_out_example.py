import h5py
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='Cut out a subset')
parser.add_argument('--ra-min', type=float, default=60)
parser.add_argument('--ra-max', type=float, default=61)
parser.add_argument('--dec-min', type=float, default=-31)
parser.add_argument('--dec-max', type=float, default=-30)
parser.add_argument('--thin', type=int, default=5)


def copy_group(group_in, group_out, thin, bounds):
    if "ra" in group_in.keys():
        ra = group_in['ra'][:]
        dec = group_in['dec'][:]
        ra_min, ra_max, dec_min, dec_max = bounds
        mask = (ra > ra_min) & (ra < ra_max)
        mask &= (dec > dec_min) & (dec < dec_max)
        if thin != 1:
            mask[np.arange(mask.size) % thin > 0] = False

    for key, value in group_in.attrs.items():
        group_out.attrs[key] = value

    for name, item in group_in.items():
        if isinstance(item, h5py.Group):
            subgroup_out = group_out.create_group(name)
            copy_group(item, subgroup_out, mask, thin, bounds)
        else:
            if mask is None:
                raise ValueError("No ra/dec for", name)
            data = item[:]
            data = data[mask]
            print("Copying", name)
            dataset_out = group_out.create_dataset(name, data=data)
            for key, value in item.attrs.items():
                dataset_out.attrs[key] = value


def trim_file(input_file, output_file, group_name, thin,
              ra_min, ra_max, dec_min, dec_max):
    bounds = (ra_min, ra_max, dec_min, dec_max)
    with h5py.File(input_file, 'r') as file_in:
        with h5py.File(output_file, 'w') as file_out:
            group_in = file_in[group_name]
            group_out = file_out.require_group(group_name)

            if 'provenance' in file_in:
                file_in.copy('provenance', file_out)
            copy_group(group_in, group_out, thin, bounds)


if __name__ == '__main__':
    args = parser.parse_args()
    trim_file('photometry_catalog.hdf5', 'example_photometry_catalog.hdf5',
              'photometry', args.thin, args.ra_min, args.ra_max, args.dec_min, args.dec_max)
    trim_file('shear_catalog.hdf5', 'example_shear_catalog.hdf5',
              'shear', args.thin, args.ra_min, args.ra_max, args.dec_min, args.dec_max)
    trim_file('star_catalog.hdf5', 'example_star_catalog.hdf5',
              'stars', args.thin, args.ra_min, args.ra_max, args.dec_min, args.dec_max)
