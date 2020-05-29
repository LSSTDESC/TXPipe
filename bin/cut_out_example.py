import h5py
import numpy as np
#generate_subset.py

ra_min = 60
ra_max = 61
dec_min = -31
dec_max = -30

photo_in = h5py.File('photometry_catalog.hdf5', 'r')
photo_out = h5py.File('example_photometry_catalog.hdf5', 'w')

thin = 5


group_in = photo_in['photometry']
group_out = photo_out.create_group('photometry')

photo_in.copy('provenance', photo_out)

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


shear_in = h5py.File('shear_catalog.hdf5', 'r')
shear_out = h5py.File('example_shear_catalog.hdf5', 'w')

shear_in.copy('provenance', shear_out)

group_in = shear_in['metacal']
group_out = shear_out.create_group('metacal')

for col in group_in.keys():
    print("Copying", col)
    d = group_in[col][w]
    group_out.create_dataset(col, data=d)
