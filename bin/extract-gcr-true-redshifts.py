# Modify these to use different data sets:
cat_name='cosmoDC2_v1.0'
shear_name='/global/projecta/projectdirs/lsst/groups/WL/users/zuntz/data/inputs-cosmoDC2/shear_catalog.fits'
z_name = 'Z.npy'

import GCRCatalogs
import fitsio
import numpy as np

# Truth data
gc = GCRCatalogs.load_catalog(cat_name)
cols = ['galaxy_id', 'redshift']

# Shear catalog, needed for list of used objects
f = fitsio.FITS(shear_name)
ext = f[1]

# Loops through GCR producing chunk of data at a time
it = gc.get_quantities(cols, return_iterator=True)

# Lookup table ID->z
lookup = {}


# Add another chunk of true redshifts from GCR
def add_to_lookup():
    global lookup
    # Clear the current table
    lookup = {}
    # Get the next chunk
    truth = next(it)
    id_true = truth['galaxy_id']
    z_true = truth['redshift']
    ntrue = z_true.size

    # Fill in the dict with this chunk of data
    print(f"Added chunk of size {ntrue}")
    for id_t, z_t in zip(id_true, z_true):
        lookup[id_t] = z_t


# First chunk of data
add_to_lookup()


# Now we go through the FITS file "n" objects at a time,
# finding the z values for each galaxy we used.
pos = 0
n = 100000

max_size = ext.get_nrows()
Z = np.zeros(max_size)
cur=0
print(f"max size = {max_size}")

while (pos < max_size):
    # load a chunk of galaxy IDs
    print(f"Loading from {pos}")
    ids = ext['id'][pos:pos+n]
    pos += n

    # Lookup the z for each one.  If we have reached the end of our
    # lookup table then we get a new one
    for ID in ids:
        z = lookup.pop(ID, None)
        x = 0
        # Keep going till we find our data.  This works because the IDs are ordered
        # monotonically in both catalogs.
        while z is None:
            add_to_lookup()
            z = lookup.pop(ID, None)
            x += 1
        Z[cur] = z
        cur += 1
        
np.save(z_name, Z)
