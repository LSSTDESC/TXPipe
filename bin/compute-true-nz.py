"""
Hack a 2pt file to replace the n(z) with truth values.
The true z values must be in the photometry file.
"""
import h5py
import numpy as np
import sacc
import sys
import os

photo_file = sys.argv[1]
tomo_file = sys.argv[2]
sacc_file = sys.argv[3]

output_file = os.path.splitext(sacc_file)[0] + '.true_nz.sacc'

S = sacc.Sacc.load_fits(sacc_file)

example_tracer = S.get_tracer_combinations()[0][0]

zmin = 0
zmax = 4.0
nz = 400
dz = zmax/nz
z = np.arange(zmin, zmax, dz)


photo = h5py.File(photo_file, "r")
tomo = h5py.File(tomo_file, "r")


max_size = photo['photometry/redshift_true'].size
nbin_source = tomo['tomography'].attrs['nbin_source']
nbin_lens = tomo['tomography'].attrs['nbin_lens']

nz_source = [np.zeros(nz) for i in range(nbin_source)]
nz_lens = [np.zeros(nz) for i in range(nbin_lens)]

assert len(z) == len(nz_lens[0])

chunk_size = 1_000_000
s = 0
while True:
    e = s + chunk_size
    print(s,e)
    zt = photo['photometry/redshift_true'][s:e]
    source_bin = tomo['tomography/source_bin'][s:e]
    lens_bin = tomo['tomography/source_bin'][s:e]

    for i in range(nbin_source):
        w = np.where(source_bin==i)
        count, _ = np.histogram(zt[w], bins=nz, range=(zmin, zmax))
        nz_source[i] += count
    for i in range(nbin_lens):
        w = np.where(lens_bin==i)
        count, _ = np.histogram(zt[w], bins=nz, range=(zmin, zmax))
        nz_lens[i] += count
    s = e
    if e > max_size:
        break

for i in range(nbin_source):
    t = S.tracers[f'source_{i}']
    t.z = z
    t.nz = nz_source[i]

for i in range(nbin_lens):
    t = S.tracers[f'lens_{i}']
    t.z = z
    t.nz = nz_lens[i]

S.save_fits(output_file)
