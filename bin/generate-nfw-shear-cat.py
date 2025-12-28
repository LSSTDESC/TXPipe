import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="simple_cat.txt")
parser.add_argument("--nmax", type=int, default=10000, help="Number of objects to generate, though cat size is reduced by redshift cut")
parser.add_argument("--mass", type=float, default=10.0e14, help="Cluster mass in Msun")
parser.add_argument("--cluster_z", type=float, default=0.22, help="Cluster redshift")
parser.add_argument("--concentration", type=float, default=4.0, help="Cluster concentration parameter")
parser.add_argument("--size", type=float, default=240.0, help="Size of the region in arcmin")
parser.add_argument("--mean-z", type=float, default=0.5, help="Mean redshift of the background galaxies")
parser.add_argument("--sigma-z", type=float, default=0.1, help="Redshift std dev of the background galaxies")
parser.add_argument("--mean-snr", type=float, default=20., help="Mean SNR")
parser.add_argument("--mean-size", type=float, default=0.3, help="Galaxy mean size^2 T parameter in arcsec")
parser.add_argument("--sigma-size", type=float, default=0.3, help="Galaxy std dev size^2 T parameter in arcsec")
args = parser.parse_args()

mass = args.mass
cluster_z = args.cluster_z
concentration = args.concentration


import galsim
import numpy as np
from astropy.table import Table
import argparse

nfw = galsim.NFWHalo(mass, concentration, cluster_z)

half_size = args.size / 2
xmin = -half_size
xmax = half_size
ymin = -half_size
ymax = half_size

N = 10000

x = np.random.uniform(xmin, xmax, N)
y = np.random.uniform(ymin, ymax, N)
z = np.random.normal(args.mean_z, args.sigma_z, N)

w = np.where(z > nfw.z)
x1 = x[w]
y1 = y[w]
z1 = z[w]
n = z1.size

ra = np.zeros(n) + x1 / 3600
dec = np.zeros(n) + y1 / 3600
g1, g2 = nfw.getShear([x1, y1], z1, reduced=False)
s2n = np.random.exponential(args.mean_snr, size=n)
print(s2n.mean())

# This should give plenty of selected objects since we cut on T/Tpsf > 0.5 by default
# and Tpsf default is ~.2
T = np.random.normal(args.mean_size, args.sigma_size, size=n).clip(0.01, np.inf)

data = {
    "ra": ra,
    "dec": dec,
    "g1": g1,
    "g2": g2,
    "s2n": s2n,
    "T": T,
    "redshift": z1,
}

table = Table(data=data)
table.write(args.output, overwrite=True, format="ascii.commented_header")
