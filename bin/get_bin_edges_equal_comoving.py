import jax_cosmo
import jax_cosmo.background
import numpy as np

def get_z_edges(n_bin, zmax):
    zgrid = np.linspace(0, zmax, 1000)
    agrid = 1 / (1 + zgrid)
    model = jax_cosmo.parameters.Planck15()
    chi_grid = jax_cosmo.background.radial_comoving_distance(model, agrid)
    # Compute bin edges that are equally spaced in chi.
    chi_edges = np.linspace(0, chi_grid[-1], n_bin + 1)
    z_edges = np.empty(n_bin + 1)
    z_edges[0] = 0.
    z_edges[-1] = zmax
    z_edges[1:-1] = np.interp(chi_edges[1:-1], chi_grid, zgrid)
    print('Cosmology used:', model)
    print('n_bin, zmax:', n_bin, zmax)
    print('z edges:', z_edges)


n_bin = 5
zmax = 2.
get_z_edges(n_bin, zmax)

n_bin = 8
zmax = 3.
get_z_edges(n_bin, zmax)




