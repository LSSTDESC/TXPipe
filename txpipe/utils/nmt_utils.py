import pymaster as nmt
import numpy as np
import healpy
import pathlib

class MyNmtBinFlat(nmt.NmtBinFlat):
    def __init__(self, l0, lf):
        super().__init__(l0, lf)
        self.ell_min = l0
        self.ell_max = lf

    def get_window(self, b):
        ell = np.arange(self.ell_min[b], self.ell_max[b]+1)
        w = np.ones_like(ell)
        return (ell, w)

    def get_ell_min(self, b):
        return self.ell_min[b]

    def get_ell_max(self, b):
        return self.ell_max[b]

    def is_flat(self):
        return True

    def apply_window(self, b, c_ell):
        b0, b1 = self.get_window(b)
        return c_ell[b0:b1+1].mean()

class MyNmtBin(nmt.NmtBin):
    def __init__(self, nside, bpws=None, ells=None, weights=None, nlb=None, lmax=None):
        super().__init__(nside, bpws=bpws, ells=ells, weights=weights, nlb=nlb, lmax=lmax)
        self.ell_max = self.lmax

    def get_window(self, b):
        ls = self.get_ell_list(b)
        w = self.get_weight_list(b)
        return (ls, w)

    def get_ell_min(self, b):
        return self.get_ell_list(b)[0]

    def get_ell_max(self, b):
        return self.get_ell_list(b)[-1]

    def is_flat(self):
        return False

    def apply_window(self, b, c_ell):
        ell, weight = self.get_window(b)
        return (c_ell[ell]*weight).sum() / weight.sum()


import healpy



class WorkspaceCache:
    def __init__(self, dirname):
        self.path = pathlib.Path(dirname)
        self.path.mkdir(exist_ok=True)
        self._loaded = {}

    def get(self, key):
        if key in self._loaded:
            return self._loaded[key]

        p = self.path / f'workspace_{key}.dat'

        if not p.exists():
            return None

        # Initialize a workspace and populate
        # it from file
        workspace = nmt.NmtWorkspace()
        workspace.read_from(str(p))

        self._loaded[key] = workspace

        return workspace

    def put(self, workspace):
        key = workspace.txpipe_key
        p = self.path / f'workspace_{key}.dat'
        if p.exists():
            return False

        print(f"Saving workspace to {p}")
        workspace.write_to(str(p))
        return True


def iterate_randomly_rotated_maps(g1, g2, n_rot):
    # Mask
    good = (g1 != healpy.UNSEEN) & (g2 != healpy.UNSEEN)
    n = good.sum()

    maps = []
    for i in range(n_rot):
        # Do the random rotation, just on the good pixels,
        # since that's faster
        phi = np.random.uniform(0, 2*np.pi, n)
        rot = np.exp(1j*phi)
        g_rot = rot * (g1[good] + g2[good]*1j)

        # Make a map full of UNSEENs to start with
        g1_rot = np.repeat(healpy.UNSEEN, g1.shape)
        g2_rot = np.repeat(healpy.UNSEEN, g2.shape)

        # Fill in the rotated pixels
        g1_rot[good] = g_rot.real
        g2_rot[good] = g_rot.imag
        yield g1_rot, g2_rot

#     return maps

def iterate_randomly_rotated_fields(g1, g2, lw, n_rot):
    for g1_rot, g2_rot in iterate_randomly_rotated_maps(g1, g2, n_rot):
        field = nmt.NmtField(lw, [g1_rot, g2_rot])
        yield field
