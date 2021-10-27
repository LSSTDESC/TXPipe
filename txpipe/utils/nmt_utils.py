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
    def __init__(self, nside=None, bpws=None, ells=None, weights=None, nlb=None, lmax=None, is_Dell=False, f_ell=None):
        super().__init__(nside=nside, bpws=bpws, ells=ells, weights=weights, nlb=nlb, lmax=lmax, is_Dell=False, f_ell=None)
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

        p = self.get_path(key)

        if not p.exists():
            return None

        # Initialize a workspace and populate
        # it from file
        workspace = nmt.NmtWorkspace()
        workspace.read_from(str(p))

        self._loaded[key] = workspace

        return workspace

    def get_path(self, key):
        return self.path / f'workspace_{key}.dat'


    def put(self, workspace):
        key = workspace.txpipe_key
        p = self.get_path(key)
        if p.exists():
            return False

        print(f"Saving workspace to {p}")
        workspace.write_to(str(p))


def build_MyNmtBin_from_binning_info(ell_min, ell_max, n_ell, ell_spacing):
    # Creating the ell binning from the edges using this Namaster constructor.
    if ell_spacing == 'log':
        edges = np.unique(np.geomspace(ell_min, ell_max, n_ell).astype(int))
    else:
        edges = np.unique(np.linspace(ell_min, ell_max, n_ell).astype(int))

    ell_bins = MyNmtBin.from_edges(edges[:-1], edges[1:], is_Dell=False)

    return ell_bins
