import pymaster as nmt
import numpy as np

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

