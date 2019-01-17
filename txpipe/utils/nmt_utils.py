import pymaster as nmt
import numpy as np

class MyNmtBinFlat(nmt.NmtBinFlat):
    def __init__(self, l0, lf):
        super().__init__(l0, lf)
        self.ell_min = l0
        self.ell_max = lf

    def get_window(self, b):
        return (self.ell_min[b], self.ell_max[b])
    
    def get_ell_min(self, b):
        return self.ell_min[b]

    def get_ell_max(self, b):
        return self.ell_max[b]

class MyNmtBin(nmt.NmtBin):
    def get_window(self, b):
        ls = self.get_ell_list(b)
        w = self.get_weight_list(b)
        return (ls, w)

    def get_ell_min(self, b):
        return self.get_ell_list(b)[0]

    def get_ell_max(self, b):
        return self.get_ell_list(b)[-1]
