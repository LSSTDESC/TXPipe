from ..utils.theory import theory_3x2pt, fill_empty_sacc
import pyccl
import sacc
import numpy as np


def make_sacc():
    sacc_data = sacc.Sacc()
    nbin_source = 2
    nbin_lens = 3
    z = np.linspace(0.0, 2.0, 200)
    mu_source = [0.6, 1.0]
    mu_lens = [0.2, 0.3, 0.4]
    sigma = 0.1
    for i in range(nbin_source):
        Nz = np.exp(-0.5 * ((z - mu_source[i]) / sigma) ** 2)
        sacc_data.add_tracer("NZ", f"source_{i}", z, Nz)
    for i in range(nbin_lens):
        Nz = np.exp(-0.5 * ((z - mu_lens[i]) / sigma) ** 2)
        sacc_data.add_tracer("NZ", f"lens_{i}", z, Nz)
    return sacc_data



def test_firecrown_theory():
    sacc_data = make_sacc()
    ell_values = np.geomspace(10, 2000, 20)
    fill_empty_sacc(sacc_data, ell_values=ell_values)
    cosmo = pyccl.CosmologyVanillaLCDM()
    theory_sacc = theory_3x2pt(cosmo, sacc_data)
    print(theory_sacc.get_mean())