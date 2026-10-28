from ..utils.theory import theory_3x2pt, fill_empty_sacc
import pyccl
import pytest
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


@pytest.mark.parametrize(
    "space, data_types",
    [
        ("fourier", ["galaxy_shear_cl_ee", "galaxy_shearDensity_cl_e", "galaxy_density_cl"]),
        ("real", ["galaxy_shear_xi_plus", "galaxy_shear_xi_minus", "galaxy_shearDensity_xi_t", "galaxy_density_xi"]),
    ],
)
def test_firecrown_theory(space, data_types):
    sacc_data = make_sacc()
    if space == "fourier":
        fill_empty_sacc(sacc_data, ell_values=np.geomspace(10, 2000, 20))
    else:
        fill_empty_sacc(sacc_data, theta_values=np.geomspace(2.5, 250, 20))
    cosmo = pyccl.CosmologyVanillaLCDM()
    theory_sacc = theory_3x2pt(cosmo, sacc_data)

    mean = theory_sacc.get_mean()
    assert len(mean) == len(sacc_data)
    assert np.all(np.isfinite(mean))

    # Every 3x2pt statistic should have a non-zero prediction
    assert set(theory_sacc.get_data_types()) == set(data_types)
    for data_type in data_types:
        assert np.all(theory_sacc.get_mean(data_type) != 0)
