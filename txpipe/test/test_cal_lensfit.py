from ..utils.calibration_tools import MeanShearInBins, LensfitCalculator
from ..utils import LensfitCalibrator, NullCalibrator
import numpy as np
import mockmpi


def select_all_bool(data):
    return np.repeat(True, data["g2"].size)


def select_all_index(data):
    return np.arange(data["g2"].size)


def select_all_where(data):
    # we just want to select everything here too
    return np.where(data["g2"] * 0 == 0)


def core_lensfit(comm):

    nproc = 1 if comm is None else comm.size

    N = 10
    dec = np.random.normal(-5, 4, size=N)
    g1_true = np.random.normal(0, 0.1, size=N)
    g2_true = np.random.normal(0, 0.1, size=N)
    K_true = np.array([0.11])
    m = np.array([K_true[0]] * len(g1_true))
    g1 = (g1_true) * (1 + K_true[0])
    g2 = (g1_true) * (1 + K_true[0])
    weight = np.random.uniform(0, 1, size=N)

    if comm is None:
        C1_true = np.average(g1, weights=weight)
        C2_true = np.average(g2, weights=weight)
    else:
        wsum = np.zeros_like(g1)
        comm.Allreduce(weight, wsum)
        csum = np.zeros_like(g1)
        comm.Allreduce(g1 * weight, csum)
        C1_true = csum.sum() / wsum.sum()
        comm.Allreduce(g2 * weight, csum)
        C2_true = csum.sum() / wsum.sum()

    C_true = np.array([C1_true, C2_true])  # mean of g1, g2

    data = {
        "dec":dec,
        "g1": g1,
        "g2": g2,
        "m": m,
        "weight": weight,
    }

    # test each type of selector
    for sel in [select_all_bool, select_all_where, select_all_index]:
        cal = LensfitCalculator(sel)
        cal.add_data(data)

        K, C_N, C_S, n, _ = cal.collect(comm, allgather=True)
        assert np.allclose(C, C_true)
        assert np.allclose(K, K_true)
        assert n == N * nproc


def test_lensfit_serial():
    core_lensfit(None)


def test_lensfit_parallel():
    mockmpi.mock_mpiexec(2, core_lensfit)
    mockmpi.mock_mpiexec(10, core_lensfit)


def test_mean_shear():
    name = "x"
    limits = [-1.0, 0.0, 1.0]
    delta_gamma = 0.02

    # equal weights
    b1 = MeanShearInBins(name, limits, delta_gamma, shear_catalog_type="lensfit")

    # fake data set in two bins, with the x values in the middle
    # of the bins and the g1, g2 some fixed simple values, with metacal
    # factors perfectly applied
    data = {
        "dec": np.random.normal(-5, 4, size=8),
        "x": np.array([-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5]),
        "g1": np.array([-0.7, -0.6, -0.4, -0.3, 0.7, 0.6, 0.4, 0.3]),
        "g2": 2 * np.array([-0.7, -0.6, -0.4, -0.3, 0.7, 0.6, 0.4, 0.3]),
        "m": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "weight": np.array([1, 1, 1, 1, 1, 1, 1, 1]),
    }
    b1.add_data(data)

    mu, g1, g2, sigma1, sigma2 = b1.collect()

    assert np.allclose(mu, [-0.5, 0.5])
    assert np.allclose(g1, [-0.5, 0.5])
    assert np.allclose(g2, [-1.0, 1.0])

    # using var([0.7, 0.6, 0.4, 0.3]) == var([-0.2, -0.1, 0.1, 0.2])
    # this should equal the sigma (error on the mean) from the numbers above.
    expected_sigma1 = np.std([-0.2, -0.1, 0.1, 0.2]) / np.sqrt(4)
    expected_sigma2 = 2 * expected_sigma1

    assert np.allclose(sigma1, expected_sigma1)
    assert np.allclose(sigma2, expected_sigma2)


def test_mean_shear_weights():
    name = "x"
    limits = [-1.0, 0.0, 1.0]
    delta_gamma = 0.02
    # downweighting half of samples
    b1 = MeanShearInBins(name, limits, delta_gamma, shear_catalog_type="lensfit")
    dec = np.random.normal(-5, 4, size=8)
    x = np.array([-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5])
    g1 = np.array([-0.7, -0.6, -0.4, -0.3, 0.7, 0.6, 0.4, 0.3])
    g2 = 2 * g1
    data = {
        "dec": dec,
        "x": x,
        "g1": g1,
        "g2": g2,
        "m": np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        "weight": np.array([1, 1, 0, 0, 0, 0, 1, 1]),
    }
    b1.add_data(data)

    mu, g1, g2, sigma1, sigma2 = b1.collect()
    # Now we have downweighted some of the samples some of these values change.
    assert np.allclose(mu, [-0.5, 0.5])
    assert np.allclose(g1, [-0.65, 0.35])
    assert np.allclose(g2, [-1.3, 0.7])
    expected_sigma1 = np.std([-0.2, -0.1]) / np.sqrt(2)
    expected_sigma2 = 2 * expected_sigma1

    assert np.allclose(sigma1, expected_sigma1)
    assert np.allclose(sigma2, expected_sigma2)


def test_lensfit_scalar():
    # lensfit calibrator
    K = np.array([0.9])
    C = np.array([0.11, 0.22])
    dec = -3
    g1 = 0.2
    g2 = -0.3
    g1_obs = (g1) * (1 + K[0]) + C[0]
    g2_obs = (g2) * (1 + K[0]) + C[1]
    g_obs = np.array([g1_obs, g2_obs])
    cal = LensfitCalibrator(K, C)
    g1_, g2_ = cal.apply(dec,g_obs[0], g_obs[1], subtract_mean=True)

    assert np.allclose(g1_, g1)
    assert np.allclose(g2_, g2)
    assert type(g1) == float
    assert type(g2) == float


def test_lensfit_array():
    # array version
    K = np.array([0.9])
    C = np.array([0.11, 0.22])
    cal = LensfitCalibrator(K, C)
    dec = np.random.normal(-5, 4, size=10)
    g1 = np.random.normal(size=10)
    g2 = np.random.normal(size=10)
    g1_obs = (g1) * (1 + K[0]) + C[0]
    g2_obs = (g2) * (1 + K[0]) + C[1]
    g_obs = [g1_obs, g2_obs]
    g1_, g2_ = cal.apply(g_obs[0], g_obs[1])

    assert np.allclose(g1_, g1)
    assert np.allclose(g2_, g2)
    assert type(g1) == np.ndarray
    assert type(g2) == np.ndarray


def test_null():
    # null calibrator
    K = 1.0
    C = [0.0, 0.0]
    dec = -3
    g1 = 0.2
    g2 = -0.3
    g1 = (g1 + C[0]) * (K)
    g2 = (g2 + C[1]) * (K)
    g_obs = np.array([g1, g2])
    assert g_obs.shape == (2,)
    cal = NullCalibrator()
    g1_, g2_ = cal.apply(dec, float(g_obs[0]), float(g_obs[1]))
    assert np.allclose(g1_, g1)
    assert np.allclose(g2_, g2)
    assert type(g1) == float
    assert type(g2) == float
    dec = np.random.normal(-5, 4, size=10)
    g1 = np.random.normal(size=10)
    g2 = np.random.normal(size=10)
    g1 = (g1 + C[0]) * (K)
    g2 = (g2 + C[1]) * (K)
    g_obs = [g1, g2]
    g1_, g2_ = cal.apply(dec,g_obs[0], g_obs[1])

    assert np.allclose(g1_, g1)
    assert np.allclose(g2_, g2)
    assert type(g1) == np.ndarray
    assert type(g2) == np.ndarray


if __name__ == "__main__":
    test_lensfit_serial()
    test_lensfit_parallel()
    test_mean_shear()
    test_mean_shear_weights()
