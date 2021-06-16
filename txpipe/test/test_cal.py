from ..utils.calibration_tools import MeanShearInBins, MetacalCalculator
from ..utils import MetaCalibrator, LensfitCalibrator, NullCalibrator
import numpy as np
import mockmpi



def select_all_bool(data):
    return np.repeat(True, data['mcal_g2'].size)

def select_all_index(data):
    return np.arange(data['mcal_g2'].size)

def select_all_where(data):
    # we just want to select everything here too
    return np.where(data['mcal_g2'] * 0 == 0)



def core_metacal(comm):
    delta_gamma = 0.02

    nproc = 1 if comm is None else comm.size

    N = 10
    g1_true = np.random.normal(0, 0.1, size=N)
    g2_true = np.random.normal(0, 0.1, size=N)
    g_true = np.array([g1_true, g2_true])
    R_true = np.array([[0.9, 0.1], [0.07, 0.8]])
    g = R_true @ g_true
    g_1p = R_true @ (g_true + 0.5*delta_gamma * np.array([+1, 0])[:, np.newaxis])
    g_1m = R_true @ (g_true + 0.5*delta_gamma * np.array([-1, 0])[:, np.newaxis])
    g_2p = R_true @ (g_true + 0.5*delta_gamma * np.array([0, +1])[:, np.newaxis])
    g_2m = R_true @ (g_true + 0.5*delta_gamma * np.array([0, -1])[:, np.newaxis])
    weight = np.ones(N)

    data = {
        "mcal_g1": g[0],
        "mcal_g1_1p": g_1p[0],
        "mcal_g1_1m": g_1m[0],
        "mcal_g1_2p": g_2p[0],
        "mcal_g1_2m": g_2m[0],
        "mcal_g2": g[1],
        "mcal_g2_1p": g_1p[1],
        "mcal_g2_1m": g_1m[1],
        "mcal_g2_2p": g_2p[1],
        "mcal_g2_2m": g_2m[1],
        "weight": weight,
    }

    # test each type of selector
    for sel in [select_all_bool, select_all_where, select_all_index]:
        cal = MetacalCalculator(select_all_bool, delta_gamma)
        cal.add_data(data)
        R, S, n = cal.collect(comm, allgather=True)

        assert np.allclose(R, R_true)
        assert np.allclose(S, 0.0)
        assert n == N * nproc

    # equal non-unit weights - everything should be the same.
    data["weight"] *= 0.5
    # test each type of selector
    for sel in [select_all_bool, select_all_where, select_all_index]:
        cal = MetacalCalculator(sel, delta_gamma)
        cal.add_data(data)
        R, S, n = cal.collect(comm, allgather=True)
        print("R = ", R)

        assert np.allclose(R, R_true)
        assert np.allclose(S, 0.0)
        assert n == N * nproc

    # random weights.  since R is constant this should still be the same
    data["weight"] = np.random.uniform(0, 1, size=N)
    # test each type of selector
    for sel in [select_all_bool, select_all_where, select_all_index]:
        cal = MetacalCalculator(sel, delta_gamma)
        cal.add_data(data)
        R, S, n = cal.collect(comm, allgather=True)
        print("R = ", R)

        assert np.allclose(R, R_true)
        assert np.allclose(S, 0.0)
        assert n == N * nproc

def test_metacalibrator_serial():
    core_metacal(None)

def test_metacalibrator_parallel():
    mockmpi.mock_mpiexec(2, core_metacal)
    mockmpi.mock_mpiexec(10, core_metacal)


def test_mean_shear():
    name = "x"
    limits = [-1., 0., 1.]
    delta_gamma = 0.02

    # equal weights
    b1 = MeanShearInBins(name, limits, delta_gamma)

    # fake data set in two bins, with the x values in the middle
    # of the bins and the g1, g2 some fixed simple values, with metacal
    # factors perfectly applied
    data = {
        "x": np.array([-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5]),
        "x_1p": np.array([-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5]),
        "x_1m": np.array([-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5]),
        "x_2p": np.array([-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5]),
        "x_2m": np.array([-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5]),
        "mcal_g1": np.array([-0.7, -0.6, -0.4, -0.3, 0.7, 0.6, 0.4, 0.3]),
        "mcal_g1_1p": np.array([-0.7, -0.6, -0.4, -0.3, 0.7, 0.6, 0.4, 0.3]) + 0.5*delta_gamma,
        "mcal_g1_1m": np.array([-0.7, -0.6, -0.4, -0.3, 0.7, 0.6, 0.4, 0.3]) - 0.5*delta_gamma,
        "mcal_g1_2p": np.array([-0.7, -0.6, -0.4, -0.3, 0.7, 0.6, 0.4, 0.3]),
        "mcal_g1_2m": np.array([-0.7, -0.6, -0.4, -0.3, 0.7, 0.6, 0.4, 0.3]),
        "mcal_g2": 2*np.array([-0.7, -0.6, -0.4, -0.3, 0.7, 0.6, 0.4, 0.3]),
        "mcal_g2_1p": 2*np.array([-0.7, -0.6, -0.4, -0.3, 0.7, 0.6, 0.4, 0.3]),
        "mcal_g2_1m": 2*np.array([-0.7, -0.6, -0.4, -0.3, 0.7, 0.6, 0.4, 0.3]),
        "mcal_g2_2p": 2*np.array([-0.7, -0.6, -0.4, -0.3, 0.7, 0.6, 0.4, 0.3]) + 0.5*delta_gamma,
        "mcal_g2_2m": 2*np.array([-0.7, -0.6, -0.4, -0.3, 0.7, 0.6, 0.4, 0.3]) - 0.5*delta_gamma,
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
    expected_sigma2 = 2*expected_sigma1
    assert np.allclose(sigma1, expected_sigma1)
    assert np.allclose(sigma2, expected_sigma2)

def test_mean_shear_weights():
    name = "x"
    limits = [-1., 0., 1.]
    delta_gamma = 0.02
    # downweighting half of samples
    b1 = MeanShearInBins(name, limits, delta_gamma)

    x = np.array([-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5])
    g1 = np.array([-0.7, -0.6, -0.4, -0.3, 0.7, 0.6, 0.4, 0.3])
    g2 = 2 * g1
    data = {
        "x": x,
        "x_1p": x,
        "x_1m": x,
        "x_2p": x,
        "x_2m": x,
        "mcal_g1": g1,
        "mcal_g1_1p": g1 + 0.5*delta_gamma,
        "mcal_g1_1m": g1 - 0.5*delta_gamma,
        "mcal_g1_2p": g1,
        "mcal_g1_2m": g1,
        "mcal_g2":    g2,
        "mcal_g2_1p": g2,
        "mcal_g2_1m": g2,
        "mcal_g2_2p": g2 + 0.5*delta_gamma,
        "mcal_g2_2m": g2 - 0.5*delta_gamma,
        "weight": np.array([1, 1, 0, 0, 0, 0, 1, 1]),
    }
    b1.add_data(data)

    mu, g1, g2, sigma1, sigma2 = b1.collect()
    # Now we have downweighted some of the samples some of these values change.
    assert np.allclose(mu, [-0.5, 0.5])
    assert np.allclose(g1, [-0.65, 0.35])
    assert np.allclose(g2, [-1.3, 0.7])
    expected_sigma1 = np.std([-0.2, -0.1]) / np.sqrt(2)
    expected_sigma2 = 2*expected_sigma1
    print(sigma1, expected_sigma1)
    assert np.allclose(sigma1, expected_sigma1)
    assert np.allclose(sigma2, expected_sigma2)


def test_metacal_scalar():
    # metacal calibrator
    R = np.array([[2, 3], [4, 5]])
    g1 = 0.2
    g2 = -0.3
    g_obs = R @ [g1, g2]
    mu = np.zeros(2)
    cal = MetaCalibrator(R, mu)
    g1_, g2_ = cal.apply(g_obs[0], g_obs[1])

    assert np.allclose(g1_, g1)
    assert np.allclose(g2_, g2)
    assert type(g1) == float
    assert type(g2) == float

def test_metacal_array():
    # array version
    R = np.array([[2, 3], [4, 5]])
    mu = np.zeros(2)
    cal = MetaCalibrator(R, mu)
    g1 = np.random.normal(size=10)
    g2 = np.random.normal(size=10)
    g_obs = R @ [g1, g2]
    g1_, g2_ = cal.apply(g_obs[0], g_obs[1])

    assert np.allclose(g1_, g1)
    assert np.allclose(g2_, g2)
    assert type(g1) == np.ndarray
    assert type(g2) == np.ndarray

def test_metacal_mean():
    # array version with mean
    R = np.array([[2, 3], [4, 5]])
    mu = [0.1, 0.2]
    g1 = np.random.normal(size=10)
    g2 = np.random.normal(size=10)
    g_obs = R @ [g1, g2]
    g_obs[0] += mu[0]
    g_obs[1] += mu[1]
    cal = MetaCalibrator(R, mu, mu_is_calibrated=False)
    g1_, g2_ = cal.apply(g_obs[0], g_obs[1])
    assert np.allclose(g1_, g1)
    assert np.allclose(g2_, g2)
    assert type(g1) == np.ndarray
    assert type(g2) == np.ndarray


    g_obs = R @ [g1 + mu[0], g2 + mu[1]]
    cal = MetaCalibrator(R, mu, mu_is_calibrated=True)
    g1_, g2_ = cal.apply(g_obs[0], g_obs[1])
    assert np.allclose(g1_, g1)
    assert np.allclose(g2_, g2)
    assert type(g1) == np.ndarray
    assert type(g2) == np.ndarray



def test_null():
    # null calibrator
    R = np.eye(2)
    g1 = 0.2
    g2 = -0.3
    g_obs = R @ [g1, g2]
    assert g_obs.shape == (2,)
    cal = NullCalibrator()
    g1_, g2_ = cal.apply(float(g_obs[0]), float(g_obs[1]))
    assert np.allclose(g1_, g1)
    assert np.allclose(g2_, g2)
    assert type(g1) == float
    assert type(g2) == float
    
    g1 = np.random.normal(size=10)
    g2 = np.random.normal(size=10)
    g_obs = R @ [g1, g2]
    g1_, g2_ = cal.apply(g_obs[0], g_obs[1])

    assert np.allclose(g1_, g1)
    assert np.allclose(g2_, g2)
    assert type(g1) == np.ndarray
    assert type(g2) == np.ndarray



if __name__ == '__main__':
    test_metacalibrator_serial()
    test_metacalibrator_parallel()
    test_mean_shear()
    test_mean_shear_weights()
