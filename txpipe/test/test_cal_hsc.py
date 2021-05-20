from ..utils.calibration_tools import MeanShearInBins, HSCCalculator
from ..utils import HSCCalibrator, NullCalibrator
import numpy as np
import mockmpi



def select_all_bool(data):
    return np.repeat(True, data['g2'].size)

def select_all_index(data):
    return np.arange(data['g2'].size)

def select_all_where(data):
    # we just want to select everything here too
    return np.where(data['g2'] * 0 == 0)



def core_hsc(comm):

    nproc = 1 if comm is None else comm.size

    N = 10
    g1_true = np.random.normal(0, 0.1, size=N)
    g2_true = np.random.normal(0, 0.1, size=N)
    g_true = np.array([g1_true, g2_true])
    R_true = [0.96]
    K_true = [0.11]
    m = np.array(K_true*len(g1_true))
    sigma_e = np.array([0.2]*len(g1_true))
    R_true = np.array(R_true)
    K_true = np.array(K_true)
    g = (g_true*(1+K_true))*(2*R_true)
    weight = np.random.uniform(0, 1, size=N)

    data = {
        "g1": g[0],
        "g2": g[1],
        "m": m,
        "sigma_e": sigma_e,
        "weight": weight,
    }

    # test each type of selector
    for sel in [select_all_bool, select_all_where, select_all_index]:
        cal = HSCCalculator(select_all_bool)
        cal.add_data(data)
        R, K, n = cal.collect(comm)
        print('K_true',K_true)
        print('K',K)
        print('R_true',R_true)
        print('R',R)
        assert np.allclose(R, R_true)
        assert np.allclose(K, K_true)
        assert n == N * nproc

    # test each type of selector
    for sel in [select_all_bool, select_all_where, select_all_index]:
        cal = HSCCalculator(sel)
        cal.add_data(data)
        R, K, n = cal.collect(comm)
        print('K_true',K_true)
        print('K',K)
        print('R_true',R_true)
        print('R',R)

        assert np.allclose(R, R_true)
        assert np.allclose(K, K_true)
        assert n == N * nproc

def test_hsc_serial():
    core_hsc(None)

def test_hsc_parallel():
    mockmpi.mock_mpiexec(2, core_hsc)
    mockmpi.mock_mpiexec(10, core_hsc)


def test_mean_shear():
    name = "x"
    limits = [-1., 0., 1.]
    delta_gamma = 0.02

    # equal weights
    b1 = MeanShearInBins(name, limits, delta_gamma, shear_catalog_type='hsc')

    # fake data set in two bins, with the x values in the middle
    # of the bins and the g1, g2 some fixed simple values, with metacal
    # factors perfectly applied
    data = {
        "x": np.array([-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5]),
        "g1": np.array([-0.7, -0.6, -0.4, -0.3, 0.7, 0.6, 0.4, 0.3]),
        "g2": 2*np.array([-0.7, -0.6, -0.4, -0.3, 0.7, 0.6, 0.4, 0.3]),
        "c1": np.array([-0.07, -0.06, -0.04, -0.03, 0.07, 0.06, 0.04, 0.03]),
        "c2": 2*np.array([-0.07, -0.06, -0.04, -0.03, 0.07, 0.06, 0.04, 0.03]),
        "m": np.array([1., 1., 1., 1., 1., 1., 1., 1.]),
        "sigma_e": np.array([0., 0., 0., 0., 0., 0., 0., 0.]),
        "weight": np.array([1, 1, 1, 1, 1, 1, 1, 1]),
    }
    data['g1']+=data['c1']
    data['g2']+=data['c2']
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
    b1 = MeanShearInBins(name, limits, delta_gamma, shear_catalog_type='hsc')

    x = np.array([-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5])
    g1 = np.array([-0.7, -0.6, -0.4, -0.3, 0.7, 0.6, 0.4, 0.3])
    g2 = 2 * g1
    c1 = np.array([-0.07, -0.06, -0.04, -0.03, 0.07, 0.06, 0.04, 0.03])
    c2 = 2 * c1
    g1 += c1
    g2 += c2
    data = {
        "x": x,
        "g1": g1,
        "g2":    g2,
        "c1": np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        "c2": np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        "weight": np.array([1, 1, 0, 0, 0, 0, 1, 1]),
        "m": np.array([1., 1., 1., 1., 1., 1., 1., 1.]),
        "sigma_e": np.array([0., 0., 0., 0., 0., 0., 0., 0.])
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


def test_hsc_scalar():
    # hsc calibrator
    R = np.array([0.9])
    K = np.array([0.11])
    g1 = 0.2
    g2 = -0.3
    c1 = 0.02
    c2 = 0.03
    g1_obs = (g1 * (1 + K) + c1) * (2 * R)
    g2_obs = (g2 * (1 + K) + c2) * (2 * R)
    g_obs = np.array([g1_obs,g2_obs])
    cal = HSCCalibrator(R, K)
    g1_, g2_ = cal.apply(g_obs[0], g_obs[1], c1, c2)

    assert np.allclose(g1_, g1)
    assert np.allclose(g2_, g2)
    assert type(g1) == float
    assert type(g2) == float

def test_hsc_array():
    # array version
    R = np.array([1.0])
    K = np.array([0.0])
    cal = HSCCalibrator(R, K)
    g1 = np.random.normal(size=10)
    g2 = np.random.normal(size=10)
    c1 = 0.1*g1
    c2 = 0.1*g2
    g1_obs = (g1 * (1 + K) + c1) * (2 * R)
    g2_obs = (g2 * (1 + K) + c2) * (2 * R)

    g_obs = np.array([g1_obs,g2_obs])

    g1_, g2_ = cal.apply(g_obs[0], g_obs[1], c1, c2)

    print('test g1',g1)
    print('test g2',g2)

    print('tool g1',g1_)
    print('tool g2',g2_)

    assert np.allclose(g1_, g1)
    assert np.allclose(g2_, g2)
    assert type(g1) == np.ndarray
    assert type(g2) == np.ndarray


#def test_null():
    # null calibrator
#    R = 1
#    K = 1
#    g1 = 0.2
#    g2 = -0.3
#    c1 = 0
#    c2 = 0
#    g1 = (g1+c1)*(1+K)*(2*R)
#    g2 = (g2+c2)*(1+K)*(2*R)
#    g_obs = np.array([g1, g2])
#    assert g_obs.shape == (2,)
#    cal = NullCalibrator()
#    g1_, g2_ = cal.apply(float(g_obs[0]), float(g_obs[1]), c1, c2)
#    assert np.allclose(g1_, g1)
#    assert np.allclose(g2_, g2)
#    assert type(g1) == float
#    assert type(g2) == float

#    g1 = np.random.normal(size=10)
#    g2 = np.random.normal(size=10)
#    g1 = (g1+c1)*(1+K)*(2*R)
#    g2 = (g2+c2)*(1+K)*(2*R)
#    g_obs = np.array([g1, g2])
#    g1_, g2_ = cal.apply(g_obs[0], g_obs[1], c1, c2)

#    assert np.allclose(g1_, g1)
#    assert np.allclose(g2_, g2)
#    assert type(g1) == np.ndarray
#    assert type(g2) == np.ndarray



if __name__ == '__main__':
    test_hsccalibrator_serial()
    test_hsccalibrator_parallel()
    test_mean_shear()
    test_mean_shear_weights()
