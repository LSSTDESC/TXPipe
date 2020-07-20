from ..utils.calibration_tools import MeanShearInBins
import numpy as np

def test_mean_shear():
    name = "x"
    limits = [-1., 0., 1.]
    delta_gamma = 0.02

    # equal weights
    b1 = MeanShearInBins(name, limits, delta_gamma)

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
    expected_sigma1 = np.std([-0.2, -0.1, 0.1, 0.2]) / np.sqrt(4)
    expected_sigma2 = 2*expected_sigma1
    assert np.allclose(sigma1, expected_sigma1)
    assert np.allclose(sigma2, expected_sigma2)

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
    assert np.allclose(mu, [-0.5, 0.5])
    assert np.allclose(g1, [-0.65, 0.35])
    assert np.allclose(g2, [-1.3, 0.7])
    expected_sigma1 = np.std([-0.2, -0.1]) / np.sqrt(2)
    expected_sigma2 = 2*expected_sigma1
    print(sigma1, expected_sigma1)
    assert np.allclose(sigma1, expected_sigma1)
    assert np.allclose(sigma2, expected_sigma2)


if __name__ == '__main__':
    test_mean_shear()