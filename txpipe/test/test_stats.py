from ..utils.stats import ParallelStatsCalculator
from .mock_mpi import mock_mpiexec
import numpy as np


def example(comm):
    p = ParallelStatsCalculator(2, weighted=False)
    pixel = 0
    values = np.array([1., 2., 3., 4., 5.])
    p.add_data(pixel, values)
    pixel = 1
    values = 10 + np.array([1., 2., 3., 4., 5.])
    p.add_data(pixel, values)
    w, mu, sigma2 = p.collect(comm)

    expected_weight = np.array([5, 5])
    expected_sigma2 = np.var([-2,-1,0,1,2])
    expected_mu = np.array([3, 13])

    if comm is not None:
        expected_weight *= comm.size

    if comm is None or comm.rank == 0:
        # print(w, expected_weight)
        # print(mu, expected_mu)
        # print(sigma2, expected_sigma2)
        assert np.allclose(w, expected_weight)
        assert np.allclose(mu, expected_mu)
        assert np.allclose(sigma2, expected_sigma2)

def example_weights(comm):
    p = ParallelStatsCalculator(2, weighted=True)
    pixel = 0
    values = np.array([1., 2., 3., 4., 5.])
    weights = np.array([0., 0., 1., 1., 1.])
    p.add_data(pixel, values, weights=weights)
    pixel = 1
    values = np.array([1., 2., 3., 4., 5.])
    weights = np.array([0., 0., 0.5, 0.5, 0.5])
    p.add_data(pixel, values, weights=weights)
    w, mu, sigma2 = p.collect(comm)

    expected_weight = np.array([3, 1.5])
    expected_mu = np.array([4.0, 4.0])
    expected_sigma2 = np.var([-1,0,1,-1,0,1])

    if comm is not None:
        expected_weight *= comm.size

    if comm is None or comm.rank == 0:
        # print(w, expected_weight)
        # print(mu, expected_mu)
        # print(sigma2, expected_sigma2)
        assert np.allclose(w, expected_weight)
        assert np.allclose(mu, expected_mu)
        assert np.allclose(sigma2, expected_sigma2)

def example_missing(comm):
    # now set three pixels, with nothing in the third one
    p = ParallelStatsCalculator(3, weighted=True)
    pixel = 0
    values = np.array([1., 2., 3., 4., 5.])
    weights = np.array([0., 0., 1., 1., 1.])
    p.add_data(pixel, values, weights=weights)
    pixel = 1
    values = np.array([1., 2., 3., 4., 5.])
    weights = np.array([0., 0., 0.5, 0.5, 0.5])
    p.add_data(pixel, values, weights=weights)
    w, mu, sigma2 = p.collect(comm)

    expected_weight = np.array([3, 1.5])
    expected_mu = np.array([4.0, 4.0])
    expected_sigma2 = np.var([-1,0,1,-1,0,1])

    if comm is not None:
        expected_weight *= comm.size

    if comm is None or comm.rank == 0:
        assert np.allclose(w[:2], expected_weight)
        assert np.allclose(mu[:2], expected_mu)
        assert np.allclose(sigma2[:2], expected_sigma2)
        assert w[2] == 0
        assert np.isnan(mu[2])
        assert np.isnan(sigma2[2])

def example_partial_missing(comm):
    if comm is None:
        return
    # now set three pixels, with nothing in the third one
    # on one proc but something on another
    p = ParallelStatsCalculator(3, weighted=True)

    pixel = 0
    values = np.array([1., 2., 3., 4., 5.])
    weights = np.array([0., 0., 1., 1., 1.])
    p.add_data(pixel, values, weights=weights)

    pixel = 1
    values = np.array([1., 2., 3., 4., 5.])
    weights = np.array([0., 0., 0.5, 0.5, 0.5])
    p.add_data(pixel, values, weights=weights)

    pixel = 2
    values = np.array([1., 2., 3., 4., 5.])
    if comm.rank == 0:
        weights = np.array([0., 0., 0.3, 0.3, 0.3])
    else:
        weights = np.array([0., 0., 0.0, 0.0, 0.0])
    p.add_data(pixel, values, weights=weights)

    w, mu, sigma2 = p.collect(comm)
    expected_weight = np.array([3, 1.5, 0.9])
    expected_mu = np.array([4.0, 4.0, 4.0])
    expected_sigma2 = np.var([-1,0,1])

    if comm is not None:
        expected_weight[:2] *= comm.size

    if comm is None or comm.rank == 0:
        assert np.allclose(w, expected_weight)
        assert np.allclose(mu, expected_mu)
        assert np.allclose(sigma2, expected_sigma2)



def test_stats():
    example(None)
    mock_mpiexec(2,example)
    example_weights(None)
    mock_mpiexec(2,example_weights)
    example_missing(None)
    mock_mpiexec(2,example_missing)
    mock_mpiexec(2,example_partial_missing)

if __name__ == '__main__':
    test_stats()
