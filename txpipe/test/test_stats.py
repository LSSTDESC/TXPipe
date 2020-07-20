from ..utils.stats import ParallelStatsCalculator, ParallelSum
from .mock_mpi import mock_mpiexec
import numpy as np


def run_stats(comm):
    # test the parallel stats calculator, without weights
    # create a two pixel "map" and put some values in for
    # each pixel
    p = ParallelStatsCalculator(2, weighted=False)
    pixel = 0
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    p.add_data(pixel, values)
    pixel = 1
    values = 10 + np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    p.add_data(pixel, values)

    # get result back and compare
    w, mu, sigma2 = p.collect(comm)
    expected_weight = np.array([5, 5])
    expected_sigma2 = np.var([-2, -1, 0, 1, 2])
    expected_mu = np.array([3, 13])

    # if testing under MPI, all procs put in the
    # same values, so the means and vars should not
    # change, but the weights should scale
    if comm is not None:
        expected_weight *= comm.size

    # only root process gets the results, others
    # get None
    if comm is None or comm.rank == 0:
        assert np.allclose(w, expected_weight)
        assert np.allclose(mu, expected_mu)
        assert np.allclose(sigma2, expected_sigma2)


def run_stats_weights(comm):
    # as above, but this time downweight some pixels
    # to zero to test the weights facility.
    # same 2-pixel "map"
    p = ParallelStatsCalculator(2, weighted=True)
    pixel = 0
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    p.add_data(pixel, values, weights=weights)
    pixel = 1
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([0.0, 0.0, 0.5, 0.5, 0.5])
    p.add_data(pixel, values, weights=weights)

    # results are different here
    w, mu, sigma2 = p.collect(comm)
    expected_weight = np.array([3, 1.5])
    expected_mu = np.array([4.0, 4.0])
    expected_sigma2 = np.var([-1, 0, 1, -1, 0, 1])

    if comm is not None:
        expected_weight *= comm.size

    if comm is None or comm.rank == 0:
        assert np.allclose(w, expected_weight)
        assert np.allclose(mu, expected_mu)
        assert np.allclose(sigma2, expected_sigma2)


def run_stats_sparse(comm):
    # same as above, but using a sparse mode
    # instsead, and three pixels.
    # The last pixel will not be set, so should come back with
    # NaNs
    p = ParallelStatsCalculator(3, weighted=True, sparse=True)
    pixel = 0
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    p.add_data(pixel, values, weights=weights)
    pixel = 1
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([0.0, 0.0, 0.5, 0.5, 0.5])
    p.add_data(pixel, values, weights=weights)

    # Expected results for pixels 0, 1
    w, mu, sigma2 = p.collect(comm)
    expected_weight = np.array([3, 1.5])
    expected_mu = np.array([4.0, 4.0])
    expected_sigma2 = np.var([-1, 0, 1, -1, 0, 1])

    if comm is not None:
        expected_weight *= comm.size

    if comm is None or comm.rank == 0:
        # first two pix should be as above
        for i in range(2):
            assert np.allclose(w[i], expected_weight[i])
            assert np.allclose(mu[i], expected_mu[i])
            assert np.allclose(sigma2[i], expected_sigma2)
        # last should be 0 and NaN
        assert w[2] == 0.0
        assert np.isnan(mu[2])
        assert np.isnan(sigma2[2])


def run_stats_missing(comm):
    # as above, but with an unhit pixel in non-sparse
    # mode, to check it comes back as NaN
    p = ParallelStatsCalculator(3, weighted=True)
    pixel = 0
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    p.add_data(pixel, values, weights=weights)
    pixel = 1
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([0.0, 0.0, 0.5, 0.5, 0.5])
    p.add_data(pixel, values, weights=weights)

    w, mu, sigma2 = p.collect(comm)
    expected_weight = np.array([3, 1.5])
    expected_mu = np.array([4.0, 4.0])
    expected_sigma2 = np.var([-1, 0, 1, -1, 0, 1])

    if comm is not None:
        expected_weight *= comm.size

    if comm is None or comm.rank == 0:
        assert np.allclose(w[:2], expected_weight)
        assert np.allclose(mu[:2], expected_mu)
        assert np.allclose(sigma2[:2], expected_sigma2)
        assert w[2] == 0
        assert np.isnan(mu[2])
        assert np.isnan(sigma2[2])


def run_stats_partial(comm):
    # Now a weighted version where one process doesn't
    # hit one pixel but others do. Should recover value
    # despite this.
    # In serial this is the same as the test above, so
    # skip it.
    if comm is None:
        return
    # now set three pixels, with nothing in the third one
    # on one proc but something on another
    p = ParallelStatsCalculator(3, weighted=True)

    pixel = 0
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    p.add_data(pixel, values, weights=weights)

    pixel = 1
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([0.0, 0.0, 0.5, 0.5, 0.5])
    p.add_data(pixel, values, weights=weights)

    pixel = 2
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # zero weight for all but one of the procs.
    if comm.rank == 0:
        weights = np.array([0.0, 0.0, 0.3, 0.3, 0.3])
    else:
        weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    p.add_data(pixel, values, weights=weights)

    w, mu, sigma2 = p.collect(comm)
    expected_weight = np.array([3, 1.5, 0.9])
    expected_mu = np.array([4.0, 4.0, 4.0])
    expected_sigma2 = np.var([-1, 0, 1])

    if comm is not None:
        expected_weight[:2] *= comm.size

    if comm is None or comm.rank == 0:
        assert np.allclose(w, expected_weight)
        assert np.allclose(mu, expected_mu)
        assert np.allclose(sigma2, expected_sigma2)


def run_sums(comm):
    # test of the paralle sum facility - only non-trivial
    # in parallel, so skip a serial test
    if comm is None:
        return

    # 10 bins to sum into
    s = ParallelSum(10)

    if comm.rank > 0:
        for i in range(10):
            s.add_data(i, [2.0])

    count, sums = s.collect(comm)

    assert np.allclose(count, comm.size - 1)
    assert np.allclose(sums, 2 * comm.size - 1)


def run_sums_sparse(comm):
    # as above but using a sparse sum - expect
    # zeros for unhit pixels
    if comm is None:
        return
    s = ParallelSum(5, sparse=True)

    s.add_data(0, [1.0])
    s.add_data(1, [2.0])
    s.add_data(2, [3.0])

    count, sums = s.collect(comm)

    assert count[0] == comm.size
    assert count[1] == comm.size
    assert count[2] == comm.size
    assert count[3] == 0
    assert sums[0] == comm.size
    assert sums[1] == 2.0 * comm.size
    assert sums[2] == 3.0 * comm.size
    assert sums[3] == 0.0


# run all the tests above in both serial
# and with different numbers of mock processors

def test_stats():
    run_stats(None)
    mock_mpiexec(2, run_stats)


def test_stats_weights():
    run_stats_weights(None)
    mock_mpiexec(2, run_stats)
    mock_mpiexec(3, run_stats)


def test_stats_sparse():
    run_stats_missing(None)
    mock_mpiexec(2, run_stats_sparse)
    mock_mpiexec(3, run_stats_sparse)


def test_missing():
    run_stats_missing(None)
    mock_mpiexec(2, run_stats_missing)
    mock_mpiexec(3, run_stats_missing)


def test_partial():
    run_stats_partial(None)
    mock_mpiexec(2, run_stats_partial)
    mock_mpiexec(3, run_stats_partial)


def test_sums():
    run_sums(None)
    mock_mpiexec(2, run_sums)
    mock_mpiexec(3, run_sums)


def test_sparse_sums():
    run_sums(None)
    mock_mpiexec(2, run_sums_sparse)
    mock_mpiexec(3, run_sums_sparse)


if __name__ == "__main__":
    test_stats()
