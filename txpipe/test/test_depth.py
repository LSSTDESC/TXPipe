from txpipe.utils.stats import *
import numpy as np

def test_stats():
    
    npix = 10
    stats = ParallelStatsCalculator(10)
    map_values = [np.random.uniform(size=20) for pixel in range(npix)]

    def iterator():
        for pixel, values in enumerate(map_values):
            yield pixel, values

    count, mean, var = stats.calculate(iterator())
    simple_count = [len(v) for v in map_values]
    simple_mean = np.mean(map_values,axis=1)
    simple_var = np.var(map_values,axis=1, ddof=0)

    assert np.allclose(count, simple_count)
    assert np.allclose(mean, simple_mean)
    assert np.allclose(var, simple_var)


def test_stats_sparse():
    
    npix = 10000
    stats = ParallelStatsCalculator(10000, sparse=True)
    used_pixels = [1,10,100,1000,5000,6000,7000,8000, 9000, 9500]
    map_values = [np.random.uniform(size=20) for pixel in used_pixels]

    def iterator():
        for pixel, values in zip(used_pixels, map_values):
            yield pixel, values

    count, mean, var = stats.calculate(iterator())
    simple_count = [len(v) for v in map_values]
    simple_mean = np.mean(map_values, axis=1)
    simple_var = np.var(map_values, axis=1, ddof=0)

    for i,p in enumerate(used_pixels):
        assert np.isclose(simple_count[i], count[p])
        assert np.isclose(simple_mean[i], mean[p])
        assert np.isclose(simple_var[i], var[p])


def mpi_test_stats():
    import mpi4py.MPI
    np.random.seed(10)

    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    
    npix = 100
    nsamp_per_pix = 100
    stats = ParallelStatsCalculator(npix, sparse=False)



    if rank==0:
        map_values = [np.random.uniform(size=nsamp_per_pix) 
                      for pixel in range(npix)]
    else:
        map_values = None

    map_values = comm.bcast(map_values)

    def iterator():
        for i in range(npix):
            # Each rank only works with some pixels
            if i % size != rank:
                continue
            yield i, map_values[i]

    count, mean, var = stats.calculate(iterator(), comm)

    if comm.Get_rank()==0:
        simple_count = [nsamp_per_pix for i in range(npix)]
        simple_mean = np.mean(map_values,axis=1)
        simple_var = np.var(map_values,axis=1)
        assert np.allclose(count, simple_count)
        assert np.allclose(mean, simple_mean)
        assert np.allclose(var, simple_var)
    else:
        assert count is None
        assert mean is None
        assert var is None


def mpi_test_stats_gather():
    import mpi4py.MPI

    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    
    npix = 100
    stats = ParallelStatsCalculator(npix, sparse=False)



    if rank==0:
        map_values = [np.random.uniform(size=80) for pixel in range(npix-1)]
        map_values.append(np.array([]))
    else:
        map_values = None

    map_values = comm.bcast(map_values)


    def iterator():
        for pixel, values in enumerate(map_values):
            # Each rank only works with some pixels
            if pixel<3 and size>1:
                if rank==0:
                    yield pixel, values[:40]
                    continue
                elif rank==1:
                    yield pixel, values[40:]
                    continue
                else:
                    continue
            elif pixel % size != rank:
                continue
            yield pixel, values

    count, mean, var = stats.calculate(iterator(), comm, mode='allgather')

    simple_count = [len(v) for v in map_values]
    simple_mean = np.mean(map_values[:-1],axis=1)
    simple_var = np.var(map_values[:-1],axis=1)
    assert np.allclose(count, simple_count)
    assert np.allclose(mean[:-1], simple_mean)
    assert np.allclose(var[:-1], simple_var)
    assert np.isnan(var[-1])
    assert np.isnan(mean[-1])
    assert count[-1]==simple_count[-1]



def mpi_test_stats_sparse():
    import numpy as np
    import mpi4py.MPI

    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    npix = 10000

    used_pixels = [1,10,100,1000,5000,6000,7000,8000, 9000, 9500]
    if rank==0:
        map_values = [np.random.uniform(size=20) for pixel in used_pixels]
    else:
        map_values = None
    map_values = comm.bcast(map_values)

    stats = ParallelStatsCalculator(npix, sparse=True)


    def iterator():
        for pixel, values in zip(used_pixels, map_values):
            # Each rank only works with some pixels
            if pixel % size != rank:
                continue
            yield pixel, values

    count, mean, var = stats.calculate(iterator(), comm)

    if comm.Get_rank()==0:
        simple_count = [len(v) for v in map_values]
        simple_mean = np.mean(map_values,axis=1)
        simple_var = np.var(map_values,axis=1)
        for i,p in enumerate(used_pixels):
            assert np.isclose(simple_count[i], count[p])
            assert np.isclose(simple_mean[i], mean[p])
            assert np.isclose(simple_var[i], var[p])
    else:
        assert count is None
        assert mean is None
        assert var is None


if __name__ == '__main__':
    test_stats()    
    test_stats_sparse()
    mpi_test_stats()
    mpi_test_stats_gather()
    mpi_test_stats_sparse() 
