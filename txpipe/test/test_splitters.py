from ..utils import Splitter, DynamicSplitter
import numpy as np
import tempfile
import os


def test_fixed_splitter():
    import h5py
    # make some data
    cols = ["x", "y", "z"]
    dtypes = {"z": np.int32}
    name = "subset"


    # make fake data
    nbin = 6
    bins = np.random.randint(0, nbin, 100)
    x = np.random.normal(size=100)
    y = np.random.normal(size=100)
    z = np.random.randint(0, 1000, size=100)

    # get counts
    counts = {b: (bins==b).sum() for b in range(nbin)}


    with tempfile.TemporaryDirectory() as dirname:
        filename = os.path.join(dirname, "tmp.hdf5")

        f = h5py.File(filename, "w")
        g = f.create_group("ggg")

        # run splitter
        splitter = Splitter(g, name, cols, counts, dtypes=dtypes)

        for i in range(10):
            s = i * 10
            e = s + 10
            for b in range(6):
                w = np.where(bins[s:e]==b)
                data = {"x": x[s:e][w], "y":y[s:e][w], "z":z[s:e][w]}
                splitter.write_bin(data, b)
        # does checks internally on size
        splitter.finish()

        assert g.attrs['nbin'] == np.unique(bins).size

        # load data from split file and compare to expected
        for b in bins:
            assert np.allclose(g[f'subset_{b}/x'][:], x[bins==b])
            assert np.allclose(g[f'subset_{b}/y'][:], y[bins==b])
            assert np.allclose(g[f'subset_{b}/z'][:], z[bins==b])
            assert g[f'subset_{b}/x'].dtype == np.float64
            assert g[f'subset_{b}/y'].dtype == np.float64
            assert g[f'subset_{b}/z'].dtype == np.int32

def test_dynamic_splitter():
    import h5py
    # make some data
    cols = ["x", "y", "z"]
    dtypes = {"z": np.int32}
    name = "subset"

    # make fake data
    bins = np.random.randint(0, 6, 100)
    x = np.random.normal(size=100)
    y = np.random.normal(size=100)
    z = np.random.randint(0, 1000, size=100)

    # get counts - this is an initial count size,
    # will be increased
    counts = {b: 5 for b in range(6)}


    with tempfile.TemporaryDirectory() as dirname:
        filename = os.path.join(dirname, "tmp.hdf5")

        f = h5py.File(filename, "w")
        g = f.create_group("ggg")

        splitter = DynamicSplitter(g, name, cols, counts, dtypes=dtypes)

        for i in range(10):
            s = i * 10
            e = s + 10
            for b in range(6):
                w = np.where(bins[s:e]==b)
                data = {"x": x[s:e][w], "y":y[s:e][w], "z":z[s:e][w]}
                splitter.write_bin(data, b)

        # splitter.finish() resizes the bins.
        # Check that the intitial sizes are all large enough; the final
        # sizes are checked in the next set of assertions
        for b in range(6):
            assert splitter.bin_sizes[b] >= np.count_nonzero(bins==b)
        splitter.finish()

        # load data from split file and compare to expected
        assert g.attrs['nbin'] == np.unique(bins).size
        for b in bins:
            assert np.allclose(g[f'subset_{b}/x'][:], x[bins==b])
            assert np.allclose(g[f'subset_{b}/y'][:], y[bins==b])
            assert np.allclose(g[f'subset_{b}/z'][:], z[bins==b])
            assert g[f'subset_{b}/x'].dtype == np.float64
            assert g[f'subset_{b}/y'].dtype == np.float64
            assert g[f'subset_{b}/z'].dtype == np.int32