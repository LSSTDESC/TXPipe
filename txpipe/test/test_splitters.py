from ..utils import Splitter, DynamicSplitter
import numpy as np
import tempfile
import os


# We can't test this under mockmpi because the
# comm objects gets passed down to the C level
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
        for b in bins:
            assert np.allclose(g[f'subset_{b}/x'][:], x[bins==b])
            assert np.allclose(g[f'subset_{b}/y'][:], y[bins==b])
            assert np.allclose(g[f'subset_{b}/z'][:], z[bins==b])
            assert g[f'subset_{b}/x'].dtype == np.float64
            assert g[f'subset_{b}/y'].dtype == np.float64
            assert g[f'subset_{b}/z'].dtype == np.int32
    # run splitter
    # load data from split file
    # compare to expected

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
        # does checks internally on size
        splitter.finish()

        assert g.attrs['nbin'] == np.unique(bins).size
        for b in bins:
            assert np.allclose(g[f'subset_{b}/x'][:], x[bins==b])
            assert np.allclose(g[f'subset_{b}/y'][:], y[bins==b])
            assert np.allclose(g[f'subset_{b}/z'][:], z[bins==b])
            assert g[f'subset_{b}/x'].dtype == np.float64
            assert g[f'subset_{b}/y'].dtype == np.float64
            assert g[f'subset_{b}/z'].dtype == np.int32