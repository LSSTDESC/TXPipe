import numpy as np
import os

class MockDaskArray(np.ndarray):
    """
    Mock dask array class that subclasses numpy.ndarray. This is used to
    replace dask arrays with numpy arrays for debugging purposes.

    This is more complex than with other classes because numpy arrays
    cannot have attributes added arbitrarily at runtime.

    See:
    https://numpy.org/doc/stable/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray
    """
    def __new__(cls, input_array, chunksize=None):
        obj = np.asarray(input_array).view(cls)
        obj.chunksize = chunksize
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.chunksize = getattr(obj, 'chunksize', None)

def from_array(arr, chunks=None):
    arr = MockDaskArray(arr)
    arr.chunksize = 1
    return arr

def compute(anything):
    return anything,

def import_dask(actually_numpy=False):
    """
    Import dask, but if actually_numpy is True, or if the TX_DASK_DEBUG
    environment variable is set, return numpy instead.

    This is to help with debugging dask by replacing it temporarily with
    numpy, which is easier to debug.

    Parameters
    ----------
    actually_numpy : bool
        If True, return numpy instead of dask.

    Returns
    -------
    dask : module
        The dask module or numpy module.
    da : module
        The dask.array module or numpy module.

    """
    actually_numpy = actually_numpy or os.environ.get("TX_DASK_DEBUG", "")
    if actually_numpy:
        print("Faking dask with numpy")
        np.from_array = from_array
        np.compute = compute
        return np, np
    else:
        import dask
        import dask.array as da
        return dask, da
