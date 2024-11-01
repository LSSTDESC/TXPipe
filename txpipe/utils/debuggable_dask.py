import numpy as np
import os

def from_array(arr, block_size):
    arr = np.array(arr)
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
