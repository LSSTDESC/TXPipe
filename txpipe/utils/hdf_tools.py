import h5py
import subprocess
import shutil
import numpy as np


def repack(filename):
    """
    In-place HDF5 repack operation on file.
    """
    tmp_name = f"{filename}.tmp_325467847"
    subprocess.check_call(f"h5repack {filename} {tmp_name}", shell=True)
    shutil.move(tmp_name, filename)


def create_dataset_early_allocated(group, name, size, dtype):
    """
    Create an HdF5 dataset, allocating the full space for it at the start of the process.
    This can make it faster to write data incrementally from multiple processes.
    The dataset is also not pre-filled, saving more time.

    Parameters
    ----------
    group: h5py.Group
        the parent for the dataset

    name: str
        name for the new dataset

    size:  int
        The size of the new data set (which must be 1D)

    dtype: str
        Data type, One of f4, f8, i4, i8

    """
    # create a data-space object, which describes the dimensions of the dataset
    space_id = h5py.h5s.create_simple((size,))

    # Create and fill a property list describing options
    # which apply to the data set.
    plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    plist.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
    plist.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)

    dtype = {
        "f8": h5py.h5t.NATIVE_DOUBLE,
        "f4": h5py.h5t.NATIVE_FLOAT,
        "i4": h5py.h5t.NATIVE_INT32,
        "i8": h5py.h5t.NATIVE_INT64,
    }[dtype]

    datasetid = h5py.h5d.create(group.id, name.encode("ascii"), dtype, space_id, plist)
    data_set = h5py.Dataset(datasetid)

    return data_set


class BatchWriter:
    """
    This class is designed to batch together writes to
    an HDF5 file to minimize the contention when many
    processes are writing to a file at the same time
    using MPI
    """

    def __init__(self, group, col_dtypes, offset, max_size=1_000_000):
        self.group = group
        self.index = 0
        self.written_index = 0
        self.offset = offset
        self.max_size = max_size
        self.cols = list(col_dtypes.keys())
        self.data = {
            name: np.empty(max_size, dtype=dtype) for name, dtype in col_dtypes.items()
        }

    def write(self, **data):
        n = None
        # check all the lengths are the same
        for name, values in data.items():
            n1 = len(values)
            if (n is not None) and (n1 != n):
                raise ValueError("Different length cols passed to Batchwriter.write")
            n = n1

        if n == 0:
            return

        # range of our output to write
        s = 0
        e = min(n, self.max_size - self.index)
        while e - s > 0:
            d = {name: col[s:e] for name, col in data.items()}
            self._batch_chunk(d, e - s)
            s = e
            e = min(n, s + self.max_size - self.index)

    def _batch_chunk(self, data, n):
        s = self.index
        e = s + n
        for name, out_col in self.data.items():
            col = data[name]
            out_col[s:e] = col[:n]
        self.index = e
        if e == self.max_size:
            self._write()
            self.index = 0

    def _write(self):
        s_in = 0
        e_in = self.index

        # number to write in this block
        n = e_in - s_in
        s_out = self.written_index
        e_out = s_out + n

        for name, col in self.data.items():
            self.group[name][s_out + self.offset : e_out + self.offset] = col[s_in:e_in]
        self.written_index = e_out

    def finish(self):
        self._write()
