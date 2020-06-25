import numpy as np

def mpi_reduce_large(data, comm, max_chunk_count=2**30, root=0, op=None, debug=False):
    """Use MPI reduce in-place on an array, even a very large one.

    MPI Reduce is a reduction operation that will, (e.g.) sum arrays
    from different processors on a single process.

    It fails whenever the size of the array is greater than 2**31,
    due to an overflow.  This version detects that case and divides
    the array up into chunks, running reduction on each one separately
    
    This specific call does in-place reduction, so that the root
    process overwrites its own array with the result.
    This minimizes memory usage.

    The default is to do a sum of all the arrays, and to collect
    at process zero, but those can be overridden.

    Parameters
    ----------
    data: array
        can be any shape but must be contiguous

    comm: MPI communictator

    max_chunk_count: int
        Optional, default=2**30.  Max number of items to allow to be sent at once

    root: int
        Optional, default=0.  Rank of process to receive final result

    op: MPI operation
        Optional, default=None. MPI operation, e.g. MPI.PROD, MPI.MAX, etc. Default is to SUM

    debug: bool
        Optional, default=False.  Whether to print out information from each rank

    """
    from mpi4py.MPI import IN_PLACE, SUM

    if not data.flags['C_CONTIGUOUS']:
        raise ValueError("Cannot reduce non-contiguous array")

    # do a sum by default.  Oher operation
    if op is None:
        op = SUM

    # flatten the array.  crucially this does not make a copy,
    # as long as the data is contiguous, which we just checked
    data = data.reshape(data.size)
    size = data.size

    if not data.flags['C_CONTIGUOUS']:
        raise RuntimeError("It seems numpy has changed its semantics and "
                           "has returned a non-contiguous array from reshape. "
                           "You will have to rewrite the mpi_utils code.")

    start = 0
    while start < size:
        end = min(start + max_chunk_count, size)
        if comm.rank == root:
            if debug:
                print(f"{comm.rank} receiving & summing {start} - {end}")
            comm.Reduce(IN_PLACE, data[start:end], root=root)
        else:
            if debug:
                print(f"{comm.rank} sending {start} - {end}")
            comm.Reduce(data[start:end], None, root=root)
        start = end


def in_place_reduce(data, comm):
    import mpi4py.MPI
    if comm.Get_rank() == 0:
        comm.Reduce(mpi4py.MPI.IN_PLACE, data)
    else:
        comm.Reduce(counts, None)


def test_reduce():
    from mpi4py.MPI import COMM_WORLD as comm
    data = np.zeros((100,200)) + comm.rank + 1
    mpi_reduce_large(data, comm, max_chunk_count=4500, debug=True)
    if comm.rank == 0:
        expected = (comm.size * (comm.size + 1)) // 2
        assert np.allclose(data, expected)

if __name__ == '__main__':
    test_reduce()