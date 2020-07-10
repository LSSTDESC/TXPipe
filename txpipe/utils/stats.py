#coding: utf-8
"""
A collection of statistical tools that may be useful     across TXPipe.

"""
import numpy as np
from .sparse import SparseArray

class ParallelHistogram:
    def __init__(self, edges):
        self.edges = edges
        self.size = len(edges) - 1
        self.counts = np.zeros(self.size)

    def add_data(self, x):
        b = np.digitize(x, self.edges) - 1
        n = self.size
        for b_i in b:
            if b_i>=0 and b_i < n:
                self.counts[b_i] += 1

    def collect(self, comm=None):
        counts = self.counts.copy()

        if comm is None:
            return counts

        if comm.rank == 0:
            comm.Reduce(mpi4py.MPI.IN_PLACE, counts)
            return counts
        else:
            comm.Reduce(counts, None)
            return None

class ParallelStatsCalculator:
    """ParallelStatsCalculator is a parallel, on-line calculator for mean
    and variance statistics.  "On-line" means that it does not need
    to read the entire data set at once, and requires only a single
    pass through the data.

    The calculator is designed for maps and similar systems, so 
    assumes that it is calculating statistics in a number of different bins
    (e.g. pixels).

    If only a few indices in the data are expected to be used, the sparse
    option can be set to change how data is represented and returned to 
    a sparse form which will use less memory and be faster below a certain
    size.

    You can either just use the calculate method with an iterator, or
    get finer grained manual usage with other methods.


    Attributes
    ----------
    size: int
        number of pixels or bins
    sparse: bool
        whether to use sparse representations of arrays
    
    Methods
    -------

    calculate(values_iterator, comm=None)
        Main public method - run through iterator returning pixel, [values] and calculate stats
    add_data(pixel, values)
        For manual usage - add another set of values for the given pixel
    finalize()
        For manual usage - after all data is passed in, return counts, means, variances
    collect(counts, means, variances)
        For manual usage - combine sequences of the statistics from different processors

    """
    def __init__(self, size, sparse=False, weighted=False):
        """Create a statistics calcuator.
        
        Parameters
        ----------
        size: int
            The number of bins (or pixels) in which statistics will be calculated
        sparse: bool, optional
            Whether to use a sparse representation of the arrays, internally and returned.
        """
        self.size = size
        self.sparse = sparse
        self.weighted = weighted
        if sparse:
            import scipy.sparse
            self._mean = SparseArray()
            self._count = SparseArray()
            self._M2 = SparseArray()
            if self.weighted:
                self._W2 = SparseArray()
        else:
            self._mean = np.zeros(size)
            self._count = np.zeros(size)
            self._M2 = np.zeros(size)
            if self.weighted:
                self._W2 = np.zeros(size)

    def calculate(self, values_iterator, comm=None, mode='gather'):
        """ Calculate statistics of an input data set.

        Operates on an iterator, which is expected to repeatedly yield
        (pixel, values) pairs to accumulate.

        Parameters
        ----------
        values_iterator: iterable
            Iterator yielding (bin, values) through all required data
        comm: MPI Communicator, optional
            If set, assume each MPI process in the comm is getting different data and combine them at the end.
            Only the master process will return the full results - the others will get None
        mode: string
            'gather', or 'allgather', only used if MPI is used

        Returns
        -------
        count: array or SparseArray
            The number of values in each bin
        mean: array or SparseArray
            An array of the computed mean for each bin
        variance: array or SparseArray
            An array of the computed variance for each bin

            


        """

        with np.errstate(divide='ignore',invalid='ignore'):
            if comm is None:
                count, mean, variance = self._calculate_serial(values_iterator)
                return count, mean, variance
            else:
                count, mean, variance = self._calculate_parallel(values_iterator, comm, mode)
                return count, mean, variance


    def add_data(self, pixel, values, weights=None):
        """Designed for manual use - in general prefer the calculate method.

        Add a set of values assinged to a given bin or pixel.

        Parameters
        ----------
        pixel: int
            The pixel or bin for these values
        values: sequence
            A sequence (e.g. array or list) of values assigned to this pixel
        """
        if self.weighted:
            if weights is None:
                raise ValueError("Weights expected in ParallelStatsCalculator")

            for value, w in zip(values, weights):
                if w == 0:
                    continue
                self._count[pixel] += w
                delta = value - self._mean[pixel]
                self._mean[pixel] += (w / self._count[pixel]) * delta
                delta2 = value - self._mean[pixel]
                self._M2[pixel] += w * delta * delta2
                self._W2[pixel] += w*w
        else:
            if weights is not None:
                raise ValueError("No weights expected n ParallelStatsCalculator")
            for value in values:
                self._count[pixel] += 1
                delta = value - self._mean[pixel]
                self._mean[pixel] += delta / self._count[pixel]
                delta2 = value - self._mean[pixel]
                self._M2[pixel] += delta * delta2



    def _get_variance(self):
        """Designed for manual use - in general prefer the calculate method.

        Add a set of values assinged to a given bin or pixel.

        Returns
        -------
        count: array or SparseArray
            The number of values in each bin
        mean: array or SparseArray
            An array of the computed mean for each bin
        variance: array or SparseArray
            An array of the computed variance for each bin

        """
        variance = self._M2 / self._count
        if not self.sparse:
            if self.weighted:
                neff = self._count**2 / self._W2
                bad = neff <= 1.000001
            else:
                bad = self._count < 2
            variance[bad] = np.nan

        return variance


    def collect(self, comm, mode='gather'):
        """Designed for manual use - in general prefer the calculate method.
        
        Combine together statistics from different processors into one.

        Parameters
        ----------
        comm: MPI Communicator or None

        mode: string
            'gather', or 'allgather'

        Returns
        -------
        count: array or SparseArray
            The number of values in each bin
        mean: array or SparseArray
            An array of the computed mean for each bin
        variance: array or SparseArray
            An array of the computed variance for each bin

        """
        if comm is None:
            results = self._count, self._mean, self._get_variance()
            self._mean[self._count == 0] = np.nan        
            del self._M2
            return results
        
        rank = comm.Get_rank()
        size = comm.Get_size()

        if mode not in ['gather', 'allgather']:
            raise ValueError("mode for ParallelStatsCalculator.collect must be"
                             "'gather' or 'allgather'")

        if self.sparse:
            send = lambda x: comm.send(x, dest=0)
        else:
            send = lambda x: comm.Send(x, dest=0)


        if rank > 0:
            send(self._count)
            del self._count
            send(self._mean)
            del self._mean
            send(self._M2)
            del self._M2
            if mode == 'allgather' and not self.sparse:
                weight = np.empty(self.size)
                mean = np.empty(self.size)
                variance = np.empty(self.size)
            else:
                weight = None
                mean = None
                variance = None
        else:
            weight = self._count
            mean = self._mean
            sq = self._M2
            if not self.sparse:
                # Buffers for the pieces from the other
                # processors
                w = np.empty(self.size)
                m = np.empty(self.size)
                s = np.empty(self.size)
            for i in range(1, size):
                if self.sparse:
                    w = comm.recv(source=i)
                    m = comm.recv(source=i)
                    s = comm.recv(source=i)
                else:
                    comm.Recv(w, source=i)
                    comm.Recv(m, source=i)
                    comm.Recv(s, source=i)

                weight, mean, sq = self._accumulate(weight, mean, sq, w, m, s)
                print(f"Done rank {i}")

            variance = sq / weight
            mean[weight == 0] = np.nan        

        if mode == 'allgather':
            if self.sparse:
                weight, mean, variance = comm.bcast([weight, mean, variance])
            else:
                comm.Bcast(weight)
                comm.Bcast(mean)
                comm.Bcast(variance)

        return weight, mean, variance

    def _accumulate(self, weight, mean, sq, w, m, s):
        weight = weight + w

        delta = m - mean
        mean = mean + (w / weight) * delta
        delta2 = m - mean
        sq = sq + s + w * delta * delta2

        return weight, mean, sq



    def _calculate_serial(self, values_iterator):
        for pixel, values in values_iterator:
            self.add_data(pixel, values)

        variance = self._get_variance()
        return self._count, self._mean, variance


    def _calculate_parallel(self, parallel_values_iterator, comm, mode):
        # Each processor calculates the values for its bits of data
        for pixel, values in parallel_values_iterator:
            self.add_data(pixel, values)
        return self.collect(comm, mode=mode)


class ParallelSum:
    def __init__(self, size, sparse=False):
        self.size = size
        self.sparse = sparse

        if sparse:
            import scipy.sparse
            self._sum = SparseArray()
            self._count = SparseArray()
        else:
            self._sum = np.zeros(size)
            self._count = np.zeros(size)

    def add_data(self, pixel, values):
        for value in values:
            self._count[pixel] += 1
            self._sum[pixel] += value

    def collect(self, comm, mode='gather'):
        if comm is None:
            return self._count, self._sum

        if self.sparse:
            if mode == 'allgather':
                self._count = comm.allreduce(self._count)
                self._sum = comm.allreduce(self._count)
            else:
                self._count = comm.reduce(self._count)
                self._sum = comm.reduce(self._count)
        else:
            in_place_reduce(self._count, comm, allreduce=(mode == 'allgather'))

        return self._count, self._sum