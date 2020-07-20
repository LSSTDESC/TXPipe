# coding: utf-8
"""
A collection of statistical tools that may be useful     across TXPipe.

"""
import numpy as np
from .sparse import SparseArray
from .mpi_utils import in_place_reduce

class ParallelHistogram:
    """Make a histogram in parallel.

    Bin edges must be pre-defined and values
    outside them will be ignored.

    The usual life-cycle of this class is to create it,
    repeatedly call add_data on chunks, and then call
    collect to finalize. You can also call the calculate
    method with an iterator to combine these.
    """
    def __init__(self, edges):
        """Create the histogram.
        
        Parameters
        ----------
        edges: sequence
            histogram bin edges
        """
        self.edges = edges
        self.size = len(edges) - 1
        self.counts = np.zeros(self.size)

    def add_data(self, x, weights=None):
        """Add a chunk of data to the histogram.

        Weights can optionally be supplied.
        
        Parameters
        ----------
        x: sequence
            Values to be histogrammed
        weights: sequence, optional
            Weights per value.
        """
        b = np.digitize(x, self.edges) - 1
        if weights is None:
            weights = np.ones(x.size)
        n = self.size
        for b_i, w_i in zip(b, weights):
            if b_i >= 0 and b_i < n:
                self.counts[b_i] += w_i

    def collect(self, comm=None):
        """Finalize and collect together histogram values

        Parameters
        ----------
        comm: MPI comm or None
            The comm, or None for serial

        Returns
        -------
        counts: array
            Total counts/weights per bin
        """
        counts = self.counts.copy()

        if comm is None:
            return counts

        import mpi4py.MPI
        if comm.rank == 0:
            comm.Reduce(mpi4py.MPI.IN_PLACE, counts)
            return counts
        else:
            comm.Reduce(counts, None)
            return None

    def calculate(self, iterator, comm=None):
        """Run the whole life cycle on an iterator returning data chunks.

        This is equivalent to calling add_data repeatedly and then collect.

        Parameters
        ----------
        iterator: iterator
            Iterator yieding values or (values, weights) pairs
        comm: MPI comm or None
            The comm, or None for serial

        Returns
        --------
        counts: array
            Total counts/weights per bin
        """
        for values in iterator:
            self.add_data(*values)
        return self.collect(comm)


class ParallelStatsCalculator:
    """ParallelStatsCalculator is a parallel, on-line calculator for mean
    and variance statistics.  "On-line" means that it does not need
    to read the entire data set at once, and requires only a single
    pass through the data.

    The calculator is designed for maps and similar systems, so 
    assumes that it is calculating statistics in a number of different bins
    (e.g. pixels).

    The usual life-cycle of this class is to create it,
    repeatedly call add_data on chunks, and then call
    collect to finalize. You can also call the calculate
    method with an iterator to combine these.

    If only a few indices in the data are expected to be used, the sparse
    option can be set to change how data is represented and returned to 
    a sparse form which will use less memory and be faster below a certain
    size.

    The algorithm here is basd on Schubert & Gertz 2018,
    Numerically Stable Parallel Computation of (Co-)Variance

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
        weighted: bool, optional
            Whether to expect weights along with the data and produce weighted stats
        """
        self.size = size
        self.sparse = sparse
        self.weighted = weighted
        if sparse:
            import scipy.sparse

            self._mean = SparseArray()
            self._weight = SparseArray()
            self._M2 = SparseArray()

            if self.weighted:
                self._W2 = SparseArray()
        else:
            self._mean = np.zeros(size)
            self._weight = np.zeros(size)
            self._M2 = np.zeros(size)
            if self.weighted:
                self._W2 = np.zeros(size)

    def add_data(self, pixel, values, weights=None):
        """Add a sequence of values associated with one pixel.

        Add a set of values assinged to a given bin or pixel.
        Weights must be supplied only if you set "weighted=True"
        on creation and cannot be otherwise.

        Parameters
        ----------
        pixel: int
            The pixel or bin for these values
        values: sequence
            A sequence (e.g. array or list) of values assigned to this pixel
        weights: sequence, optional
            A sequence (e.g. array or list) of weights per value
        """
        if self.weighted:
            if weights is None:
                raise ValueError("Weights expected in ParallelStatsCalculator")

            for value, w in zip(values, weights):
                if w == 0:
                    continue
                self._weight[pixel] += w
                delta = value - self._mean[pixel]
                self._mean[pixel] += (w / self._weight[pixel]) * delta
                delta2 = value - self._mean[pixel]
                self._M2[pixel] += w * delta * delta2
                self._W2[pixel] += w * w
        else:
            if weights is not None:
                raise ValueError("No weights expected n ParallelStatsCalculator")
            for value in values:
                self._weight[pixel] += 1
                delta = value - self._mean[pixel]
                self._mean[pixel] += delta / self._weight[pixel]
                delta2 = value - self._mean[pixel]
                self._M2[pixel] += delta * delta2

    def collect(self, comm=None, mode="gather"):
        """Finalize the statistics calculation, collecting togther results
        from multiple processes.

        If mode is set to "allgather" then every calling process will return
        the same data.  Otherwise the non-root processes will return None
        for all the values.

        You can only call this once, when you've finished calling add_data.
        After that internal data is deleted.

        Parameters
        ----------
        comm: MPI Communicator or None

        mode: string
            'gather', or 'allgather'

        Returns
        -------
        weight: array or SparseArray
            The total weight or count in each bin
        mean: array or SparseArray
            An array of the computed mean for each bin
        variance: array or SparseArray
            An array of the computed variance for each bin

        """
        # Serial version - just take the values from this processor,
        # set the values where the weight is zero, and return
        if comm is None:
            results = self._weight, self._mean, self._get_variance()
            very_bad = self._weight == 0
            # Deal with pixels that have been hit, but only with
            # zero weight
            if self.sparse:
                for i in very_bad:
                    self._mean[i] = np.nan
            else:
                self._mean[very_bad] = np.nan

            del self._M2
            del self._weight
            del self._mean
            return results

        # Otherwise we do this in parallel.  The general approach is
        # a little crude because the reduction operation here is not
        # that simple (we can't just sum things, because we also need
        # the variance and combining those is slightly more complicated).
        rank = comm.Get_rank()
        size = comm.Get_size()

        if mode not in ["gather", "allgather"]:
            raise ValueError(
                "mode for ParallelStatsCalculator.collect must be"
                "'gather' or 'allgather'"
            )

        # The send command differs depending whether we are sending
        # a sparse object (which is pickled) or an array.
        if self.sparse:
            send = lambda x: comm.send(x, dest=0)
        else:
            send = lambda x: comm.Send(x, dest=0)

        # If we are not the root process we send our results
        # to the root one by one.  Then delete them to save space,
        # since for the mapping case this can get quite large.
        if rank > 0:
            send(self._weight)
            del self._weight
            send(self._mean)
            del self._mean
            send(self._M2)
            del self._M2

            # If we are running allgather and need dense arrays
            # then we make a buffer for them now and will send
            # them below
            if mode == "allgather" and not self.sparse:
                weight = np.empty(self.size)
                mean = np.empty(self.size)
                variance = np.empty(self.size)
            else:
                weight = None
                mean = None
                variance = None
        # Otherwise this is the root node, which accumulates the
        # results
        else:
            # start with our own results
            weight = self._weight
            mean = self._mean
            sq = self._M2
            if not self.sparse:
                # In the sparse case MPI4PY unpickles and creates a new variable.
                # In the dense case we have to pre-allocate it.
                w = np.empty(self.size)
                m = np.empty(self.size)
                s = np.empty(self.size)

            # Now received each processes's data chunk in turn
            # at root.
            for i in range(1, size):
                if self.sparse:
                    w = comm.recv(source=i)
                    m = comm.recv(source=i)
                    s = comm.recv(source=i)
                else:
                    comm.Recv(w, source=i)
                    comm.Recv(m, source=i)
                    comm.Recv(s, source=i)

                # Add this to the overall sample.  This is very similar
                # to what's done in add_data except it combines all the
                # pixels/bins at once.
                weight, mean, sq = self._accumulate(weight, mean, sq, w, m, s)
                print(f"Done rank {i}")

            # get the population variance from the squared deviations
            # and set the mean to nan where we can't estimate it.
            variance = sq / weight

        if mode == "allgather":
            if self.sparse:
                weight, mean, variance = comm.bcast([weight, mean, variance])
            else:
                comm.Bcast(weight)
                comm.Bcast(mean)
                comm.Bcast(variance)

        return weight, mean, variance

    def calculate(self, iterator, comm=None, mode="gather"):
        """Run the whole life cycle on an iterator returning data chunks.

        This is equivalent to calling add_data repeatedly and then collect.

        Parameters
        ----------
        iterator: iterator
            Iterator yieding (pixel, values) or (pixel, values, weights)
        comm: MPI comm or None
            The comm, or None for serial
        mode: str
            "gather" or "allgather"

        Returns
        -------
        weight: array or SparseArray
            The total weight or count in each bin
        mean: array or SparseArray
            An array of the computed mean for each bin
        variance: array or SparseArray
            An array of the computed variance for each bin
        """
        for values in iterator:
            self.add_data(*values)
        return self.collect(comm=comm, mode=mode)

    def _get_variance(self):
        # Compute the variance from the previously
        # computed squared deviations. 
        variance = self._M2 / self._weight
        if not self.sparse:
            if self.weighted:
                neff = self._weight ** 2 / self._W2
                bad = neff < 1.000001
            else:
                bad = self._weight < 2
            variance[bad] = np.nan

        return variance


    def _accumulate(self, weight, mean, sq, w, m, s):
        # Algorithm from Shubert and Gertz.
        weight = weight + w
        delta = m - mean
        mean = mean + (w / weight) * delta
        delta2 = m - mean
        sq = sq + s + w * delta * delta2

        return weight, mean, sq


class ParallelSum:
    """Sum up values in pixels in parallel, on-line.

    See ParallelStatsCalculator for details of the motivation.
    Like that code you can specify sparse if only a few pixels
    will be hit.

    The usual life-cycle of this class is to create it,
    repeatedly call add_data on chunks, and then call
    collect to finalize. You can also call the calculate
    method with an iterator to combine these.

    Unlike that class you cannot yet supply weights here, since
    we have not yet needed that use case.
    """
    def __init__(self, size, sparse=False):
        """Create the calculator

        Parameters
        ----------
        size: int
            The maximum number of bins or pixels
        sparse: bool, optional
            If True, use sparse arrays to minimize memory usage
        """
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
        """Add a chunk of data to the sum.

        Parameters
        ----------
        pixel: int
            Index of bin or pixel these value apply to
        values: sequence
            Values for this pixel to accumulate
        """
        for value in values:
            self._count[pixel] += 1
            self._sum[pixel] += value

    def collect(self, comm=None, mode="gather"):
        """Finalize the sum and return the counts and the sums.

        The "mode" decides whether all processes receive the results
        or just the root.

        Parameters
        ----------
        comm: mpi communicator or None
            If in parallel, supply this
        mode: str, optional
            "gather" or "allgather"

        Returns
        -------
        count: array or SparseArray
            The number of values hitting each pixel
        sum: array or SparseArray
            The total of values hitting each pixel
        """
        if comm is None:
            return self._count, self._sum

        if self.sparse:
            if mode == "allgather":
                self._count = comm.allreduce(self._count)
                self._sum = comm.allreduce(self._count)
            else:
                self._count = comm.reduce(self._count)
                self._sum = comm.reduce(self._count)
        else:
            in_place_reduce(self._count, comm, allreduce=(mode == "allgather"))

        return self._count, self._sum

    def calculate(self, iterator, comm=None, mode="gather"):
        """Run the whole life cycle on an iterator returning data chunks.

        This is equivalent to calling add_data repeatedly and then collect.

        Parameters
        ----------
        iterator: iterator
            Iterator yielding (pixel, values) pairs
        comm: MPI comm or None
            The comm, or None for serial

        Returns
        -------
        count: array or SparseArray
            The number of values hitting each pixel
        sum: array or SparseArray
            The total of values hitting each pixel
        """        
        for values in iterator:
            self.add_data(*values)
        return self.collect(comm=comm, mode=mode)
