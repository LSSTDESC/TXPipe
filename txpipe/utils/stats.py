#coding: utf-8
"""
A collection of statistical tools that may be useful     across TXPipe.

"""
import numpy as np

class SparseArray:
    """
    A sparse 1D array class.

    This is incomplete, and is currently designed only to support the
    use case below in the ParallelStats calculator.

    Attributes
    ----------
    d : dict
        The dictionary of set indices (keys) and values

    Methods
    -------
    to_dense()
        Make a dense array.
    to_arrays()
        Return index and value arrays
    """
    def __init__(self, size=None, dtype=np.float64):
        """Create a sparse array.
        No size is needed, just an optional dtype, which defaults to float64.

        Parameters
        ----------
        size: int, optional
            The maximum size of the array.  
            Used only on conversion to dense array, and when checking inputs.
            It can be left as None to have no maximum size, in which case dense
            array output will just use whatever the maximum set index was.
        dtype: data-type, optional
        
        Returns
        -------
        SparseArray
        """
        self.d={}
        self.size=size
        self.dtype=dtype

    def count_nonzero(self):
        return len(self.d)

    def __setitem__(self, index, value):
        if self.size is not None and index>=self.size:
            raise IndexError("")
        self.d[index] = self.dtype(value)

    def _set_direct(self, index, value):
        # Like __setitem__ but bypassing the checks
        # and type conversion for speed.
        self.d[index] = value

    def __getitem__(self, index):
        return self.d.get(index, 0.0)

    def __mul__(self, other):
        x = SparseArray()
        for k,v in self.d.items():
            x[k] = v * other[k]
        return x

    def __truediv__(self, other):
        x = SparseArray()
        for k,v in self.d.items():
            x[k] = v / other[k]
        return x

    def __add__(self, other):
        keys = set()
        keys.update(self.d.keys(), other.d.keys())
        x = SparseArray()
        for k in keys:
            x[k] = self[k] + other[k]
        return x

    def __iadd__(self, other):
        for k,v in other.d.items():
            self[k] += v
        return self

    def __sub__(self, other):
        keys = set()
        keys.update(self.d.keys(), other.d.keys())
        x = SparseArray()
        for k in keys:
            x[k] = self[k] - other[k]
        return x

    def __pow__(self, y):
        x = SparseArray()
        for k in self.d.keys():
            x[k] = self[k]**y
        return x


    def to_dense(self):
        """
        Make a dense version of the array, just as a plain numpy array.
        Un-set values will be zero.

        Returns
        -------
        dense: array
            Dense version of array
        """
        if self.size is None:
            size = max(self.d.keys()) + 1
        else:
            size = self.size
        dense = np.zeros(size)
        for k,v in self.d.items():
            dense[k] = v
        return dense

    @classmethod
    def from_dense(cls, dense):
        """
        Convert a standard (dense) 1D array into a sparse array,
        elements with value zero will not be set in the new array.

        Parameters
        ----------
        dense: array
            1D numpy array to convert to sparse form

        Returns
        -------
        sparse: SparseArray
            
            
        """        
        dense = np.atleast_1d(dense)
        if dense.ndim>1:
            raise ValueError("Only 1D arrays can be made sparse")

        sparse = cls(size=dense.size, dtype=dense.dtype)

        for k,v in enumerate(dense):
            # bypasses the checks in __setitem__
            if v!=0:
                sparse._set_direct(k,v)
        return sparse

    def to_arrays(self):
        """
        Return the indices (keys) and values of elements that have been set.

        Returns
        -------
        indices: array
            indices of elements that have been set.
        values: array
            values of elements that have been set.
            
            
        """
        indices = np.fromiter(self.d.keys(), dtype=np.int64)
        values = np.fromiter(self.d.values(), dtype=self.dtype)
        order = indices.argsort()
        indices = indices[order]
        values = values[order]
        return indices, values




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
    def __init__(self, size, sparse=False):
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
        if sparse:
            import scipy.sparse
            self._mean = SparseArray()
            self._count = SparseArray()
            self._M2 = SparseArray()
        else:
            self._mean = np.zeros(size)
            self._count = np.zeros(size)
            self._M2 = np.zeros(size)


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


    def add_data(self, pixel, values):
        """Designed for manual use - in general prefer the calculate method.

        Add a set of values assinged to a given bin or pixel.

        Parameters
        ----------
        pixel: int
            The pixel or bin for these values
        values: sequence
            A sequence (e.g. array or list) of values assigned to this pixel
        """
        for value in values:
            self._count[pixel] += 1
            delta = value - self._mean[pixel]
            self._mean[pixel] += delta/self._count[pixel]
            delta2 = value - self._mean[pixel]
            self._M2[pixel] += delta * delta2

    def _finalize(self):
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
        self._variance = self._M2/ self._count
        if not self.sparse:
            bad = self._count<2
            self._variance[bad] = np.nan
        del self._M2
        return self._count, self._mean, self._variance        


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
        self._finalize()

        if comm is None:
            return self._count, self._mean, self._variance
        
        rank = comm.Get_rank()
        size = comm.Get_size()

        if mode not in ['gather', 'allgather']:
            raise ValueError("mode for ParallelStatsCalculator.collect must be"
                             "'gather' or 'allgather'")


        if self.sparse:
            send = lambda x: comm.send(x, dest=0)
        else:
            send = lambda x: comm.Send(x, dest=0)

        if rank==0:            
            self._S0 = self._variance * self._count
            self._T0 = self._mean * self._count
            self._C0 = self._count
            if not self.sparse:
                # Buffers for the pieces from the other
                # processors
                c1 = np.empty(self.size)
                m1 = np.empty(self.size)
                v1 = np.empty(self.size)

        for i in range(1, size):
            if rank==i:
                send(self._count)
                send(self._mean)
                send(self._variance)
                del self._count
                del self._mean
                del self._variance
            elif rank==0:
                if self.sparse:
                    c1 = comm.recv(source=i)
                    m1 = comm.recv(source=i)
                    v1 = comm.recv(source=i)
                else:
                    comm.Recv(c1, source=i)
                    comm.Recv(m1, source=i)
                    comm.Recv(v1, source=i)
                self._accumulate(c1, m1, v1)
                print(f"Done rank {i}")
        if rank == 0:
            count, mean, variance = self._C0, self._T0/self._C0, self._S0/self._C0
            del self._S0, self._C0, self._T0
            if not self.sparse:
                mean[count<1] = np.nan
                variance[count<2] = np.nan
        else:
            count, mean, variance = None, None, None

        if mode == 'allgather':
            if self.sparse:
                count, mean, variance = comm.bcast([count, mean, variance])
            else:
                if rank != 0:
                    count = np.zeros(self.size)
                    mean = np.zeros(self.size)
                    variance = np.zeros(self.size)
                comm.Bcast(count)
                comm.Bcast(mean)
                comm.Bcast(variance)

        return count, mean, variance

    def _accumulate(self, c1, m1, v1):
        if not self.sparse:
            m1[c1<1] = 0
            v1[c1<2] = 0

        Cold = self._C0
        Told = self._T0

        Tnext = m1*c1
        C = Cold + c1
        T = Told + Tnext

        if self.sparse:
            self._S0 = self._S0 + v1*c1 + Cold / (Cold*C) * (Told*c1/Cold - Tnext)**2
        else:
            w = np.where(c1>0)
            self._S0[w] = self._S0[w] + v1[w]*c1[w] \
                 + Cold[w] / (Cold[w]*C[w]) * (Told[w]*c1[w]/Cold[w] - Tnext[w])**2

            w = np.where(Cold==0)
            self._S0[w] = v1[w]*c1[w]
        self._C0 = C
        self._T0 = T



    def _calculate_serial(self, values_iterator):
        for pixel, values in values_iterator:
            self.add_data(pixel, values)

        return self._finalize()


    def _calculate_parallel(self, parallel_values_iterator, comm, mode):
        # Each processor calculates the values for its bits of data
        self._calculate_serial(parallel_values_iterator)
        return self.collect(comm, mode=mode)


def combine_variances(counts, means, variances, sparse=False):
# eq 3.1b of Chan, Golub, & LeVeque 1979
    S = variances[0] * counts[0]
    T = means[0] * counts[0]
    C = counts[0]
    N = len(counts)
    
    for i in range(1,N):
        Told = T

        Tnext = means[i]*counts[i]
        Cold = C
        C = Cold + counts[i]
        T = Told + Tnext

        if sparse:
            S = S + variances[i]*counts[i] \
            + Cold / (Cold*C) * (Told*counts[i]/Cold - Tnext)**2
        else:
            w = np.where(counts[i]>0)
            S[w] = S[w] + variances[i][w]*counts[i][w] \
            + Cold[w] / (Cold[w]*C[w]) * (Told[w]*counts[i][w]/Cold[w] - Tnext[w])**2

            w = np.where(Cold==0)
            S[w] = variances[i][w]*counts[i][w]


    mu = T / C
    sigma2 = S / C

    return C, mu, sigma2

