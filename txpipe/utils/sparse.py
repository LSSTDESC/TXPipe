import numpy as np

class SparseArray:
    """
    A sparse 1D array class.

    This is incomplete, and is currently designed only to support the
    use case in stats.ParallelStats calculator.

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

    def __eq__(self, val):
        if np.isscalar(val):
            inds = [k for k,v in self.d.items() if v == val]
            return np.array(inds)
        elif isinstance(val, SparseArray):
            if val.keys() != self.keys():
                raise ValueError("Cannot compare two sparse arrays with different hit indices")
            inds = [k for k,v in self.d.items() if v == val[k]]
            return np.array(inds)


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


