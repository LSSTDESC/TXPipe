import numpy as np
import collections

class ExtendingArrays:
    """
    This class is a helper for storage for the selection code.

    It operates a bit like a set of lists that you can append to,
    but the data is stored as arrays to make it more memory efficent.
    """
    def __init__(self, n, size_step, dtypes):
        self.dtypes = dtypes
        self.arrays = collections.defaultdict(list)
        self.pos = np.zeros(n, dtype=int)
        self.size_step = size_step
        self.narr = len(dtypes)
        self.counts = np.zeros(n, dtype=int)

    def nbytes(self):
        n = 0
        for l in self.arrays.values():
            for group in l:
                for arr in group:
                    n += arr.nbytes
        return n

    def collect(self, index):
        if index not in self.arrays:
            return [np.array([],dtype=dt) for dt in self.dtypes]
        arrays = self.arrays[index]
        last_count = self.pos[index]
        n = len(self.arrays[index])
        output = []
        for i in range(self.narr):
            arrs = [a[i] for a in arrays]
            # chop the end off the last array
            arrs[-1] = arrs[-1][:last_count]
            output.append(np.concatenate(arrs))
        return output
                    
    def total_counts(self):
        return self.counts.sum()

    def extend(self, index):
        l = self.arrays[index]
        l.append([np.zeros(self.size_step, dtype=dt) for dt in self.dtypes])

    def append(self, index, values):
        l = self.arrays[index]
        if not l:
            self.extend(index)
        c = self.pos[index]
        if c == self.size_step:
            self.extend(index)
            c = 0
        arrs = l[-1]
        for arr, value in zip(arrs, values):
            arr[c] = value
        self.pos[index] = c + 1
        self.counts[index] += 1
