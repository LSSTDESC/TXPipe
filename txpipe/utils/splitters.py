import numpy as np
from .misc import multi_where


class Splitter:
    """
    Helper class to write out data that is split into bins
    """

    def __init__(self, group, name, columns, bin_sizes, dtypes=None):
        self.bins = list(bin_sizes.keys())
        self.group = group
        self.columns = columns
        self.index = {b: 0 for b in self.bins}
        self.bin_sizes = bin_sizes
        self.subgroups = {b: self.group.create_group(f"{name}_{b}") for b in self.bins}

        for i, b in enumerate(self.bins):
            self.group.attrs[f"bin_{i}"] = b

        self.setup_columns(dtypes or {})

    def setup_columns(self, dtypes):
        for b, sz in self.bin_sizes.items():
            sub = self.subgroups[b]
            for col in self.columns:
                dt = dtypes.get(col, "f8")
                sub.create_dataset(col, (sz,), dtype=dt)

    def write(self, data, bins):

        wheres = multi_where(bins, self.bins)

        for b in self.bins:
            # Get the index of objects that fall in this bin
            w = wheres[b]
            if w.size == 0:
                continue

            # Make the right subsets of the data according to this index
            bin_data = {col: data[col][w] for col in self.columns}

            # Save this chunk of data
            self.write_bin(bin_data, b)

        return wheres

    def write_bin(self, data, b):
        # Length of this chunk
        n = len(data[self.columns[0]])
        # Group where we will write the data
        group = self.subgroups[b]
        # Indices of this output
        s = self.index[b]
        e = s + n

        self.size_check(b, e)

        # Write to columns
        for col in self.columns:
            group[col][s:e] = data[col]

        # Update overall index
        self.index[b] = e

    def size_check(self, b, e):
        n = self.bin_sizes[b]
        if e > n:
            raise ValueError(f"Too much data added bin {b}: got {e}, expected max {n}")

    def finish(self, my_bins):
        for b in my_bins:
            c = self.bin_sizes[b]
            n = self.index[b]
            if c != n:
                raise ValueError(
                    f"Count error in bin {b}: expected {c} but copied in {n}"
                )


class DynamicSplitter(Splitter):
    """
    This version of the splitter dynamically adjusts the sizes of the columns using
    chunking, so you don't need to know the sizes in advance.  The cost is that it
    can't be used in parallel.

    The sizes pased to the data
    """

    def setup_columns(self, dtypes):
        for b, sz in self.bin_sizes.items():
            sub = self.subgroups[b]
            for col in self.columns:
                dt = dtypes.get(col, "f8")
                sub.create_dataset(col, (sz,), dtype=dt, maxshape=(None,))

    def size_check(self, b, e):
        n = self.bin_sizes[b]
        if e > n:
            sub = self.subgroups[b]
            new_size = int(n * 1.5)
            for col in self.columns:
                sub[col].resize(new_size)
            self.bin_sizes[b] = new_size

    def finish(self):
        # resize everything to actual size
        for b, sub in self.subgroups.items():
            sz = self.index[b]
            for col in self.columns:
                sub[col].resize((sz,))
