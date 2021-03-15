import numpy as np


class Splitter:
    """
    Helper class to split a catalog into bins

    This class is used to automate the case where you want
    to split a catalog in single large columns into different
    subsets, for example tomographic bins.  The splitters handle
    setting up the structure of the new file, and copying data into
    it in the right places.

    There are two classes, for two cases:
    - Splitter, for when you know the sizes of the subsets in advance
    - DynamicSplitter, for when you don't.

    Splitter can be used in parallel by bin; DynamicSplitter cannot, as it
    has to resize the arrays as it goes along, which doesn't work with
    parallel HDF5.

    In each case the lifecycle is:
    1. open the output file and create the group you want
    2. pass this group and setup information to init the splitter
    3. write data to the splitter in chunks
    4. finalize the splitter

    The bins don't have to be non-overlapping.

    """

    def __init__(self, group, name, columns, bin_sizes, dtypes=None):
        """Create a fixed-size splitter

        Parameters
        ----------
        group: h5py.Group
            The group where the output data will be written
        name: str
            The base name of the different bins to write
        columns: list
            The str names of the columns to be split
        bin_sizes: dict
            Maps bin_name (can be anything printable) to final_bin_size (int)
        dtypes: dict or None
            Maps bins to HDF5 data types for the output columns.  Bins default
            to 8 byte floats if not found in this.
        """
        self.bins = list(bin_sizes.keys())
        self.group = group
        self.columns = columns
        self.index = {b: 0 for b in self.bins}
        self.bin_sizes = bin_sizes
        self.subgroups = {b: self.group.create_group(f"{name}_{b}") for b in self.bins}

        self.group.attrs['nbin'] = len(self.bins)
        for i, b in enumerate(self.bins):
            self.group.attrs[f"bin_{i}"] = b

        self._setup_columns(dtypes or {})

    def _setup_columns(self, dtypes):
        # set up the columns with fixed sizes according to the
        # bin and column names.
        # Do this for each bin
        for b, sz in self.bin_sizes.items():
            sub = self.subgroups[b]
            # and for each column
            for col in self.columns:
                dt = dtypes.get(col, "f8")
                sub.create_dataset(col, (sz,), dtype=dt)


    def write_bin(self, data, b):
        """
        Write a single chunk of data to the output, all to the same bin

        The bin value must be one of those specified on init.

        This will work in parallel provided each process only adds data
        to a single bin.

        Parameters
        ----------
        data: dict
            Maps column names specified for output to arrays of those values
            to be split up. All must have the same size.
        b: any
            A single value of the bin to look up.  Must be in self.bins
        """
        # Length of this chunk
        n = len(data[self.columns[0]])
        # Group where we will write the data
        group = self.subgroups[b]
        # Indices of this output
        s = self.index[b]
        e = s + n

        self._size_check(b, e)

        # Write to columns
        for col in self.columns:
            group[col][s:e] = data[col]

        # Update overall index
        self.index[b] = e

    def _size_check(self, b, e):
        n = self.bin_sizes[b]
        if e > n:
            raise ValueError(f"Too much data added bin {b}: got {e}, expected max {n}")

    def finish(self, bins=None):
        """
        Finish up by checking that the right amount of data has been written to the file.

        If running in parallel, only bins used by this process will have the correct sizes
        in this object.  In that case, specify my_bins for the list of bins this process
        should check

        Parameters
        ----------
        bins: list or None
            The list of bins that this process has saved data for.
        """
        if bins is None:
            bins = self.bins

        for b in bins:
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

    The sizes pased to the initialization in this case represent

    See the Splitter docstring for more detail.
    """

    def __init__(self, group, name, columns, bin_sizes, dtypes=None):
        """Create a dynamic splitter.


        Parameters
        ----------
        group: h5py.Group
            The group where the output data will be written
        name: str
            The base name of the different bins to write
        columns: list
            The str names of the columns to be split
        bin_sizes: dict
            Maps bin_name (can be anything printable) to an initial guess
            of the size.  Bins will be expanded as needed above this.
        dtypes: dict or None
            Maps bins to HDF5 data types for the output columns.  Bins default
            to 8 byte floats if not found in this.
        """
        super().__init__(group, name, columns, bin_sizes, dtypes=dtypes)

    def _setup_columns(self, dtypes):
        # same as in the parent class except we make them extensible by
        # setting maxshape
        for b, sz in self.bin_sizes.items():
            sub = self.subgroups[b]
            for col in self.columns:
                dt = dtypes.get(col, "f8")
                sub.create_dataset(col, (sz,), dtype=dt, maxshape=(None,))

    def _size_check(self, b, e):
        n = self.bin_sizes[b]

        # Expand the columns by 50% if needed
        if e > n:
            sub = self.subgroups[b]
            new_size = int(n * 1.5)
            for col in self.columns:
                sub[col].resize(new_size)
            self.bin_sizes[b] = new_size

    def finish(self):
        """
        Finish up by resizing all the bin columns to the correct size, stripping off
        any excess space.  It's important to call this.
        """
        # resize everything to actual size
        for b, sub in self.subgroups.items():
            sz = self.index[b]
            for col in self.columns:
                sub[col].resize((sz,))
