import h5py
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
        'f8': h5py.h5t.NATIVE_DOUBLE,
        'f4': h5py.h5t.NATIVE_FLOAT,
        'i4': h5py.h5t.NATIVE_INT32,
        'i8': h5py.h5t.NATIVE_INT64,
    }[dtype]

    datasetid = h5py.h5d.create(group.id, name.encode('ascii'), dtype, space_id, plist)
    data_set = h5py.Dataset(datasetid)

    return data_set
