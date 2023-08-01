Adding new pipeline stages
==========================

You can easily add new stages to TXPipe to calculate new observables or tests, or to modify existing steps.  

Nothing in TXPipe is set in stone, so if you need to reconfigure other parts of the pipeline to make things work then please get in touch on the slack channel #desc-3x2pt to discuss it.

Planning your changes
---------------------

First, plan how your work fits in with existing TXPipe stages. Does it replace an existing stage, or maybe several of them?  Or is it a new addition? 

Consider how your work should be split into different stages - what intermediate computations are there in the pipeline? The decision is often a trade-off between the need for repeated I/O and flexibility and ease of re-running things.

If you are important data from outside the pipeline, consider whether the process that generates that data could be a pipeline stage too, to allow people to re-run it easily when new data is available.

Have a look at flow charts for the existing pipelines, like :ref:`the example pipeline here<The Example Pipeline>`, to help you decide - look at the inputs that are available from existing code, for example.

Where to put your new work
--------------------------

If your new work is part of the core mission of TXPipe - computing clustering and lensing 2pt functions and associated information  - then make new files for your code inside the ``txpipe/`` directory. If it is an extension project, make a new directory in ``txpipe/extensions``.

It's encouraged to put the bulk of your code in an external library and write a fairly simple pipeline stage in TXPipe.

Writing your stage
-------------------

Setting up your class
^^^^^^^^^^^^^^^^^^^^^

The template below is an example of a new pipeline stage.

.. code-block:: python

    from .base_stage import PipelineStage
    from .data_types import HDFFile
    import numpy as np

    class TXYourStageName(PipelineStage):
        name = "TXYourStageName"

        inputs = [
            ("some_input_tag", HDFFile),
            # ...
        ]

        outputs = [
            ("some_input_tag", HDFFile),
            # ...
        ]

        config_options = {
            "name_of_option": "default_value",
            # or for things without a default, specify the type,
            # e.g. int, float, str, [float]
            # the latter is for a list of floats.
            "name_of_required_param": int, 
        }

        def run(self):
            # import any other required modules at the top
            # and then put the rest of your code here

This class will inherit lots of behaviour from its parent PipelineStage class, which tell it how to connect to other pipeline stages, how to run, and the facilities described below.

The name of the class and the attribute ``name`` should be the same, and be descriptive and clear. For core TXPipe modules it should start ``TX``; for extensions you can choose your own prefix.

You need to decide on the inputs and outputs for the file, and give them tags and types. 

* For inputs, search the page on :ref:`current TXPipe files<TXPipe File Tags and Types>`. 

* For each output, you can choose a tag, which will determine the name of the output file, and choose a file type from the various classes in the data\_types page in the stages listing for details.



Using configuration parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you run a stage, a dictionary called ``self.config`` will be created with all the configuration information in it. The dict is populated with this priority:

* Parameters set on the command line (top priority)
* Parameters set in the config file
* Parameter defaults in the class


Reading input data
^^^^^^^^^^^^^^^^^^

You should always use TXPipe to find the paths to input data files. You can also use it to load data from them if you prefer - that's especially helpful when running in parallel, since there are tools for that.

``self.get_input(tag)`` - returns the path to a file. Tag is one of the tags you listed in the ``inputs`` field in your classs.  The method returns a string.

``self.open_input(tag, wrapper=False)`` - returns an open handle to the named input tag. If ``wrapper`` is False then this method will return a low-level object, such as an open python ``file`` object, ``h5py.File``, or ``fitsio.FITS``, for example.  It's usually better to set ``wrapper=True``, in which case you get an instance of the class named in the inputs list. You can always access the underlying file object with ``obj.file``.  See the data_types page in the stages list for the methods these classes have.


``self.iterate_hdf(tag, group_name, cols, chunk_rows)`` and ``self.iterate_fits(tag, hdunum, cols, chunk_rows)`` - use these to make an iterator that you can use in a ``for`` loop to read chunks for data at a time from the chosen file; it yields a tuple of start index, end index, and a data dict. This will also read in parallel (see below) when running under MPI.

.. code-block:: python

    it = self.iterate_hdf("shear_catalog", "shear", ["ra", "dec"], 100_000)
    for start, end, data in it:
        print(f"Read data from {start} - {end}")
        ra = data["ra"]
        dec = data["dec"]
        ...


``self.combined_iterators(self, rows, *inputs)`` - combines several calls to ``iterate_hdf`` together to pull columns from different files or groups. Yields the same tuple as ``iterate_hdf``, with all the data combined into one dict. For example:

.. code-block:: python

    it = self.combined_iterators(100_000,
        "shear_catalog", "shear", ["ra", "dec"],
        "shear_tomography_catalog", "tomography", ["bin"],
    )
    for start, end, data in it:
        print(f"Read data from {start} - {end}")
        ra = data["ra"]
        dec = data["dec"]
        source_bin = data["bin"]
        ...




Writing output data
^^^^^^^^^^^^^^^^^^^
As with input files, you should use parent methods to find paths for and open output files. Unlike with inputs, though, you are strongly encouraged to use ``wrapper=True``, since this also automatically saves a wide range of provenance data in the output file.


``self.get_output(tag)`` - return the path to the file. Not preferred - use the next method instead, as noted above.

``self.open_output(tag, wrapper=False)``- return an open file object or data file instance. It is preferred to set ``wrapper=True`` and use the object returned.


Running and testing your stage
------------------------------

First, in ``txpipe/__init__.py``, import your new stage(s) from your python module(s). This lets TXPipe know about the new modules.

Then you can run a stage and get a list of options with::

    python -m txpipe TXYourStageName --help

It will tell you the options you can specify to set input paths. Output paths are optional, and if left out the outputs will be put in your current directory.


Parallelizing your stage
------------------------

It is relatively easy to parallelize TXPipe stages using either MPI (with `mpi4py <https://rabernat.github.io/research_computing/parallel-programming-with-mpi-for-python.html>`_) or `Dask <https://docs.dask.org/en/latest/>`_.

TXPipe stages are assumed to be parallel by default unless you set ``parallel = False`` in the class (alongside ``inputs``, etc.).  They will use MPI by default; to use ``dask``, set ``dask_parallel = True``.

You can run a stage in parallel on the command line using (on local machines) ``mpirun -n <number-of-processes> python -m txpipe TXYourStageName --mpi ...`` followed by other options.

Parallelizing at NERSC
^^^^^^^^^^^^^^^^^^^^^^

The NERSC computers are particularly for parallel TXPipe because they can store files to make them accessible quickly from multiple processes at the same time.  Run the command ``stripe_large`` on a directory before copying any files in it to enable this; it makes a big difference.


Parallelizing with mpi4py
^^^^^^^^^^^^^^^^^^^^^^^^^

In the MPI model, all the different processes run the same program. Each process, though, has an index number, called the rank, which tells it which processes it is. The different processes then decide which data to process, or what to do, based on their rank.

The rank zero process is usually called the root process and is often in charge of tasks that should only be performed by one process.

All TXPipe stage instances have these three attributes:
* ``self.size`` - the number of processes
* ``self.rank`` - the index of the process, ranging from ``0 .. self.size - 1``
* ``self.comm`` - an ``mpi4py`` communicator object. Methods on this object like ``send``, ``recv``, ``bcast`` can be used to communicate between processes. This will be set to ``None`` if the stage was not run in parallel.

You can use these; don't forget to check ``if self.comm is not None`` in case the stage has been run in serial (non-parallel) mode.

See mpi4py and MPI documentation to learn more about using MPI in TXPipe.  The 
`Parallel Statistics Library <https://parallel-statistics.readthedocs.io/>`_ may also be useful.


Parallel I/O with mpi4py
^^^^^^^^^^^^^^^^^^^^^^^^^

Reading data in parallel is usually straightforward - multiple processes can always open the same file for reading at the same time.

The ``iterate_hdf`` and related methods described above all operate in parallel by default - if you call them from an MPI process then each process will load different chunks of data, and get different sets of ``start`` and ``end`` indices.

Parallel writing is more subtle, and requires more coordination. It is only supported for HDF5 files. You can pass the keyword ``parallel=True`` to the ``open_output`` method to return a file ready for parallel writing. Then every process can write to the file, provided that they don't write to the same part of it (it's often useful to use the same start/end indices here). 

Parallelizing with dask
^^^^^^^^^^^^^^^^^^^^^^^

You can use the `Dask <https://docs.dask.org/en/latest/>`_ library as an alternative to HDF5 by putting ``dask_parallel = True`` at the top of the class. In this model only one process actually runs the code. One more is reserved as a work manager, and then the rest are all workers that tasks are automatically sent out to. You can then use dask's extensive library of numpy-like tools to do calculations.
