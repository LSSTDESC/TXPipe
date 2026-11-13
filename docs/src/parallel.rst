TXPipe Parallelization
======================

The different stages we run can be parallelized either at the *process* level (using MPI) or the *thread* level (usually using OpenMP).

Each process has its own independent memory space, so processes have to explicitly send each other messages.  Using multiple processes is required to operate on separate nodes, since you can't share memory between nodes.

One process can be split into multiple threads, and the different threads share the same memory.  That means that thread-parallelism is suitable when operating on shared data, but it is easier to make a mistake and have them overwrite each other's data.  Each thread has exactly one process.

In TXPipe, I/O happens at the process level.  You can use the *striping* option at NERSC to make sure different processes can read and write data without competing.  Most TXPipe stages do not have any significant threading, and so there's no point using multiple threads.

In some cases you also have to worry about memory usage, if each process uses lots of memory you may have to split over multiple nodes.

The exact choices of what's most efficient depend on the size of the dataset used.
See examples/cosmodc2/pipeline.yml for some choices for large-ish data sets.

Ingestion Parallelization
-------------------------

GCR does not parallelize very well, so in general it's only worth using a single process with a single thread for ingestion stages.


Selection Parallelization
--------------------------

The selection stages parallelize I/O, so it's worth using lots of processes as long as striping is switched on.  There is no threading.

Photo-Z Parallelization
------------------------

Internally some of the photo-z codes (including FlexCode and I think MLZ) can use threads, so it's worth having a small number of threads per process, e.g. 2-4

Mapping Parallelization
----------------------------

The mapping stages allocate quite a lot of memory, and there is a trade-off between speeding up the initial I/O phase and the final reduction phase (the latter could be speeded up), so there is a limit to how many processes it's worth using.  In general I find using about 8 processes on a node works well.

Two-point Parallelization
-------------------------

Both the real-space and fourier-space two-point and covariance stages can efficiently use threads, so it's worth using the maximum number of them (64 on Cori Haswell).  The stages that do tomography (i.e. not the systematics ones) can also split the pairs over several nodes - it's worth doing so and using all your nodes.
not parallelize very well, so in general it's only worth using a single process with a single thread for ingestion stages.
