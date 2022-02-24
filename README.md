DESC 3x2pt Pipeline Stages
==========================

This is a collection of modules for the DESC 3x2pt pipeline.
We will build up the modules needed for the analysis as shown in the Pipelines repository.

It builds on the ceci repository for the infrastructure.

Tutorial
--------

[See our tutorial](https://docs.google.com/presentation/d/1haMu1eLBzfYjAlcZ-En7PdBxA4ebnWPwp8D2mCdvZV8/edit?usp=sharing) for information on understanding and developing for TXPipe.


Getting the code and some test data
-----------------------------------

Get the TXPipe code like this:
```bash
git clone --recurse-submodules https://github.com/LSSTDESC/TXPipe
cd TXPipe
```

The flag tells git to also download several other repositories
that we depend on.

You can get some input test data like this:

```bash
curl -O https://portal.nersc.gov/cfs/lsst/txpipe/data/example.tar.gz
tar -zxvf example.tar.gz
```

Updating
--------

If you have a previous installation of TXPipe and have used `git pull` to
update it, you can load the submodules using:

```bash
git submodule update --init
```

Permissions
-----------

Email or Slack Joe Zuntz to be added to the development team.



Getting the code and some test data
-----------------------------------

Get the TXPipe code like this (on NERSC, do this in your scratch space):
```bash
git clone https://github.com/LSSTDESC/TXPipe
cd TXPipe
```

You can get some input test data like this:

```bash
curl -O https://portal.nersc.gov/cfs/lsst/txpipe/data/example.tar.gz
tar -zxvf example.tar.gz
```



Getting Dependencies
--------------------

TXPipe requires python>=3.6.


**Dependencies with Conda**

You can use the commands below to get a conda environment with everything you need.

On a cluster with an existing MPI implementation you should modify `mpich` to `mpich=3.4.*=external_*` or whatever version of MPI you have.

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
chmod +x Miniforge3-Linux-x86_64.sh 
./Miniforge3-Linux-x86_64.sh -b -p ./conda
source ./conda/bin/activate


# conda forge installations
conda install -y scipy matplotlib camb healpy psutil numpy scikit-learn fitsio pandas astropy pyccl mpi4py treecorr namaster  dask mpich 'h5py=*=mpi_mpich_*'

# pip installations
pip install threadpoolctl ceci sacc parallel_statistics git+git://github.com/LSSTDESC/gcr-catalogs#egg=GCRCatalogs  git+git://github.com/LSSTDESC/qp git+git://github.com/LSSTDESC/desc_bpz healsparse flexcode  xgboost==1.1.1  git+https://github.com/dask/dask-mpi cosmosis-standalone git+https://github.com/LSSTDESC/firecrown@v0.4 git+git://github.com/LSSTDESC/desc_bpz git+git://github.com/LSSTDESC/qp
```



**Dependencies with pip**

The various stages within it depend on the python packages listed in requirements.txt, and can be install using:
```
pip install -r requirements.txt
```

The twopoint_fourier stage also requires NaMaster, which must be manually installed.  For testing, stick to a real-space analysis.

To try a C_ell analysis, it can be installed from here:

https://github.com/LSSTDESC/NaMaster/

although there is a conda recipe to install it.  You will need to install the pymaster python library along with it.


**Dependencies using Docker**

In Docker, from the main directory, this will get you the newest version of the required dependencies:

```bash
docker pull joezuntz/txpipe
docker run --rm -it -v$PWD:/opt/txpipe joezuntz/txpipe
```

**Dependencies on NERSC's Cori**

On cori, use Shifter to access dependencies.  On the login nodes, this will get you into the environment:

```bash
shifter --image docker:joezuntz/txpipe bash
```

You can then run individual TXPipe steps.

If you want to run inside a job (interactive or batch) under MPI using srun, you do so *outside* the shifter call, like this:

```bash
srun -n 32 shifter --image docker:joezuntz/txpipe python3 -m txpipe ...
```

If you want to run pipelines under MPI, you can install a minimal environment on cori with just a few dependencies, as shown below.  Run this inside your TXPipe directory:

```bash
source examples/nersc/setup
python -m venv env
source env/bin/activate
pip install ceci numpy scipy parallel_statistics
```

Then use shifter to run the actual jobs.



Running the pipeline
--------------------

Once you have installed the dependecies you can run this using the test data you downloaded above

```bash
ceci examples/metacal/pipeline.yml
```

to run the implemented stages.

You can get a list of the individual commands that will be run like this:

```bash
ceci --dry-run examples/metacal/pipeline.yml
```

so that you can run and examine them individually.



Larger runs
-----------

Example larger runs, which can be run on NERSC under interactive jobs (for now) can be run by doing:
```bash
# To get an interactive job:
salloc -N 2  -q interactive -C haswell -t 01:00:00 -A m1727
# <wait for allocation>
ceci examples/2.2i/pipeline.yml
```

A smaller run is in `examples/2.2i/pipeline_1tract.yml`.

Batch runs
----------

You can launch an example of a batch run (a submitted job that queues so you don't have to wait around), by executing this on cori:

```bash
sbatch examples/skysim/cori-skysim.sub
```


Implementation
--------------

Each pipeline stage is implemented as a python class inheriting from ceci.Pipeline stage.  These subclasses should:

- have a "name" attribute string.
- have class attributes "inputs" and "outputs", each of which is a list of tuple pairs with a string tag and a FileType class.
- define a "config_options" dictionary of options it expects to find in its section of the main config file, with the value as a default for the option or a type if there is no default.
- implement a "run" method doing the actual work of the class.  Your class should then call the methods described below to interact with the pipeline


Some implementation notes:

- Our catalogs will be large. Wherever possible stages should operate on chunks of their input data at once. ceci has some methods for this (see README)
- Pipeline stages shouldn't copy existing columns to new data, in general.
- Structure your output clearly!  No header-less text files!
- Python 3.6+
- We will do code review


Pipeline Stage Methods
----------------------

The pipeline stages can use these methods to interact with the pipeline:

Basic tools to find the file path:

- `self.get_input(tag)`
- `self.get_output(tag)`

Get base class to find and open the file for you:

- `self.open_input(tag, **kwargs)`
- `self.open_output(tag, parallel=True, **kwargs)`


Look for a section in a yaml input file tagged "config"
and read it.  If the config_options class variable exists in the class
then it checks those options are set or uses any supplied defaults:

- `self.config['name_of_option']`

Parallelization tools - MPI attributes:

- `self.rank`
- `self.size`
- `self.comm`

(Parallel) IO tools - reading data in chunks, splitting up 
according to MPI rank:

- `self.iterate_fits(tag, hdunum, cols, chunk_rows)`
- `self.iterate_hdf(tag, group_name, cols, chunk_rows)`




Setting up stages that need the LSST environment
--------------------------------------------------

Some ingestion stages may need the LSST environment.


1. Start the LSST env using `python /global/common/software/lsst/common/miniconda/start-kernel-cli.py desc-stack` (or any new instructions from https://confluence.slac.stanford.edu/display/LSSTDESC/Setting+Up+the+DM+Stack+at+NERSC)

2. Create and start a virtual env based on this (first time only)

```
python -m venv --system-site-packages lsst-env
. lsst-env/bin/activate
```

3. Install ceci in that venv

```
pip install git+https://github.com/LSSTDESC/ceci@v0.2
```

(or a newer version if you're reading this in the future)

Then you're ready.


Continuous Integration
----------------------

Github actions is set up to run pytest and then three pipelines (metacal, metacal + redmagic, and lensfit) whenever commits are pushed.  We need to keep this pipeline up to date, and to add more things to it as they are added: https://github.com/LSSTDESC/TXPipe/actions

Site and launcher options
-------------------------

You now (with ceci 1.0) specify a *launcher*, which is the tool used to run the pipeline, and a *site*, which is where it should be run.

The options are show below

The best-tested launcher is mini, and the best-tested sites are local and cori-interactive.


Launchers specify how to run the pipeline.  The options now are *parsl*, *mini*, and *cwl*:
```yaml

launcher:
    name: mini
    # Seconds between updates. Default as shown.
    interval: 3

# OR

launcher:
    name: parsl

# OR

launcher:
    name: cwl
    # required:
    dir: ./cwl # dir for cwl files
    # command used to launch pipeline. Default as shown. Gets some flags added if left as this.
    launch: cwltool 

```

The site specifies where to run the pipeline.  The options now are *local*, *cori-interactive*, and *cori-batch*

```yaml
site:
    name: local
    # Number of jobs to run at once.  Default as shown.
    max_threads: 4
    # These are available for every site.  The default is not to use them:
    # docker/shifter image
    image: joezuntz/txpipe
    #docker/shifter volume mounting
    volume: ${PWD}:/opt/txpipe 

# OR

site:
    name: cori-interactive
    # Number of jobs to run at once.  Default as shown.
    max_threads: ${SLURM_JOB_NUM_NODES}

# OR

site:
    name: cori
    # These are the defaults:
    mpi_command: srun -un
    cpu_type: haswell
    queue: debug
    max_jobs: 2
    account: m1727
    walltime: 00:30:00
    setup: /global/projecta/projectdirs/lsst/groups/WL/users/zuntz/setup-cori

```

Parallelization Choices
-----------------------

The different stages we run can be parallelized either at the *process* level (using MPI) or the *thread* level (usually using OpenMP).

Each process has its own independent memory space, so processes have to explicitly send each other messages.  Using multiple processes is required to operate on separate nodes, since you can't share memory between nodes.

One process can be split into multiple threads, and the different threads share the same memory.  That means that thread-parallelism is suitable when operating on shared data, but it is easier to make a mistake and have them overwrite each other's data.  Each thread has exactly one process.

In TXPipe, I/O happens at the process level.  You can use the *striping* option at NERSC to make sure different processes can read and write data without competing.  Most TXPipe stages do not have any significant threading, and so there's no point using multiple threads.

In some cases you also have to worry about memory usage, if each process uses lots of memory you may have to split over multiple nodes.

The exact choices of what's most efficient depend on the size of the dataset used.
See examples/cosmodc2_pipeline.yml for some choices for large-ish data sets.

### Ingestion

GCR does not parallelize very well, so in general it's only worth using a single process with a single thread for ingestion stages.


### Selection stages

The selection stages parallelize I/O, so it's worth using lots of processes as long as striping is switched on.  There is no threading.

### Photo-Z

Internally some of the photo-z codes (including FlexCode and I think MLZ) can use threads, so it's worth having a small number of threads per process, e.g. 2-4

### Mapping

The mapping stages allocate quite a lot of memory, and there is a trade-off between speeding up the initial I/O phase and the final reduction phase (the latter could be speeded up), so there is a limit to how many processes it's worth using.  In general I find using about 8 processes on a node works well.

### Two-point

Both the real-space and fourier-space two-point and covariance stages can efficiently use threads, so it's worth using the maximum number of them (64 on Cori Haswell).  The stages that do tomography (i.e. not the systematics ones) can also split the pairs over several nodes - it's worth doing so and using all your nodes.
