DESC 3x2pt Pipeline Stages
==========================

This is a collection of modules for the DESC 3x2pt pipeline.
We will build up the modules needed for the analysis as shown in the Pipelines repository.

It builds on the ceci repository for the infrastructure.

Permissions
-----------

Email or Slack Joe Zuntz to be added to the development team.

Goals
-----

- Test using parsl for some of our larger more complex analyses.
- Build and test a prototype pipeline infrastructure.
- Build and run prototype.
- Perform and publish a DC2 3x2pt analysis.


Getting Dependencies
--------------------

TXPipe requires python>=3.6.

**Dependencies on your laptop**

The various stages within it depend on the python packages listed in requirements.txt, and can be install using:
```
pip install -r requirements.txt
```

**NOTE** The current pipeline version needs the *minirunner* branch of *ceci*.  This is installed by requirements.txt

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

If you want to run pipelines under MPI, you can install a minimal environment on cori with just ceci inside (no other dependencies) like this:

```bash
source examples/nersc/setup
python -m venv env
source env/bin/activate
pip install -e git://github.com/LSSTDESC/ceci@minirunner#egg=ceci
```

Then use shifter to run the actual jobs.



Getting the code and some test data
-----------------------------------

Get the TXPipe code like this:
```bash
git clone https://github.com/LSSTDESC/TXPipe
cd TXPipe
```

You can get some input test data like this:

```bash

mkdir -p data/example/inputs
cd data/example/inputs
curl -O https://portal.nersc.gov/project/lsst/WeakLensing/shear_catalog.hdf5
curl -O https://portal.nersc.gov/project/lsst/WeakLensing/photometry_catalog.hdf5
curl -O https://portal.nersc.gov/project/lsst/WeakLensing/sample_cosmodc2_w10year_errors.dat
curl -O https://portal.nersc.gov/project/lsst/WeakLensing/cosmoDC2_trees_i25.3.npy
curl -O https://portal.nersc.gov/project/lsst/WeakLensing/exposures.hdf5
curl -O https://portal.nersc.gov/project/lsst/WeakLensing/star_catalog.hdf5

cd ../../..
```


Running the pipeline
--------------------

Once you have installed the dependecies you can run this using the test data you downloaded above

```bash
ceci examples/laptop.yml
```

to run the implemented stages.

You can get a list of the individual commands that will be run like this:

```bash
ceci --dry-run examples/laptop.yml
```

so that you can run and examine them individually.




Larger runs
-----------

Example larger runs, which can be run on NERSC under interactive jobs (for now) can be run by doing:
```bash
# To get an interactive job:
salloc -N 2  -q interactive -C haswell -t 01:00:00 -A m1727
# <wait for allocation>
ceci examples/2.1.1i.yml
```

A larger run is in `examples/2.1i.yml`.



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
