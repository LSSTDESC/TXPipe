DESC 3x2pt Pipeline Stages
==========================

This is a collection of modules for the DESC 3x2pt pipeline.
We will build up the modules needed for the analysis as shown in the Pipelines repository.

It builds on the ceci repository for the infrastructure.

Permissions
===========

Email or Slack Joe Zuntz to be added to the development team.

Goals
======

- Test using parsl for some of our larger more complex analyses.
- Build and test a prototype pipeline infrastructure.
- Build and run prototype.
- Perform and publish a DC2 3x2pt analysis.

Getting the code and some test data
====================================

Get the TXPipe code like this:
```bash
git clone https://github.com/LSSTDESC/TXPipe
cd TXPipe
```

You can get some input test data like this:

```bash

mkdir -p data/example
cd data/example
curl -O https://portal.nersc.gov/project/lsst/WeakLensing/shear_catalog.hdf
curl -O https://portal.nersc.gov/project/lsst/WeakLensing/photometry_catalog.hdf
curl -O https://portal.nersc.gov/project/lsst/WeakLensing/sample_cosmodc2_w10year_errors.dat
curl -O https://portal.nersc.gov/project/lsst/WeakLensing/cosmoDC2_trees_i25.3.npy
curl -O https://portal.nersc.gov/project/lsst/WeakLensing/exposures.hdf

cd ../..
```


Dependencies
============

TXPipe requires python>=3.6.

The various stages within it depend on the python packages listed in requirements.txt, and can be install using:
```
pip install -r requirements.txt
```

The twopoint_fourier stage also requires NaMaster, which must be manually installed.  For testing, stick to a real-space analysis.

To try a C_ell analysis, it can be installed from here:

https://github.com/LSSTDESC/NaMaster/

although there is a conda recipe to install it.  You will need to install the pymaster python library along with it.


Dependencies using Docker
-------------------------

In Docker, from the main directory, this will get you the newest version of the required dependencies:

```bash
docker pull joezuntz/txpipe
docker run --rm -it -v$PWD:/opt/txpipe joezuntz/txpipe
```

Dependencies on NERSC's Cori
----------------------------

These dependencies are all already prepared on cori - use this environment:

```bash

# On the login nodes:
source /global/projecta/projectdirs/lsst/groups/WL/users/zuntz/setup-cori-nompi

# When submitting jobs:
source /global/projecta/projectdirs/lsst/groups/WL/users/zuntz/setup-cori

export HDF5_USE_FILE_LOCKING=FALSE

```


Running the pipeline
--------------------

Once you have installed the dependecies you can run:

```bash
export DATA=data/example
ceci test/test-real.yml
```

to run the implemented stages.

You can get a list of the individual commands that will be run like this:

```bash
export DATA=data/example
ceci --dry-run test/test-fourier.yml
```

so that you can run and examine them individually.


Using docker (shifter) on NERSC
-------------------------------

You can use docker images on NERSC using the Shifter program.  This is usually much faster to load than using NERSC libraries directly, and also means you can use the same environment on your laptop as on NERSC.

To get a bash environment with the TXPipe dependecies set up, run:

```
shifter --image docker:joezuntz/txpipe bash
```

You can then run individual TXPipe steps.

If you want to run inside a job (interactive or batch) under MPI using srun, you do so *outside* the shifter call, like this:

```
srun -n 32 shifter --image docker:joezuntz/txpipe python3 -m txpipe ...
```

ceci does not yet know how to do this yet, so it must be done manually.


Implementation
--------------

Each pipeline stage is implemented as a python class inheriting from ceci.Pipeline stage.  These subclasses should:

- have a "name" attribute string.
- have class attributes "inputs" and "outputs", each of which is a list of tuple pairs with a string tag and a FileType class.
- define a "config_options" dictionary of options it expects to find in its section of the main config file, with the value as a default for the option or a type if there is no default.
- implement a "run" method doing the actual work of the class.  Your class should then call the methods described below to interact with the pipeline


Some implementation notes:

- Our catalogs will be large. Wherever possible stages should operate on chunks of their input data at once. ceci has some methods for this (see README)
- Pipeline stages shouldn't copy existing columns to new data.
- No ASCII output allowed!
- Python 3.6
- We will do code review
- One file per box (?)

Pipeline Stage Methods
----------------------

The pipeline stages can use these methods to interact with the pipeline:

Basic tools to find the file path:

- self.get_input(tag)
- self.get_output(tag)

Get base class to find and open the file for you:

- self.open_input(tag, **kwargs)
- self.open_output(tag, **kwargs)


Look for a section in a yaml input file tagged "config"
and read it.  If the config_options class variable exists in the class
then it checks those options are set or uses any supplied defaults:

- self.config['name_of_option']

Parallelization tools - MPI attributes:

- self.rank
- self.size
- self.comm

(Parallel) IO tools - reading data in chunks, splitting up 
according to MPI rank:

- self.iterate_fits(tag, hdunum, cols, chunk_rows)
- self.iterate_hdf(tag, group_name, cols, chunk_rows)






Volunteers
----------

- Chihway C & Emily PL - TXTwoPointReal (WLPipe porting & testing)
- David A - TXSysMapMaker
- Anze S - SACC
- Tim E - TXCov
- Alex M - TXSourceSummarizer
- Antonino T & David A - TXTwoPointPower
