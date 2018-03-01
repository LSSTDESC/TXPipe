DESC 3x2pt Pipeline Stages
--------------------------

This is a collection of modules for the DESC 3x2pt pipeline.
We will build up the modules needed for the analysis as shown in the Pipelines repository.

It builds on the pipette repository for the infrastructure.

A few notes:

- Our catalogs will be large. Wherever possible stages should operate on chunks of their input data at once.
- Pipeline stages shouldn't copy existing columns to new data.

Installation
------------

Requires python3, numpy, scipy, pyyaml, fitsio, h5py (which in turn needs HDF5), and parsl.

Needs the pipette DESC library on the python path (which is not quite stable enough to be worth a setup.py yet).


Cori
----

These dependencies are all already prepared on cori - use this environment:

```bash
module swap PrgEnv-intel  PrgEnv-gnu
module unload darshan
module load hdf5-parallel/1.10.1
module load python/3.6-anaconda-4.4
module load cfitsio/3.370-reentrant
source activate /global/projecta/projectdirs/lsst/groups/WL/users/zuntz/env
```


Implementation
--------------

Each pipeline stage is implemented as a python class inheriting from pipette.Pipeline stage.  These subclasses should:

- have a "name" attribute string.
- have class attributes "inputs" and "outputs", each of which is a list of tuple pairs with a string tag and a FileType class.
- (optionally)  define a "config_options" dictionary of options it expects to find in its section of the main config file, with the value as a default for the option or "None" for no default.
- implement a "run" method doing the actual work of the class.


Roadmap
-------

- Write stages using LSS tools (e.g. from HyperSupremeStructure-HSC-LSS) to implement sysmaps.py and random_cats.py, or reorganize them as one stage perhaps.
- Use the LSS NAMaster examples to write a fourier-space 2pt stage.
- Use the WLPipe code from Troxel to write a real-space 2pt stage.
- Wrapper to generate SACC file from either of the two data sets above.
- Write Gaussian-only Covariance calculator as placeholder for real thing.

