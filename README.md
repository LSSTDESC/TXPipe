TXPipe
======

TXPipe is the DESC 3x2pt Pipeline implementation. It measures lensing and clustering 2pt quantities in real and Fourier space and performs associated tests.

It is written in Python, and requires version 3.6 or above.


Installing
----------

Download TXPipe like this:

```bash
git clone --recurse-submodules https://github.com/LSSTDESC/TXPipe
```

and then install dependencies for it with conda like this on a laptop/desktop (M1 macs are not yet supported, sorry!):

```bash
cd TXPipe
./bin/install.sh
```

You can now set up your environment using:
```bash
source ./conda/bin/activate
```
which you should run each time you start a new terminal, from the TXPipe directory.

For using TXPipe on NERSC, you can use the Shifter system, which creates a container with all the requirements pre-built. See the ReadtheDocs page (https://txpipe.readthedocs.io/en/latest) under 'Running TXPipe at NERSC.' 

Running
-------

Download some test data:

```bash
curl -O https://portal.nersc.gov/cfs/lsst/txpipe/data/example.tar.gz
tar -zxvf example.tar.gz
```

Then run a pipeline on that data:

```bash
ceci examples/metadetect/pipeline.yml
```

You can find the outputs in the directory `data/example/outputs_metadetect`


Running the example notebook on jupyter at NERSC
-------------------------------------------------

Follow these steps to run the jupyter notebook at nersc.

1. In a terminal on NERSC, run these commands to prepare and download code and data

```bash
# Set up the TXPipe kernel
/global/cfs/cdirs/lsst/groups/WL/users/zuntz/setup-txpipe-jupyter.sh

# Get TXPipe
cd $SCRATCH
git clone --recurse-submodules https://github.com/LSSTDESC/TXPipe
cd TXPipe
# While this is still in a branch, you also have to do this:
git checkout example-notebook

# Get data
curl -O https://portal.nersc.gov/cfs/lsst/txpipe/data/example.tar.gz
tar -zxvf example.tar.gz
```

2. Go to https://jupyter.nersc.gov, log in, and select "Perlmutter shared node"

3. Navigate to this notebook in your cloned directory's `notebooks` subdirectory (via the $PSCRATCH link) and open it.

4. Click the button on the top right of the notebook that usually says something "NERSC Python" and select the "TXPipe" kernel in the pop-up.

5. You can now run the notebook


Learning more
-------------

See the [ReadTheDocs page for much more documentation](https://txpipe.readthedocs.io/en/latest).


Permissions
-----------

Email or Slack Joe Zuntz to be added to the development team.


Continuous Integration
----------------------

Github actions is set up to run unit tests several pipelines whenever commits are pushed to an open pull request.  We need to keep these pipelines up to date: https://github.com/LSSTDESC/TXPipe/actions
