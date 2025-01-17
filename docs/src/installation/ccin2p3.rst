:orphan:

CC-IN2P3 Installation
==========================

You can get a conda environment on the CC-IN2P3 machine with the instructions below.

Shared environment
------------------

There is a shared environment installed that you can activate like this::

    source /pbs/throng/lsst/users/jzuntz/txpipe-environments/setup-txpipe


This contains the TXPipe dependencies, but you have to clone TXPipe for yourself.  The master branch should be up-to-date; you might want to work in a new branch though, that should be fine.

You should probably clone somewhere under The PBS space in ``/pbs/thron/lsst`` because there will be some large files.

Don't load the conda module before doing this.

Making your own environment
---------------------------

To make / modify your own version of the environment, run::

    /pbs/throng/lsst/users/jzuntz/txpipe-environments/make-env.sh  </path/to/TXPipe>  <version_suffix>

this will use the two in2p3 files in ``/path/to/TXPipe/bin`` to define the environment. It will make ``./conda-{version_suffix}`` where you can then install stuff.  Then you can set up with::

    source /pbs/throng/lsst/users/jzuntz/txpipe-environments/setup-txpipe  /path/to/conda-{version_suffix}

The PBS file system is very slow, so this will take a while.

Jupyter
-------


You can add a jupyter kernel that uses this environment by running::

    /pbs/throng/lsst/users/jzuntz/txpipe-environments/make-jupyter-kernel.sh  2023-Jul-12 /pbs/throng/lsst/users/jzuntz/txpipe-environments/conda-2023-Jul-12

The first arg is the suffix of the name of the kernel in jupyter hub.

The second is the conda environment path to use, so if you make a new env you can change it.

Notes
-----

* I used conda + external OpenMPI + the OpenMPI module

* Building was extremely slow on the PBS space which is assigned to code, but it got there in the end.

* I use a manual download of Miniforge (a conda installer) because the available one on the system is very old

* I saw repeated network problems when trying to pip install things. You might need to keep trying.

* I had to separate out the conda-installations and pip-installations because otherwise the latter failed; I didn't track down why
