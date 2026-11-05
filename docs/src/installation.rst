TXPipe Installation
===================

TXPipe is developed and tested with Python 3.12.  Download the TXPipe code like this:

.. code-block:: bash

    git clone --recurse-submodules https://github.com/LSSTDESC/TXPipe
    cd TXPipe

Then on a personal machine or NERSC, install with:

.. code-block:: bash

    ./bin/install.sh

This will create a conda environment with all the necessary dependencies in the ./conda directory. It will use ``mamba`` or ``conda`` if you already have one of them, and install ``mamba`` temporarily if not. On NERSC Perlmutter the script automatically runs ``./bin/perlmutter-install.sh``, which uses a slightly different set of packages.

If you already have a ``./conda`` directory the installer will not run again; move or delete it first if you want to re-install.

When it finishes it will print out the command to activate the environment, which you can copy and paste into your shell. You need to activate the environment each time you start a new shell:

.. code-block:: bash

    conda activate ./conda

Check that the installation worked by running the unit tests:

.. code-block:: bash

    pytest txpipe

Some of the unit tests use the example data set described in :ref:`Running an example pipeline`, so download that first if you want to run them all.

The environment files are ``bin/environment-local.yml`` and ``bin/environment-perlmutter.yml``. They include development versions of a few RAIL packages that TXPipe's tomography and photo-z stages currently need.


See :ref:`CC-IN2P3 Installation` for the French CC-IN2P3 machine instructions.

For other clusters, please open an issue.