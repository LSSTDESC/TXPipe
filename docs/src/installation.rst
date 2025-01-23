TXPipe Installation
===================

TXPipe requires python 3.9 or above.  Download the TXPipe code like this:

.. code-block:: bash

    git clone --recurse-submodules https://github.com/LSSTDESC/TXPipe
    cd TXPipe

Then on a personal machine or NERSC, install with:

.. code-block:: bash

    ./bin/install.sh

This will create a conda environment with all the necessary dependencies in the ./conda directory.
It will print out the command to activate the environment, which you can copy and paste into your shell.


See :ref:`CC-IN2P3 Installation` for the French CC-IN2P3 machine instructions.

For other clusters, please open an issue.