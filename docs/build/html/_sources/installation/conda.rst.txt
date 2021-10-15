Conda Installation
==================

The easiest way to install TXPipe on your laptop or desktop is to used `Conda Forge <https://conda-forge.org/>`_.

You can get a complete environment from scratch that can run TXPipe using these commands on most operating systems.  This will give you an isolated environment.

First download one of these Conda Forge Mini installers, depending on your operating system::

.. code-block:: bash

    # For Linux:
    wget -O Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

    # For Mac:
    wget -O Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh

    # For Mac Silicon:
    wget -O Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh


Then install the requirements:

.. code-block:: bash

    # Download and activate the conda forge mini-installer
    chmod +x Miniforge3.sh 
    ./Miniforge3.sh -b -p ./conda
    source ./conda/bin/activate

    # Install main packages with conda
    conda install -c conda-forge scipy matplotlib-base camb healpy psutil numpy scikit-learn fitsio pandas astropy pyccl treecorr namaster dask cosmosis-standalone "h5py=*=*mpich*" mpi4py healsparse qp

    # Install remaining packages with pip
    pip install -r requirements.txt


Then whenever you start a new terminal, enable the system with::

    source ./conda/bin/activate
