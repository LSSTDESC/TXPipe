CC-IN2P3 Installation
==========================

You can get a conda environment on the CC-IN2P3 machine with the instructions below.

Take care when using existing conda environments, especially when they have versions of MPI or MPI-dependent  code (e.g. emcee) included; these often will not work.

Your first-time setup is this::

    # Load CC-IN2P3 modules
    module load gcc
    module load Libraries/hdf5/1.12.1
    module load Compilers/swig/4.0.2
    source /pbs/software/centos-7-x86_64/openmpi/ccenv.sh 4.1.1

    # Download and setup conda forge environment
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    chmod +x Miniforge3-Linux-x86_64.sh 
    ./Miniforge3-Linux-x86_64.sh -b -p ./conda
    source ./conda/bin/activate

    # Install requirements
    conda install -y -c conda-forge scipy matplotlib camb healpy psutil numpy scikit-learn fitsio pandas astropy pyccl treecorr namaster  dask healsparse
    HDF5_MPI=ON CC=mpicc pip install --no-binary=h5py,mpi4py  h5py mpi4py
    pip install -r requirements.txt


Then when you log in in future do this to set up your environment::

    module load gcc
    module load Libraries/hdf5/1.12.1
    module load Compilers/swig/4.0.2
    source /pbs/software/centos-7-x86_64/openmpi/ccenv.sh 4.1.1
    source ./conda/bin/activate
