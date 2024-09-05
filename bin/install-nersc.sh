#!/usr/bin/env bash
VERSION=1.0
ENV_PATH=/global/common/software/lsst/users/zuntz/txpipe/env-${VERSION}

# modules needed
module load python
module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
module load cray-mpich-abi

# create environment
mamba env create -p ${ENV_PATH} -f bin/environment-perlmutter.yml
mamba activate /global/common/software/lsst/users/zuntz/txpipe/env

# The default mpi4py version does not work at NERSC.
# we have to uninstall and re-install it.  This is even though
# we are using the external MPI package, as far as I can tell.
mamba remove --force --yes mpi4py
CC="cc -shared" MPICC="mpicc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py "mpi4py==3.*"

chmod go+rX ${ENV_PATH}
