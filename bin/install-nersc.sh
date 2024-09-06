#!/usr/bin/env bash

set -e
set -x

VERSION=1.0
ENV_PATH=/global/common/software/lsst/users/zuntz/txpipe/env-${VERSION}

module load python


conda create --yes -p ${ENV_PATH} python=3.10
conda activate ${ENV_PATH}
module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py -vv


# Okay, this is a terrible thing we're doing now.
# If I install the TX environment then it breaks something, somewhere
# in the compilation of mpi4py.  So we are going to copy it out somewhere else,
# let conda overwrite it, and then replace it again
BACKUP_DIR=mpi4py-tmp-${RANDOM}
mkdir ${BACKUP_DIR}
cp -r ${ENV_PATH}/lib/python3.10/site-packages/mpi4py*  ${BACKUP_DIR}/

# Install all the TXPipe dependencies
mamba env update --file bin/environment-perlmutter.yml

# Now we put mpi4py back manually. May God forgive me.
mamba remove --force --yes mpi4py
cp -r ${BACKUP_DIR}/* ${ENV_PATH}/lib/python3.10/site-packages/
rm -rf ${BACKUP_DIR}

# Let the rest of the collaboration read this cursed install
chmod g+rX ${ENV_PATH}
