#!/usr/bin/env bash

set -e
set -x

ENV_PATH=./conda

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
mv  ${ENV_PATH}/lib/python3.10/site-packages/mpi4py*  ${BACKUP_DIR}/

# Install all the TXPipe dependencies
mamba env update --file bin/environment-perlmutter.yml

# Now we put mpi4py back manually. May God forgive me.
mamba remove --force --yes mpi4py
cp -r ${BACKUP_DIR}/* ${ENV_PATH}/lib/python3.10/site-packages/
rm -rf ${BACKUP_DIR}

# we manually install firecrown as we have to remove numcosmo
git clone --branch v1.7.5 https://github.com/LSSTDESC/firecrown 
cd firecrown
sed -i '/numcosmo/d' setup.cfg
pip install .
cd ..
rm -rf firecrown

