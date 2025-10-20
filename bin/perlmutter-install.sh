#!/usr/bin/env bash

set -e

ENV_PATH=./conda

module load python
module load mpich/4.3.0

mamba env  create --yes -f bin/environment-perlmutter.yml -p ${ENV_PATH} python=3.10
mamba activate ./conda

# we manually install firecrown as we have to remove numcosmo to avoid clashes
FIRECROWN_DIR=firecrown-tmp-${RANDOM}
git clone --branch v1.7.5 https://github.com/LSSTDESC/firecrown  ${FIRECROWN_DIR}
cd ${FIRECROWN_DIR}
sed -i '/numcosmo/d' setup.cfg
pip install .
cd ..
rm -rf ${FIRECROWN_DIR}

# Files in the etc/conda/activate.d directory in a conda environment
# are sourced when the environment is activated.
# So we can put other environment variables and modules loads in there
# that TXPipe needs to run.
# The MPI4PY_RC_RECV_MPROBE variable is set to False because the CRAY
# MPI implementation does not support a feature that otherwise mpi4py
# tries to use. The HDF5_USE_FILE_LOCKING variable is set to FALSE because
# the lustre filesystem does not support file locking.
cat >  ./conda/etc/conda/activate.d/activate-txpipe.sh <<EOF
    module load mpich/4.3.0
    export MPI4PY_RC_RECV_MPROBE='False'
    export HDF5_USE_FILE_LOCKING=FALSE
EOF

echo ""
echo "Installation successful!"
echo "Now you can set up your TXPipe environment using the command:"
echo "module load python; conda activate ./conda"
