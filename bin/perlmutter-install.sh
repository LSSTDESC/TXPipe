#!/usr/bin/env bash

set -e

ENV_PATH=./conda

module load python
module load cray-mpich

mamba env  create --yes -f bin/environment-perlmutter.yml -p ${ENV_PATH}
mamba activate ./conda

# we manually install firecrown as we have to remove numcosmo to avoid clashes
FIRECROWN_DIR=firecrown-tmp-${RANDOM}
git clone --branch v1.7.5 https://github.com/LSSTDESC/firecrown  ${FIRECROWN_DIR}
cd ${FIRECROWN_DIR}
sed -i '/numcosmo/d' setup.cfg
pip install .
cd ..
rm -rf ${FIRECROWN_DIR}

# Now we remove the installed mpi4py version and replace it with our own.
# We also remove libfabric which is installed as a dependency of mpich
# despite us asking for the external version
mamba remove --force mpi4py libfabric libfabric1

# Re-install mpi4py. The first env var allows using flexible versions of MPICH so seemed
# a good idea.
MPI4PY_BUILD_MPIABI=1 MPICC="mpicc -shared" pip install  --no-cache-dir --no-binary=mpi4py mpi4py



# Files in the etc/conda/activate.d directory in a conda environment
# are sourced when the environment is activated.
# So we can put other environment variables and modules loads in there
# that TXPipe needs to run.
# The MPI4PY_RC_RECV_MPROBE variable is set to False because the CRAY
# MPI implementation does not support a feature that otherwise mpi4py
# tries to use. The HDF5_USE_FILE_LOCKING variable is set to FALSE because
# the lustre filesystem does not support file locking.
cat >  ./conda/etc/conda/activate.d/activate-txpipe.sh <<EOF
    module load cray-mpich
    export MPI4PY_RC_RECV_MPROBE='False'
    export HDF5_USE_FILE_LOCKING=FALSE
    export LD_LIBRARY_PATH=${MPICH_DIR}/lib-abi-mpich:${LD_LIBRARY_PATH}
    export MPICH_GPU_SUPPORT_ENABLED=0
EOF

echo ""
echo "Installation successful!"
echo "Now you can set up your TXPipe environment using the command:"
echo "module load python; conda activate ./conda"
