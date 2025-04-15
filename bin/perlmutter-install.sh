#!/usr/bin/env bash

set -e

ENV_PATH=./conda

module load python
module load mpich/4.2.2

conda env  create --yes -p ${ENV_PATH} python=3.10


# we manually install firecrown as we have to remove numcosmo to avoid clashes
FIRECROWN_DIR=firecrown-tmp-${RANDOM}
git clone --branch v1.7.5 https://github.com/LSSTDESC/firecrown  ${FIRECROWN_DIR}
cd ${FIRECROWN_DIR}
sed -i '/numcosmo/d' setup.cfg
pip install .
cd ..
rm -rf ${FIRECROWN_DIR}

# Add some things that are set up automatically
cat >  ./conda/etc/conda/activate.d/activate-txpipe.sh <<EOF
module load mpich/4.2.2
export MPI4PY_RC_RECV_MPROBE='False'
export HDF5_USE_FILE_LOCKING=FALSE
EOF

echo ""
echo "Installation successful!"
echo "Now you can set up your TXPipe environment using the command:"
echo "module load python; conda activate ./conda"
