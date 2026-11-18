#!/usr/bin/env bash

# BASH_SOURCE[0] gives the path to this script.
# dirname on that gives the directory it's in
DESCPYTHONUSERBASE=$(dirname "${BASH_SOURCE[0]}")/../local
mkdir -p $DESCPYTHONUSERBASE



# Now we can pip install using --user anything that is needed as an extra over
# the main env. There's nothing here now but dev branches may need to do so.
# If so, uncomment these:
export DESCPYTHONUSERBASE=$(realpath ${DESCPYTHONUSERBASE})
source /global/common/software/lsst/common/miniconda/setup_current_python.sh
unset PYTHONNOUSERSITE

python -m pip install --user --no-build-isolation  \
    git+https://github.com/LSSTDESC/rail_sompz@u/jzuntz/iterate-tomobin-nz-onesom \
    git+https://github.com/ashaeres/pyfsb@86989ab125b843628156868eaadc6e1816ff82cb \
    git+https://github.com/LSSTDESC/WLMassMap@c9180673c3084cb948eac730531165be35c99d5d


echo ""
echo "TXPipe now uses the shared DESC conda environment at NERSC."
echo "Activate it with:"
echo "source ./bin/setup-nersc"
