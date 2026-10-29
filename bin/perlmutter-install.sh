#!/usr/bin/env bash

# BASH_SOURCE[0] gives the path to this script.
# dirname on that gives the directory it's in
DESCPYTHONUSERBASE=$(dirname "${BASH_SOURCE[0]}")/../local
mkdir -p $DESCPYTHONUSERBASE



# Now we can pip install using --user anything that is needed as an extra over
# the main env. There's nothing here now but dev branches may need to do so.
# If so, uncomment these:
# export DESCPYTHONUSERBASE=$(realpath ${DESCPYTHONUSERBASE})
# source /global/common/software/lsst/common/miniconda/setup_current_python.sh
# unset PYTHONNOUSERSITE

# pip install --user --no-build-isolation <NewPackage>


echo ""
echo "TXPipe now uses the shared DESC conda environment at NERSC."
echo "Activate it with:"
echo "source ./bin/setup-nersc"
