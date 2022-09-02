#!/usr/bin/env bash

# Stop on error
set -e

if [ -d "./conda" ]
then
    echo "You already have a directory called ./conda here"
    echo "So it looks like you already ran this installer."
    echo "To re-install, delete or move that directory."
    exit
fi

# Figure out operating system details
OS=$(uname -s)

if [ "$OS" = "Darwin" ]
then
    OS=MacOSX
    CHIPSET=$(uname -m)
else
    CHIPSET=x86_64
fi

if [ "$CHIPSET" = "aarch64" ]
then
    echo "Sorry - TXPipe does not yet install on non-x86 systems like M1 macs"
    exit 1
fi

if [ "$CHIPSET" = "arm64" ]
then
    echo "Sorry - TXPipe does not yet install on non-x86 systems like M1 macs"
    exit 1
fi



# URL to download
URL="https://github.com/conda-forge/miniforge/releases/download/4.11.0-4/Mambaforge-4.11.0-4-${OS}-${CHIPSET}.sh"
# Download and run the conda installer Miniforge conda installer
echo "Downloading conda installer from $URL"
wget -O Miniforge3.sh $URL
chmod +x Miniforge3.sh
./Miniforge3.sh -b -p ./conda
source ./conda/bin/activate

# Activate conda env


# Install requirements
mamba install -c conda-forge -y scipy matplotlib camb healpy psutil numpy scikit-learn fitsio pandas astropy pyccl mpi4py treecorr namaster  dask mpich 'h5py=*=mpi_mpich_*' cosmosis-standalone

# On some systems installing with +git works and on some it's https. If https fails then fall back to git.
PIP_PACKAGES_GIT="threadpoolctl ceci sacc parallel_statistics git+git://github.com/LSSTDESC/gcr-catalogs#egg=GCRCatalogs  git+git://github.com/LSSTDESC/qp git+git://github.com/LSSTDESC/desc_bpz healsparse flexcode  xgboost==1.1.1  git+https://github.com/dask/dask-mpi  git+https://github.com/LSSTDESC/firecrown@v0.4 git+git://github.com/LSSTDESC/desc_bpz git+git://github.com/LSSTDESC/qp"
PIP_PACKAGES_HTTPS="${PIP_PACKAGES_GIT//+git/+https}"

pip install $PIP_PACKAGES_GIT || pip install $PIP_PACKAGES_HTTPS

echo ""
echo "Installation successful!"
echo "Now you can set up your TXPipe environment using the command:"
echo "source ./conda/bin/activate"
