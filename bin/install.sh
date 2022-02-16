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

# URL to download
URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-${OS}-${CHIPSET}.sh"

# Download and run the conda installer Miniforge conda installer
echo "Downloading conda installer from $URL"
wget -O Miniforge3.sh $URL
chmod +x Miniforge3.sh
./Miniforge3.sh -b -p ./conda
source ./conda/bin/activate

# Activate conda env


# Install requirements
conda install -c conda-forge -y scipy matplotlib camb healpy psutil numpy scikit-learn fitsio pandas astropy pyccl mpi4py treecorr namaster  dask mpich 'h5py=*=mpi_mpich_*'
pip install threadpoolctl ceci sacc parallel_statistics git+git://github.com/LSSTDESC/gcr-catalogs#egg=GCRCatalogs  git+git://github.com/LSSTDESC/qp git+git://github.com/LSSTDESC/desc_bpz healsparse flexcode  xgboost==1.1.1  git+https://github.com/dask/dask-mpi cosmosis-standalone git+https://github.com/LSSTDESC/firecrown@v0.4 git+git://github.com/LSSTDESC/desc_bpz git+git://github.com/LSSTDESC/qp

echo ""
echo "Installation successful!"
echo "Now you can set up your TXPipe environment using the command:"
echo "source ./conda/bin/activate"
