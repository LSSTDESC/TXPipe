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

# if [ "$CHIPSET" = "aarch64" ]
# then
#     echo "Sorry - TXPipe does not yet auto-install on non-x86 systems like M1 macs. You can use the conda install though."
#     exit 1
# fi

# if [ "$CHIPSET" = "arm64" ]
# then
#     echo "Sorry - TXPipe does not yet auto-install on non-x86 systems like M1 macs. You can use the conda install though."
#     exit 1
# fi

export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# URL to download
URL="https://github.com/conda-forge/miniforge/releases/download/4.11.0-4/Mambaforge-4.11.0-4-${OS}-${CHIPSET}.sh"

# Download and run the conda installer Miniforge conda installer
echo "Downloading conda installer from $URL"
wget -O Mambaforge3.sh $URL
chmod +x Mambaforge3.sh
./Mambaforge3.sh -b -p ./conda
source ./conda/bin/activate

# conda-installable stuff
mamba env update -vv --file environment.yml

pip install ceci==1.13 \
    "git+https://github.com/jlvdb/hyperbolic@b88b107a291fa16c2006cf971ce610248d58e94c"  \
    "git+https://github.com/joezuntz/dask-mpi@31e35759cfb03780a12ad6ffe2b82279b0fab442"  \
    "healpix==2023.1.13"  \
    "parallel-statistics==0.13"  \
    "pz-rail[dev]==0.99.1"  \
    "sacc==0.8"  \
    "tjpcov==0.3.0"


if [[ "$CHIPSET" = "arm64" || "$CHIPSET" = "aarch64" ]]
then
    echo "Pymaster cannot be correctly conda- or pip-installed on Apple Silicon yet, so we are skipping it."
    echo "The twopoint fourier and some covariance stage(s) will not work"
else
    mamba install -c conda-forge namaster
fi

echo ""
echo "Installation successful!"
echo "Now you can set up your TXPipe environment using the command:"
echo "source ./conda/bin/activate"
