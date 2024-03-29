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

export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# URL to download
URL="https://github.com/conda-forge/miniforge/releases/download/23.1.0-3/Mambaforge-23.1.0-3-${OS}-${CHIPSET}.sh"
# Download and run the conda installer Miniforge conda installer
echo "Downloading conda installer from $URL"
wget -O Mambaforge3.sh $URL
chmod +x Mambaforge3.sh
./Mambaforge3.sh -b -p ./conda
source ./conda/bin/activate

# conda-installable stuff
mamba env update   --file environment-nopip.yml 
mamba env update   --file environment-piponly.yml 


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
