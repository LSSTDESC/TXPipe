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
wget -O Mambaforge3.sh $URL
chmod +x Mambaforge3.sh
./Mambaforge3.sh -b -p ./conda
source ./conda/bin/activate

# conda-installable stuff
mamba install -c conda-forge -y --file conda.txt
# everything else
pip install -r requirements.txt || ( sed 's/git+https/git+git/' requirements.txt > requirements2.txt && pip install -r requirements2.txt )

echo ""
echo "Installation successful!"
echo "Now you can set up your TXPipe environment using the command:"
echo "source ./conda/bin/activate"
