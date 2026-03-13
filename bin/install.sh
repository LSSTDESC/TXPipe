#!/usr/bin/env bash

# Stop on error
set -e
set -x

if [ -d "./conda" ]
then
    echo "You already have a directory called ./conda here"
    echo "So it looks like you already ran this installer."
    echo "To re-install, delete or move that directory."
    exit
fi

# Figure out operating system details

if [ "${NERSC_HOST}" == "perlmutter" ]
then
    echo "Installing on Perlmutter using specialised script"
    ./bin/perlmutter-install.sh
    exit
fi

# check if the conda command is already available

export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True


# We only install conda if it is not present
if command -v mamba &> /dev/null
then
    echo "Using pre-installed mamba to create environment"
    mamba create -p ./conda -y --file bin/environment-local.yml
else
    echo "mamba is not installed, installing mamba"
    # URL to download
    INSTALLER_VERSION=24.3.0-0
    URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    # Download and run the conda installer Miniforge conda installer
    echo "Downloading mamba installer from $URL"
    curl -L -o Mambaforge3.sh $URL
    chmod +x Mambaforge3.sh
    ./Mambaforge3.sh -b -p ./conda
    # Install everything
    source ./conda/bin/activate

    mamba env update --yes --file bin/environment-local.yml
fi


# We do this to get around a bug in the healpy installation
# where it installs its own copy of libomp instead of using
# the shared one.
cat > ./conda/etc/conda/activate.d/libomp_healpy_workaround.sh <<EOF
export KMP_DUPLICATE_LIB_OK=TRUE
EOF

echo ""
echo "Installation successful!"
echo "Now you can set up your TXPipe environment using the command:"
echo "source ./conda/bin/activate"
