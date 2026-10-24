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


# We only install mamba if it is not present
if command -v mamba &> /dev/null
then
    echo "Using pre-installed mamba to create environment"
    MAMBA=mamba
else
    echo "mamba is not installed, installing mamba"
    # We install a temporary copy of mamba outside the ./conda directory
    # and use it to create the real environment. We must not install into
    # ./conda and then update it in place, because that replaces mamba itself
    # while it is running, which breaks package post-link scripts.
    BOOTSTRAP_DIR=$(mktemp -d)
    URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    echo "Downloading mamba installer from $URL"
    curl -L -o "$BOOTSTRAP_DIR/miniforge.sh" "$URL"
    bash "$BOOTSTRAP_DIR/miniforge.sh" -b -p "$BOOTSTRAP_DIR/mamba"
    MAMBA="$BOOTSTRAP_DIR/mamba/bin/mamba"
fi

# We include conda itself so that ./conda/bin/activate exists.
$MAMBA create -p ./conda -y --file bin/environment-local.yml conda

mkdir -p ./conda/etc/conda/activate.d


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
