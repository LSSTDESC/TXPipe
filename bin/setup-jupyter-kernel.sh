#!/bin/bash


kernel_name=$1
kernel_friendly_name=$2

# check if kernel name and friendly name are provided
if [ -z "${kernel_name}" ] || [ -z "${kernel_friendly_name}" ] ;
then
    echo "Usage: $0 <kernel_name> <kernel_friendly_name>"
    exit 1
fi

# Check that kernel_name is all lower case letters
if ! [[ "${kernel_name}" =~ ^[a-z0-9_]+$ ]] ; then
    echo "Error: kernel_name must contain only lower case letters, numbers, and underscores."
    exit 1
fi

# Load modules and activate conda environment
module load python
conda activate ./conda

# if conda env not found, exit
if [ $? -ne 0 ] ; then
    echo "Conda environment not found. Please set up the conda environment first."
    exit 1
fi

# Create the Jupyter kernel
python -m ipykernel install --user --name=${kernel_name} --display-name="${kernel_friendly_name}"

# Define file paths into kernel
kernel_filename=${HOME}/.local/share/jupyter/kernels/${kernel_name}/kernel.json
helper_filename=${HOME}/.local/share/jupyter/kernels/${kernel_name}/launch_txpipe.sh

# Set up the kernel.json to use our helper script
cat >  ${kernel_filename} <<EOF
{
 "argv": [
  "${helper_filename}",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "${kernel_friendly_name}",
 "language": "python",
 "metadata": {
  "debugger": true
 }
}
EOF


# Create the helper script that loads modules and activates conda
cat >  ${helper_filename} <<EOF
#!/bin/bash

module load python
conda activate "${CONDA_PREFIX}"
python -m ipykernel \$@
EOF

chmod u+x ${helper_filename}
