#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --account=m1727
#SBATCH --constraint=cpu
#SBATCH --job-name=mpi-test
#SBATCH --output=mpi-test.log
#SBATCH --mail-type=END

# You can create this environment as described in the readme.
module load python
conda activate ./conda

export OMP_PROC_BIND=true
export OMP_PLACES=threads

# One node with a minimal world size
srun -u -n 3 -N 1 python -m mpi4py.bench helloworld
srun -u -n 3 -N 1 python -m mpi4py.bench pingpong
srun -u -n 3 -N 1 python -m mpi4py.bench ringtest
srun -u -n 3 -N 1 python -m mpi4py.bench futures

# Larger test on one node
srun -u -n 64 -N 1 python -m mpi4py.bench helloworld
srun -u -n 64 -N 1 python -m mpi4py.bench pingpong
srun -u -n 64 -N 1 python -m mpi4py.bench ringtest
srun -u -n 64 -N 1 python -m mpi4py.bench futures


# Larger test on two nodes
srun -u -n 64 -N 2 python -m mpi4py.bench helloworld
srun -u -n 64 -N 2 python -m mpi4py.bench pingpong
srun -u -n 64 -N 2 python -m mpi4py.bench ringtest
srun -u -n 64 -N 2 python -m mpi4py.bench futures
