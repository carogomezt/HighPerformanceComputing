#!/bin/bash

#SBATCH --job-name matmult_MPI
#SBATCH --output matmult_MPI.out
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 1
#sbatch --gres=gpu:1

export CUDA_VISIBLE_DEVICES=0
mpirun matmult
