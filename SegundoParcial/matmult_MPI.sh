#!/bin/bash

#SBATCH --job-name=matmult_MPI
#SBATCH --output=matmult_MPI.out
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1

export CUDA_VISIBLE_DEVICES=0
mpirun matmult
