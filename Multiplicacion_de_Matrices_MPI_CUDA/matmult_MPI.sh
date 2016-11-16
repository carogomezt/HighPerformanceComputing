#!/bin/bash
#
#SBATCH --job-name=matmult_MPI
#SBATCH --output=res_matmult_MPI.out
#SBATCH --nodes=4
#SBATCH --ntasks=4
#sbatch --gres=gpu:1

export CUDA_VISIBLE_DEVICES=0
mpirun matmult