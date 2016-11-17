#!/bin/bash

#SBATCH --job-name TestGPU
#SBATCH --output TestGPU.out
#SBATCH --nodes 6
#SBATCH --ntasks-per-node 1
#sbatch --gres=gpu:1

export CUDA_VISIBLE_DEVICES=0
mpirun TestGPU
