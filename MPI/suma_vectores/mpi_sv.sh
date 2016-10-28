#!/bin/bash
#
#SBATCH --job-name=mpi_mm
#SBATCH --output=res_mpi_mm.out
#SBATCH --nodes=6
#SBATCH --ntasks=8
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

mpirun mpi_mm
