#!/bin/bash
#SBATCH --partition=intelsr_devel
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --time=0:15:00
#SBATCH --ntasks=6
#SBATCH --chdir=~

module load Julia
julia --project=/home/s6newell_hpc/EpilepsyModels /home/s6newell_hpc/EpilepsyModels/src/Testing.jl