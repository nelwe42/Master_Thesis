#!/bin/bash
#SBATCH --partition=intelsr_short
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --time=8:00:00
#SBATCH --ntasks=10
#SBATCH --chdir=~

julia --project=/home/s6newell_hpc/EpilepsyModels /home/s6newell_hpc/EpilepsyModels/src/Testing.jl