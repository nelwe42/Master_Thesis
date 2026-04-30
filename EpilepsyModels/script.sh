#!/bin/bash
#SBATCH --partition=intelsr_short
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --time=4:00:00
#SBATCH --ntasks=6
#SBATCH --chdir=~

julia --project=/home/s6newell_hpc/EpilepsyModels /home/s6newell_hpc/EpilepsyModels/src/Testing.jl