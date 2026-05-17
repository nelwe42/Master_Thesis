#!/bin/bash
#SBATCH --partition=intelsr_short
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --time=5:00:00
#SBATCH --ntasks=20
#SBATCH --chdir=~

julia --project=/home/s6newell_hpc/EpilepsyModels /home/s6newell_hpc/EpilepsyModels/src/Testing.jl