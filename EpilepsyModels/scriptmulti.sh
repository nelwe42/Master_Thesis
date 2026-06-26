#!/bin/bash
#SBATCH --partition=intelsr_medium
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --time=24:00:00
#SBATCH --ntasks=22
#SBATCH --chdir=~

julia --project=/home/s6newell_hpc/EpilepsyModels /home/s6newell_hpc/EpilepsyModels/src/MultiData.jl