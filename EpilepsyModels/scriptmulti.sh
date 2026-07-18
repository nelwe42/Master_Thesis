#!/bin/bash
#SBATCH --array=1-200
#SBATCH --partition=intelsr_long
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --time=72:00:00
#SBATCH --ntasks=3
#SBATCH --chdir=~

julia --threads=6 --project=/home/s6newell_hpc/EpilepsyModels /home/s6newell_hpc/EpilepsyModels/src/MultiData.jl $SLURM_ARRAY_TASK_ID