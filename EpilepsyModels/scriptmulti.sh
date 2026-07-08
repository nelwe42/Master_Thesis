#!/bin/bash
#SBATCH --array=1-2
#SBATCH --partition=intelsr_short
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --chdir=~

julia --threads=2 --project=/home/s6newell_hpc/EpilepsyModels /home/s6newell_hpc/EpilepsyModels/src/MultiData.jl $SLURM_ARRAY_TASK_ID