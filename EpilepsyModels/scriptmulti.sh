#!/bin/bash
#SBATCH --array=1-400
#SBATCH --partition=intelsr_medium
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --time=24:00:00
#SBATCH --ntasks=2
#SBATCH --chdir=~

julia --threads=4 --project=/home/s6newell_hpc/EpilepsyModels /home/s6newell_hpc/EpilepsyModels/src/MultiData.jl $SLURM_ARRAY_TASK_ID