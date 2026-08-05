#!/bin/bash
#SBATCH --array=1-500
#SBATCH --partition=intelsr_long
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --time=120:00:00
#SBATCH --ntasks=5
#SBATCH --chdir=~
#SBATCH --output=slurm_%a.out
#SBATCH --error=slurm_%a.out

julia --threads=10 --gcthreads=5,1 --project=/home/s6newell_hpc/EpilepsyModels /home/s6newell_hpc/EpilepsyModels/src/MultiData.jl $SLURM_ARRAY_TASK_ID