#!/bin/bash
#SBATCH --array=1-100
#SBATCH --partition=intelsr_long
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --time=168:00:00
#SBATCH --ntasks=10
#SBATCH --chdir=~
#SBATCH --output=slurm_%a.out
#SBATCH --error=slurm_%a.out

julia --threads=20 --gcthreads=10,1 --project=/home/s6newell_hpc/EpilepsyModels /home/s6newell_hpc/EpilepsyModels/src/MultiData.jl $SLURM_ARRAY_TASK_ID