#!/bin/bash
#SBATCH --partition=intelsr_short
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --time=0:05:00
#SBATCH --ntasks=6
#SBATCH --chdir=~

module load Julia
julia --project=./EpilepsyModels ./EpilepsyModels/src/Testing.jl