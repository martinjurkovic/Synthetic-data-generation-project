#!/bin/bash
#SBATCH --job-name="BENCH"
#SBATCH --output=logs/sling-benchmark-%J.out
#SBATCH --error=logs/sling-benchmark-%J.err
#SBATCH --time=2-00:00:00 # job time limit - full format is D-H:M:S
#SBATCH --nodes=1 # number of nodes
#SBATCH --ntasks=1 # number of tasks
#SBATCH --cpus-per-task=12 # number of allocated cores
#SBATCH --mem=128G

source /d/hpc/projects/FRI/mj5835/miniconda3/etc/profile.d/conda.sh # intialize conda
conda activate realtabformer_env # activate the previously created environment
srun --nodes=1 --exclusive --ntasks=1 python /d/hpc/projects/FRI/mj5835/Synthetic-data-generation-project/src/evaluation_scripts/benchmark.py