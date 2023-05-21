#!/bin/bash
#SBATCH --job-name="ZUR_SDV"
#SBATCH --output=logs/sling-sdv-zurich-%J.out
#SBATCH --error=logs/sling-sdv-zurich-%J.err
#SBATCH --time=12:00:00 # job time limit - full format is D-H:M:S
#SBATCH --nodes=1 # number of nodes
#SBATCH --ntasks=1 # number of tasks
#SBATCH --cpus-per-task=12 # number of allocated cores

source /d/hpc/projects/FRI/mj5835/miniconda3/etc/profile.d/conda.sh # intialize conda
conda activate realtabformer_env # activate the previously created environment
srun --nodes=1 --exclusive --ntasks=1 python /d/hpc/projects/FRI/mj5835/Synthetic-data-generation-project/src/generation_scripts/generate_sdv.py --dataset-name zurich 