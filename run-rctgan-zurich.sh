#!/bin/bash
#SBATCH --job-name=RCT_ZUR
#SBATCH --output=logs/sling-rctgan-zurich-%J.out
#SBATCH --error=logs/sling-rctgan-zurich-%J.err
#SBATCH --time=2-00:00:00 # job time limit - full format is D-H:M:S
#SBATCH --nodes=1 # number of nodes
#SBATCH --gres=gpu:2 # number of gpus
#SBATCH --ntasks=1 # number of tasks
#SBATCH --mem-per-gpu=24G # memory allocation
#SBATCH --partition=gpu # partition to run on nodes that contain gpus
#SBATCH --cpus-per-task=12 # number of allocated cores

source /d/hpc/projects/FRI/mj5835/miniconda3/etc/profile.d/conda.sh # intialize conda
conda activate rctgan_env # activate the previously created environment
srun --nodes=1 --exclusive --gres=gpu:2 --ntasks=1 python /d/hpc/projects/FRI/mj5835/Synthetic-data-generation-project/src/generation_scripts/generate_rctgan.py --dataset-name zurich