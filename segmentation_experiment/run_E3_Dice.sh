#!/bin/bash
#SBATCH --job-name="E3-Dice"
#SBATCH --output=gpu_job_E3-Dice.out
#SBATCH --error=gpu_job_E3-Dice.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=3-00:00:00
#SBATCH --qos=batch-short
#SBATCH --mem=100G
#SBATCH --cpus-per-task=40
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s222440401@deakin.edu.au

module purge
module load Anaconda3
source activate 
conda activate segmentation

cd E3-Dice
uv run training.py
uv run evaluation.py