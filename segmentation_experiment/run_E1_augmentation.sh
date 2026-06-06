#!/bin/bash
#SBATCH --job-name="E1-augmentation"
#SBATCH --output=gpu_job_E1_augmentation.out
#SBATCH --error=gpu_job_E1_augmentation.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:2
#SBATCH --time=1-00:00:00
#SBATCH --qos=batch-short
#SBATCH --mem=100G
#SBATCH --cpus-per-task=40
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s222440401@deakin.edu.au

module purge
module load Anaconda3
source activate 
conda activate segmentation

cd E1-augmentation
uv run training.py
uv run evaluation.py