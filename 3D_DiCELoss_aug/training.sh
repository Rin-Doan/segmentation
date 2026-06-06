#!/bin/bash
#SBATCH --job-name="training"  
#SBATCH --output=gpu_job_training_and_evaluation.out
#SBATCH --error=gpu_job_training_and_evaluation.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s222440401@deakin.edu.au

module purge
module load Anaconda3
source activate 
conda activate segmentation


uv run training.py
uv run evaluation.py