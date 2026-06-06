#!/bin/bash
#SBATCH --job-name="E2-3D"
#SBATCH --output=gpu_job_E2-3D.out
#SBATCH --error=gpu_job_E2-3D.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=100G
#SBATCH --qos=batch-short
#SBATCH --cpus-per-task=40
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s222440401@deakin.edu.au

module purge
module load Anaconda3
source activate 
conda activate segmentation

cd E2-3D
uv run training.py
uv run evaluation.py