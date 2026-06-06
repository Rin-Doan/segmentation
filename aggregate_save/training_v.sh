#!/bin/bash
#SBATCH --job-name="training"  
#SBATCH --output=gpu_job_vista3d.out
#SBATCH --error=gpu_job_vista3d.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=3-00:00:00
#SBATCH --qos=batch-short
#SBATCH --mem=300G
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s222440401@deakin.edu.au

module purge
module load Anaconda3
source activate 
conda activate segmentation

uv run segmentation_tune.py --model vista3d