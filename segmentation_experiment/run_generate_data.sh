#!/bin/bash
#SBATCH --job-name="generate_data"
#SBATCH --output=generate_data.out
#SBATCH --error=generate_data.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=3-00:00:00
#SBATCH --qos=batch-short
#SBATCH --mem=100G
#SBATCH --cpus-per-task=40

module purge
module load Anaconda3
source activate 
conda activate segmentation

cd 3D_DiCELoss
uv run generate_segmented_data.py