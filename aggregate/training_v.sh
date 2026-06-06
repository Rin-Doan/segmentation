#!/bin/bash
#SBATCH --job-name="training"  
#SBATCH --output=gpu_job_vista3d.out
#SBATCH --error=gpu_job_vista3d.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=1-00:00:00
#SBATCH --qos=batch-short
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s222440401@deakin.edu.au

module purge
module load Anaconda3
source activate 
conda activate segmentation

uv run segmentation_vista.py \
  --model vista3d \
  --pretrained ./monai_bundles/vista3d/models/model.pt \
  --freeze_encoder \
  --target_depth 128 \
  --lr 1e-4 \
  --epochs 120
  --unfreeze_epoch -1 \