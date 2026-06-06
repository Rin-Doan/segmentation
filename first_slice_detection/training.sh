#!/bin/bash
#SBATCH --job-name="yolo_first_slice"
#SBATCH --output=gpu_job_yolo.out
#SBATCH --error=gpu_job_yolo.err
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

# uv run prepare_dataset.py
uv run train_yolo.py --epochs 100 --imgsz 512 --batch 16
uv run inference.py --weights yolov8.pth --conf 0.80
