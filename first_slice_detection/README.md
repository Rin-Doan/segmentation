# First-Slice Detection with YOLOv8n-Pose

Detect the first CT slice in which a vertebra is visible, using a
YOLOv8n-Pose model trained on vertebra bounding boxes derived from the
project's 3D segmentation masks.

## Data

Paths follow the rest of the project:

```text
DATA_PATH        = ../../../../../vast/s222440401
TRAINING_PATH    = $DATA_PATH/training_images     # DICOM series per study
SEGMENTATION_PATH= $DATA_PATH/segmentations       # .nii / .nii.gz masks
```

DICOM volumes are sorted by `InstanceNumber`; segmentations are read
with the same coordinate correction used in `aggregate/data_process.py`
(`seg[:, ::-1, ::-1].transpose(2, 1, 0)`), so the two volumes share the
same slice axis.

## Pipeline

1. `prepare_dataset.py`
   - Finds overlapping studies (DICOM + segmentation).
   - For each slice, writes an 8-bit PNG (HU window `[-200, 1800]`).
   - For slices with vertebra pixels, writes a YOLO-Pose label:
     bounding box from the left/right/top/bottom-most non-zero pixels
     plus one keypoint at the segmentation centroid.
   - Splits studies 80/20 into train/val (seed 42).
   - Writes `dataset.yaml` to `$DATA_PATH/yolo_first_slice_dataset`.

2. `train_yolo.py`
   - Fine-tunes `yolov8n-pose.pt` on the generated dataset.
   - Copies the best checkpoint to `./yolov8.pth`.

3. `inference.py`
   - Iterates every study in `TRAINING_PATH`.
   - Runs slices in `InstanceNumber` order through the model.
   - Records the first 1-indexed slice whose detection confidence
     is `>= 0.80`.
   - Writes `yolo_inference_results.csv`
     (`StudyInstanceUID,slice_number`); `0` means no detection.

## Usage

```bash
uv run prepare_dataset.py
uv run train_yolo.py --epochs 100 --imgsz 512 --batch 16
uv run inference.py --weights yolov8.pth --conf 0.80
```

or submit everything as a single SLURM job:

```bash
sbatch training.sh
```
