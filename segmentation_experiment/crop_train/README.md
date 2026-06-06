# Cropped YOLO Dataset Segmentation Training

This directory contains code for training a U-Net segmentation model on the cropped YOLO dataset.

## Overview

The cropped dataset was created by:
1. Using YOLO bounding boxes to identify vertebra regions
2. Cropping both CT slices and segmentation masks to focus on the region of interest
3. This reduces background area and mitigates class imbalance

## Dataset Structure

The cropped dataset is located at:
```
../../../../vast/s222440401/cropped_yolo_dataset/
├── train/
│   ├── images/  (PNG files)
│   └── masks/   (NIfTI .nii.gz files)
└── val/
    ├── images/  (PNG files)
    └── masks/   (NIfTI .nii.gz files)
```

## Files

- **`data_process.py`**: Dataset class for loading cropped images and masks
- **`training.py`**: Main training script
- **`evaluation.py`**: Model evaluation script
- **`computational_metrics.py`**: Memory and resource tracking utilities
- **`training_report.py`**: Training efficiency and convergence reporting

## Usage

### Training

```bash
python training.py
```

This will:
- Load cropped images and masks from `cropped_yolo_dataset`
- Train a U-Net model with ResNet34 encoder
- Save the best model as `best_unet_model.pth`
- Generate training reports in `training_reports/`
- Save training curves as `training_curves.png`

### Evaluation

```bash
python evaluation.py
```

This will:
- Load the trained model from `best_unet_model.pth`
- Evaluate on the validation set
- Generate comprehensive metrics and visualizations
- Save results in `evaluation_report/`

## Model Architecture

- **Architecture**: U-Net with EfficientNet-B0 encoder
- **Input**: 1-channel grayscale images (cropped CT slices)
- **Output**: 9-class segmentation (Background + C1-C7 + Other)
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam (lr=0.001)
- **Scheduler**: ReduceLROnPlateau

## Key Differences from E1-augmentation

1. **Dataset**: Uses pre-cropped images/masks instead of loading DICOM/NIfTI volumes
2. **Simplified Loading**: No need for spacing resampling or DICOM processing
3. **Faster**: Pre-processed data loads faster
4. **Focused**: Training on cropped regions reduces background noise

## Output Files

### Training
- `best_unet_model.pth`: Best model checkpoint
- `training_curves.png`: Training/validation loss curves
- `training_reports/`: CSV files with training efficiency metrics
- `emissions.csv`: Energy/carbon footprint tracking

### Evaluation
- `evaluation_report/evaluation_report.txt`: Detailed text report
- `evaluation_report/metrics_per_class.csv`: Per-class metrics
- `evaluation_report/sample_predictions.png`: Visual predictions
- `evaluation_report/metrics_per_class.png`: Per-class metrics plots
- `evaluation_report/confusion_matrix.png`: Confusion matrix
- `evaluation_report/evaluation_metrics.npz`: Raw metrics data

## Requirements

Same as E1-augmentation:
- PyTorch
- segmentation-models-pytorch
- torchmetrics
- nibabel
- PIL/Pillow
- scipy
- matplotlib
- pandas
- codecarbon (optional, for energy tracking)

