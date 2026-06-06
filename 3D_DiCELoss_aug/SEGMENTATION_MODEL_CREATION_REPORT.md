# 3D U-Net Segmentation Model Creation Report

## Executive Summary

This report documents the complete pipeline for creating a 3D U-Net segmentation model for cervical vertebrae segmentation in CT scans. The pipeline includes a YOLO-based preprocessing step for intelligent cropping, followed by a comprehensive 3D segmentation model training process.

---

## Table of Contents

1. [YOLO Detection Model](#1-yolo-detection-model)
2. [Data Preprocessing Pipeline](#2-data-preprocessing-pipeline)
3. [Model Architecture](#3-model-architecture)
4. [Training Configuration](#4-training-configuration)
5. [Loss Function](#5-loss-function)
6. [Data Augmentation](#6-data-augmentation)
7. [Model Performance](#7-model-performance)

---

## 1. YOLO Detection Model

### 1.1 Purpose
The YOLO model serves as a preprocessing step to identify the first slice containing vertebrae in each CT scan. This enables intelligent cropping, removing irrelevant background slices and focusing computational resources on the region of interest.

### 1.2 Model Architecture
- **Model**: YOLOv8m (medium size)
- **Task**: Binary object detection (vertebra vs. background)
- **Input**: 2D CT slices (512×512 pixels, PNG format)
- **Output**: Bounding boxes around vertebrae regions

### 1.3 Dataset Preparation

#### Data Sources
- **Training Images**: DICOM files from `training_images/` directory
- **Segmentation Masks**: NIfTI files from `segmentations/` directory
- **Study List**: CSV file containing 83 study IDs

#### Preprocessing Steps
1. **DICOM to PNG Conversion**
   - Extract 2D slices from DICOM series
   - Apply HU windowing: clip to [-200, 1800] HU
   - Normalize to [0, 1] range
   - Resize to 512×512 pixels
   - Convert to PNG format

2. **Bounding Box Extraction**
   - Load corresponding NIfTI segmentation masks
   - For each slice, create a binary mask (vertebra > 0)
   - Extract a single bounding box encompassing ALL vertebrae in the slice
   - Add 10-pixel padding around bounding boxes
   - Normalize coordinates to YOLO format (0-1 range)

3. **Dataset Split**
   - Train/Validation split: 80/20 (random_state=42)
   - Training samples: ~500 slices
   - Validation samples: ~125 slices

### 1.4 Training Configuration
- **Epochs**: 100
- **Batch Size**: 16
- **Image Size**: 640×640
- **Learning Rate**: Default YOLOv8 adaptive learning rate
- **Early Stopping**: Patience = 50 epochs
- **Optimizer**: Adam (built into YOLOv8)

### 1.5 YOLO Model Performance

**Detection Metrics (on Validation Set):**
- **mAP@0.5**: 0.9641 (96.41%)
- **mAP@0.5:0.95**: 0.9223 (92.23%)
- **Precision**: 0.9624 (96.24%)
- **Recall**: 0.9471 (94.71%)
- **F1-Score**: 0.9547 (95.47%)

**Detection Statistics:**
- Mean confidence: 0.9215
- Mean detections per image: 0.80
- Mean box width (normalized): 0.3896
- Mean box height (normalized): 0.3392

**Interpretation:**
The YOLO model achieves excellent performance with:
- High precision (96.24%): Most detected objects are actually vertebrae
- High recall (94.71%): Most vertebrae are detected
- High mAP scores indicate robust detection across different IoU thresholds

### 1.6 Inference and First Slice Detection
- YOLO model is run on all slices of each study
- First slice with detected vertebrae is identified
- Results saved to `yolo_inference_results.csv` with format:
  - `StudyInstanceUID`: Study identifier
  - `slice_number`: First slice index (1-indexed)

---

## 2. Data Preprocessing Pipeline

### 2.1 Overview
The preprocessing pipeline transforms raw DICOM volumes and NIfTI segmentations into standardized 3D volumes ready for training.

### 2.2 Step-by-Step Preprocessing

#### Step 1: Data Aggregation (`aggregate_data.py`)
1. **Load YOLO Inference Results**
   - Read `yolo_inference_results.csv` to get first slice indices for each study
   - Convert to 0-indexed slice numbers

2. **Volume Loading**
   - **DICOM Volumes**: Load entire DICOM series as 3D volumes
     - Sort slices by `InstanceNumber` DICOM tag
     - Stack into 3D array: (Depth, Height, Width)
   - **NIfTI Segmentations**: Load segmentation volumes
     - Apply coordinate correction: `[:, ::-1, ::-1].transpose(2, 1, 0)`
     - Convert to int64 format

3. **Intelligent Cropping**
   - Crop from first vertebrae slice (from YOLO) to end of volume
   - Removes irrelevant background slices above vertebrae
   - Reduces computational load and focuses on region of interest

4. **Spacing Extraction**
   - **DICOM**: Extract spacing from `PixelSpacing` and `ImagePositionPatient`
     - Returns: (z_spacing, y_spacing, x_spacing) in mm
   - **NIfTI**: Extract spacing from header `pixdim`
     - Returns: (z_spacing, y_spacing, x_spacing) in mm

5. **Resampling to Standard Spacing**
   - **Target Spacing**: (1.0, 1.0, 1.0) mm (isotropic)
   - **Image Resampling**: Linear interpolation (order=1)
   - **Segmentation Resampling**: Nearest neighbor (order=0) to preserve labels
   - Uses `scipy.ndimage.zoom` with calculated zoom factors

6. **Offline Data Augmentation** (Optional)
   - Creates `n_aug_per_study=2` augmented copies per study
   - Augmentations applied:
     - Horizontal flip (70% chance)
     - Axial rotation (-10° to +10°, 50% chance)
     - Scaling (95-105%, 40% chance)
     - Intensity scaling/shifting (60% chance)
     - Gaussian noise (40% chance)
     - Gamma correction (30% chance)
   - Augmented volumes stored with study IDs: `{study_id}_aug1`, `{study_id}_aug2`

#### Step 2: Dataset Creation (`data_process.py`)
1. **HU Windowing and Normalization**
   - Clip CT values to [-200, 1800] HU range
   - Normalize to [0, 1]: `(value - (-200)) / (1800 - (-200))`

2. **Spatial Standardization**
   - Crop or pad volumes to target shape: (128, 256, 256)
   - Center-crop if larger, center-pad if smaller
   - Maintains aspect ratio and spatial relationships

3. **Label Remapping**
   - Keep classes 0-7: background (0) + C1-C7 vertebrae (1-7)
   - Map classes 8-14 → class 8 (other vertebrae)
   - Reduces class imbalance by grouping lower vertebrae

4. **Online Data Augmentation** (Training Only)
   - Applied with probability `augment_p=0.5` during training
   - Same augmentation pipeline as offline augmentation
   - Applied on-the-fly to increase data diversity

### 2.3 Preprocessing Statistics
- **Original Studies**: 83 studies
- **Successful Processing**: ~83 studies (after filtering)
- **Augmented Copies**: 2 per study → ~249 total volumes
- **Average Resampled Slices**: ~128 slices per volume
- **Volume Shape**: (128, 256, 256) voxels

---

## 3. Model Architecture

### 3.1 Framework
- **Framework**: MONAI (Medical Open Network for AI)
- **Base Architecture**: 3D U-Net

### 3.2 Architecture Details

```python
UNet(
    spatial_dims=3,              # 3D convolutions
    in_channels=1,               # Grayscale CT input
    out_channels=9,             # 9 classes (background + C1-C7 + other)
    channels=(32, 64, 128, 256, 512),  # Feature channels at each level
    strides=(2, 2, 2, 2),        # Downsampling stride (2×2×2 pooling)
    num_res_units=2,             # Number of residual units per level
    norm='batch',                # Batch normalization
    dropout=0.1,                 # Dropout for regularization
)
```

### 3.3 Architecture Components

1. **Encoder (Downsampling Path)**
   - 5 levels of downsampling
   - Feature channels: 32 → 64 → 128 → 256 → 512
   - Each level: 2 residual units + batch normalization + dropout
   - Stride-2 convolutions for downsampling

2. **Decoder (Upsampling Path)**
   - 4 levels of upsampling
   - Feature channels: 512 → 256 → 128 → 64 → 32
   - Skip connections from encoder to decoder
   - Transposed convolutions for upsampling

3. **Bottleneck**
   - Deepest level with 512 feature channels
   - Captures high-level semantic features

4. **Output Layer**
   - Final convolution: 32 → 9 channels
   - Softmax applied in loss function (not in model)

### 3.4 Model Parameters
- **Total Parameters**: ~19,236,000 (19.2M)
- **Trainable Parameters**: ~19,236,000
- **Model Size**: ~73 MB (best model)

---

## 4. Training Configuration

### 4.1 Training Setup
- **Epochs**: 500
- **Batch Size**: 4 (limited by GPU memory for 3D volumes)
- **Device**: CUDA (GPU)
- **Number of Workers**: 2 (data loading)

### 4.2 Optimizer
- **Type**: Adam
- **Learning Rate**: 0.01 (initial)
- **Weight Decay**: 1e-5 (L2 regularization)

### 4.3 Learning Rate Scheduling
- **Scheduler**: `ReduceLROnPlateau`
- **Mode**: Minimize validation loss
- **Factor**: 0.8 (reduce LR by 20% when plateau detected)
- **Patience**: 10 epochs (wait 10 epochs before reducing LR)

### 4.4 Data Split
- **Train/Validation Split**: 80/20
- **Split Method**: By study IDs (ensures no data leakage)
- **Random State**: 42 (reproducible)
- **Training Volumes**: ~199 volumes (including augmented)
- **Validation Volumes**: ~50 volumes

### 4.5 Training Process
1. **Epoch Loop**: 500 epochs
2. **Training Phase**:
   - Forward pass through model
   - Calculate DiceCELoss
   - Backward pass (gradient computation)
   - Optimizer step (parameter update)
3. **Validation Phase**:
   - Forward pass only (no gradients)
   - Calculate validation loss
   - Update learning rate scheduler
4. **Model Saving**:
   - Save best model when validation loss improves
   - Best model saved as: `best_unet3d_model.pth`

---

## 5. Loss Function

### 5.1 DiceCELoss
The model uses **DiceCELoss** from MONAI, which combines:
- **Dice Loss**: Measures overlap between prediction and ground truth
- **Cross-Entropy Loss**: Standard classification loss

### 5.2 Loss Function Configuration

```python
DiceCELoss(
    to_onehot_y=True,      # Convert labels to one-hot encoding
    softmax=True,          # Apply softmax to predictions
    squared_pred=False,    # Use standard Dice formula (not squared)
    lambda_dice=1.0,      # Weight for Dice loss component
    lambda_ce=1.0          # Weight for CrossEntropy loss component
)
```

### 5.3 Why DiceCELoss?
1. **Dice Loss**: 
   - Specifically designed for segmentation tasks
   - Handles class imbalance better than pure CrossEntropy
   - Measures overlap directly (similar to Dice coefficient metric)

2. **Cross-Entropy Loss**:
   - Provides pixel-level classification signal
   - Helps with boundary refinement
   - Complements Dice loss

3. **Combined Benefits**:
   - Better convergence than either loss alone
   - Handles both global (Dice) and local (CE) aspects of segmentation
   - Standard in medical image segmentation

---

## 6. Data Augmentation

### 6.1 Augmentation Strategy
Two-stage augmentation pipeline:
1. **Offline Augmentation**: Pre-computed during data aggregation
2. **Online Augmentation**: Applied on-the-fly during training

### 6.2 Augmentation Techniques

#### 1. Horizontal Flip (70% chance)
- **Type**: Left-right flip along width axis
- **Rationale**: Anatomical bilateral symmetry
- **Applied to**: Both image and segmentation

#### 2. Axial Rotation (50% chance)
- **Type**: Rotation around z-axis (in-plane)
- **Range**: -10° to +10°
- **Rationale**: Patient positioning variations
- **Interpolation**: Linear for images, nearest for labels

#### 3. Scaling (40% chance)
- **Type**: Uniform scaling
- **Range**: 95% to 105%
- **Rationale**: Size variations between patients
- **Post-processing**: Crop/pad back to original shape

#### 4. Intensity Scaling and Shifting (60% chance)
- **Scaling**: 0.9× to 1.1×
- **Shifting**: -0.1 to +0.1
- **Rationale**: Scanner differences, contrast variations
- **Clipping**: Values clipped to [0, 1]

#### 5. Gaussian Noise (40% chance)
- **Type**: Additive Gaussian noise
- **Standard Deviation**: 0.01
- **Rationale**: Robustness to noise, scanner artifacts
- **Clipping**: Values clipped to [0, 1]

#### 6. Gamma Correction (30% chance)
- **Type**: Power-law transformation
- **Range**: γ ∈ [0.85, 1.15]
- **Rationale**: Contrast/brightness variations
- **Formula**: `image^γ`

### 6.3 Augmentation Impact
- **Offline**: 2 augmented copies per study → 3× data increase
- **Online**: 50% probability during training → additional diversity
- **Total Effective Augmentation**: ~6× data diversity

---

## 7. Model Performance

### 7.1 Evaluation Metrics
- **Dice Score (F1-Score)**: Overlap between prediction and ground truth
- **IoU (Jaccard Index)**: Intersection over Union
- **Pixel Accuracy**: Overall classification accuracy
- **Per-Class Metrics**: Individual performance for each vertebra class

### 7.2 Overall Performance

**On Validation Set (17 volumes):**
- **Mean Pixel Accuracy**: 0.9987 ± 0.0002 (99.87%)
- **Mean Dice Score (macro)**: 0.9354 (93.54%)
- **Mean IoU Score (macro)**: 0.8950 (89.50%)

### 7.3 Per-Class Performance

| Class | Description | Dice Score | IoU Score | Accuracy |
|-------|-------------|------------|-----------|----------|
| 0 | Background | 0.9994 ± 0.0001 | 0.9987 ± 0.0002 | 0.9993 |
| 1 | C1 (Atlas) | 0.8260 ± 0.3019 | 0.7768 ± 0.2847 | 0.8327 |
| 2 | C2 (Axis) | 0.9014 ± 0.2255 | 0.8649 ± 0.2166 | 0.9057 |
| 3 | C3 | 0.9486 ± 0.0076 | 0.9023 ± 0.0138 | 0.9468 |
| 4 | C4 | 0.9480 ± 0.0071 | 0.9013 ± 0.0128 | 0.9509 |
| 5 | C5 | 0.9472 ± 0.0067 | 0.8998 ± 0.0121 | 0.9546 |
| 6 | C6 | 0.9445 ± 0.0113 | 0.8950 ± 0.0201 | 0.9508 |
| 7 | C7 | 0.9510 ± 0.0092 | 0.9068 ± 0.0166 | 0.9573 |
| 8 | Other | 0.9522 ± 0.0157 | 0.9091 ± 0.0279 | 0.9575 |

### 7.4 Performance Analysis

**Strengths:**
- Excellent background segmentation (99.94% Dice)
- High performance on C3-C7 vertebrae (94-95% Dice)
- Consistent performance across middle vertebrae
- Low variance for C3-C8 classes

**Challenges:**
- C1 (Atlas) shows lower performance (82.6% Dice) with high variance
  - Likely due to unique anatomy and shape variations
- C2 (Axis) shows moderate performance (90.1% Dice) with high variance
  - Complex structure with dens axis

**Best Performing Classes:**
- Background: 99.94% Dice
- C7: 95.10% Dice
- Other vertebrae: 95.22% Dice

**Worst Performing Classes:**
- C1: 82.60% Dice (still good, but lower than others)

---

## 8. Key Innovations and Design Decisions

### 8.1 YOLO-Based Preprocessing
- **Innovation**: Use object detection to identify first vertebrae slice
- **Benefit**: Removes irrelevant background, focuses computation on ROI
- **Impact**: Reduces volume size, improves training efficiency

### 8.2 Isotropic Resampling
- **Decision**: Resample to (1.0, 1.0, 1.0) mm spacing
- **Benefit**: Uniform resolution in all directions
- **Impact**: Better 3D feature learning

### 8.3 Two-Stage Augmentation
- **Innovation**: Offline + online augmentation
- **Benefit**: Maximum data diversity without runtime overhead
- **Impact**: Better generalization, reduced overfitting

### 8.4 Label Remapping
- **Decision**: Group classes 8-14 into single "other" class
- **Benefit**: Reduces class imbalance
- **Impact**: Better learning for underrepresented classes

### 8.5 Higher Resolution
- **Decision**: Use (128, 256, 256) shape vs. (64, 256, 256)
- **Benefit**: More detail in depth dimension
- **Trade-off**: Increased memory usage, smaller batch size

---

## 9. Conclusion

The 3D U-Net segmentation model achieves excellent performance (93.54% mean Dice score) through:
1. Intelligent preprocessing with YOLO-based cropping
2. Comprehensive data augmentation (offline + online)
3. Isotropic resampling for uniform resolution
4. Combined Dice + CrossEntropy loss function
5. MONAI 3D U-Net architecture optimized for medical imaging

The model demonstrates strong performance across most vertebra classes, with particular strength in C3-C7 segmentation. The pipeline is reproducible, well-documented, and suitable for clinical deployment with appropriate validation.

---

## 10. Files and Artifacts

### Model Files
- `best_unet3d_model.pth`: Best model weights (73 MB)

### Data Files
- `yolo_inference_results.csv`: First slice indices from YOLO
- `aggregation_report.csv`: Data aggregation statistics

### Evaluation Files
- `evaluation_report_3d/evaluation_report.txt`: Detailed metrics
- `evaluation_report_3d/metrics_per_class.csv`: Per-class metrics
- `evaluation_report_3d/sample_predictions.png`: Visualization
- `evaluation_report_3d/confusion_matrix.png`: Confusion matrix

### Training Files
- `training.py`: Main training script
- `data_process.py`: Dataset and augmentation classes
- `aggregate_data.py`: Data preprocessing pipeline
- `evaluation.py`: Model evaluation script

---

**Report Generated**: Based on code analysis and evaluation results  
**Model Version**: 3D_DiCELoss_aug  
**Date**: 2024

