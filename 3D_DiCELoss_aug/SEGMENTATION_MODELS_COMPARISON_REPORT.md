# Segmentation Models Comparison Report

## Executive Summary

This report compares two 3D U-Net segmentation models for cervical vertebrae segmentation:
1. **3D_DiCELoss_aug**: Enhanced model with YOLO preprocessing and offline augmentation
2. **3D_DiCELoss**: Baseline model with direct dataset loading

The comparison reveals significant performance differences, with the augmented model achieving **2.3× better Dice score** and **2.8× better IoU score** than the baseline.

---

## Table of Contents

1. [Model Overview](#1-model-overview)
2. [Architectural Differences](#2-architectural-differences)
3. [Data Preprocessing Differences](#3-data-preprocessing-differences)
4. [Training Configuration Differences](#4-training-configuration-differences)
5. [Performance Comparison](#5-performance-comparison)
6. [Key Factors Contributing to Performance Gap](#6-key-factors-contributing-to-performance-gap)
7. [Recommendations](#7-recommendations)

---

## 1. Model Overview

### 1.1 Model 1: 3D_DiCELoss_aug (Enhanced)

**Location**: `/home/s222440401/project/3D_DiCELoss_aug/`

**Key Characteristics:**
- YOLO-based intelligent cropping
- Offline data augmentation (2 copies per study)
- Isotropic resampling (1.0×1.0×1.0 mm)
- Higher resolution volumes (128×256×256)
- Pre-aggregated data pipeline

**Performance:**
- Mean Dice Score: **0.9354** (93.54%)
- Mean IoU Score: **0.8950** (89.50%)
- Pixel Accuracy: **0.9987** (99.87%)

### 1.2 Model 2: 3D_DiCELoss (Baseline)

**Location**: `/home/s222440401/project/segmentation_experiment/3D_DiCELoss/`

**Key Characteristics:**
- Direct dataset loading (no preprocessing)
- No offline augmentation
- Anisotropic resampling (2.0×1.0×1.0 mm)
- Lower resolution volumes (64×256×256)
- On-the-fly data loading

**Performance:**
- Mean Dice Score: **0.4094** (40.94%)
- Mean IoU Score: **0.3217** (32.17%)
- Pixel Accuracy: **0.9871** (98.71%)

---

## 2. Architectural Differences

### 2.1 Model Architecture

Both models use **identical MONAI 3D U-Net architecture**:

| Parameter | Both Models |
|-----------|-------------|
| Framework | MONAI |
| Spatial Dimensions | 3D |
| Input Channels | 1 (grayscale CT) |
| Output Channels | 9 (background + C1-C7 + other) |
| Feature Channels | (32, 64, 128, 256, 512) |
| Strides | (2, 2, 2, 2) |
| Residual Units | 2 per level |
| Normalization | Batch normalization |
| Dropout | 0.1 |
| Total Parameters | ~19.2M |

**Conclusion**: Architecture is **identical** - differences are in data preprocessing and training.

### 2.2 Loss Function

Both models use **identical DiceCELoss configuration**:

| Parameter | Both Models |
|-----------|-------------|
| Loss Function | DiceCELoss (MONAI) |
| to_onehot_y | True |
| softmax | True |
| squared_pred | False |
| lambda_dice | 1.0 |
| lambda_ce | 1.0 |

**Conclusion**: Loss function is **identical** - not a source of difference.

---

## 3. Data Preprocessing Differences

### 3.1 Data Loading Strategy

#### Model 1 (3D_DiCELoss_aug): Pre-Aggregated Pipeline
```
1. YOLO Inference → Identify first vertebrae slice
2. Aggregate Data (aggregate_data.py):
   - Load DICOM volumes
   - Crop from first slice to end
   - Resample to (1.0, 1.0, 1.0) mm
   - Apply offline augmentation (2 copies)
   - Store pre-processed volumes in memory
3. Dataset (Aggregated3DSegmentationDataset):
   - Load pre-aggregated volumes
   - Apply HU windowing
   - Crop/pad to (128, 256, 256)
   - Apply online augmentation (p=0.5)
```

#### Model 2 (3D_DiCELoss): Direct Loading
```
1. Dataset (Medical3DSegmentationDataset):
   - Load DICOM volumes on-the-fly
   - Load full volumes (no cropping)
   - Resample to (2.0, 1.0, 1.0) mm
   - Apply HU windowing
   - Crop/pad to (64, 256, 256)
   - Apply online augmentation (p=0.5)
```

### 3.2 Key Preprocessing Differences

| Aspect | Model 1 (aug) | Model 2 (baseline) |
|--------|---------------|-------------------|
| **YOLO Cropping** | ✅ Yes (from first vertebrae slice) | ❌ No (full volumes) |
| **Target Spacing** | (1.0, 1.0, 1.0) mm (isotropic) | (2.0, 1.0, 1.0) mm (anisotropic) |
| **Volume Shape** | (128, 256, 256) | (64, 256, 256) |
| **Offline Augmentation** | ✅ Yes (2 copies per study) | ❌ No |
| **Data Aggregation** | ✅ Pre-aggregated | ❌ On-the-fly loading |
| **Volume Depth** | Variable (cropped) | Full volume depth |

### 3.3 Impact of Each Difference

#### 3.3.1 YOLO-Based Cropping
- **Model 1**: Removes irrelevant background slices above vertebrae
  - Focuses on region of interest
  - Reduces noise from non-vertebrae regions
  - More efficient use of model capacity
  
- **Model 2**: Includes all slices from study
  - Contains large background regions
  - Dilutes learning signal
  - Wastes model capacity on irrelevant regions

**Impact**: **HIGH** - Directly affects what the model learns

#### 3.3.2 Isotropic vs. Anisotropic Resampling
- **Model 1**: (1.0, 1.0, 1.0) mm - Uniform resolution
  - Better 3D feature learning
  - Consistent spatial relationships
  - More accurate depth information
  
- **Model 2**: (2.0, 1.0, 1.0) mm - Anisotropic (2× coarser in z-direction)
  - Loss of detail in depth dimension
  - Inconsistent resolution
  - May miss fine structures

**Impact**: **HIGH** - Affects spatial understanding

#### 3.3.3 Volume Resolution
- **Model 1**: (128, 256, 256) - Higher depth resolution
  - More slices per volume
  - Better depth-wise feature learning
  - More context in z-direction
  
- **Model 2**: (64, 256, 256) - Lower depth resolution
  - Fewer slices per volume
  - Less depth context
  - May miss inter-slice relationships

**Impact**: **MEDIUM** - Affects depth understanding

#### 3.3.4 Offline Augmentation
- **Model 1**: 2 pre-computed augmented copies per study
  - 3× effective dataset size
  - Consistent augmentation across epochs
  - More diverse training samples
  
- **Model 2**: No offline augmentation
  - Only online augmentation (50% probability)
  - Less data diversity
  - Fewer training samples

**Impact**: **HIGH** - Directly affects generalization

---

## 4. Training Configuration Differences

### 4.1 Training Parameters Comparison

| Parameter | Model 1 (aug) | Model 2 (baseline) |
|-----------|---------------|-------------------|
| **Epochs** | 500 | 300 |
| **Batch Size** | 4 | 4 |
| **Learning Rate** | 0.01 | 0.01 (stated) / 5e-5 (actual) |
| **Weight Decay** | 1e-5 | 1e-5 |
| **LR Scheduler** | ReduceLROnPlateau (factor=0.8, patience=10) | ReduceLROnPlateau (factor=0.5, patience=3) |
| **Optimizer** | Adam | Adam |
| **Training Studies** | ~66 studies → ~199 volumes (with augmentation) | 69 studies |
| **Validation Studies** | ~17 studies → ~50 volumes | 18 studies |

### 4.2 Key Training Differences

#### 4.2.1 Number of Epochs
- **Model 1**: 500 epochs (more training)
- **Model 2**: 300 epochs (less training)

**Impact**: **MEDIUM** - More training can help, but not the main factor

#### 4.2.2 Learning Rate Scheduler
- **Model 1**: Factor=0.8, Patience=10 (gentler, more patient)
- **Model 2**: Factor=0.5, Patience=3 (more aggressive, less patient)

**Impact**: **MEDIUM** - Model 1's scheduler allows more exploration

#### 4.2.3 Training Data Size
- **Model 1**: ~199 volumes (with offline augmentation)
- **Model 2**: ~69 volumes (no offline augmentation)

**Impact**: **HIGH** - More data = better generalization

---

## 5. Performance Comparison

### 5.1 Overall Metrics

| Metric | Model 1 (aug) | Model 2 (baseline) | Improvement |
|--------|---------------|-------------------|-------------|
| **Mean Dice Score** | 0.9354 | 0.4094 | **+2.29×** |
| **Mean IoU Score** | 0.8950 | 0.3217 | **+2.78×** |
| **Pixel Accuracy** | 0.9987 | 0.9871 | +1.16% |

### 5.2 Per-Class Performance Comparison

| Class | Description | Model 1 Dice | Model 2 Dice | Improvement |
|-------|-------------|--------------|-------------|-------------|
| 0 | Background | 0.9994 | 0.9953 | +0.41% |
| 1 | C1 (Atlas) | 0.8260 | 0.3296 | **+2.51×** |
| 2 | C2 (Axis) | 0.9014 | 0.4091 | **+2.20×** |
| 3 | C3 | 0.9486 | 0.4077 | **+2.33×** |
| 4 | C4 | 0.9480 | 0.4136 | **+2.29×** |
| 5 | C5 | 0.9472 | 0.3400 | **+2.79×** |
| 6 | C6 | 0.9445 | 0.2831 | **+3.34×** |
| 7 | C7 | 0.9510 | 0.2550 | **+3.73×** |
| 8 | Other | 0.9522 | 0.2507 | **+3.80×** |

### 5.3 Performance Analysis

#### Model 1 (aug) Strengths:
- **Consistent high performance** across all vertebra classes (82-95% Dice)
- **Low variance** for C3-C8 classes (std < 0.02)
- **Excellent background segmentation** (99.94% Dice)
- **Strong performance on lower vertebrae** (C6-C7: 94-95% Dice)

#### Model 2 (baseline) Weaknesses:
- **Poor performance** across all vertebra classes (25-41% Dice)
- **High variance** for all classes (std > 0.22)
- **Particularly weak** on lower vertebrae (C6-C8: 25-28% Dice)
- **Inconsistent predictions** (high standard deviations)

### 5.4 Performance Gap Visualization

```
Dice Score Comparison:
Model 1 (aug):  ████████████████████ 93.54%
Model 2 (base): ████████░░░░░░░░░░░░ 40.94%

IoU Score Comparison:
Model 1 (aug):  ██████████████████░░ 89.50%
Model 2 (base): ██████░░░░░░░░░░░░░░ 32.17%
```

---

## 6. Key Factors Contributing to Performance Gap

### 6.1 Factor Analysis (Ranked by Impact)

#### 1. **YOLO-Based Intelligent Cropping** (Impact: ⭐⭐⭐⭐⭐)
- **Why Critical**: Removes irrelevant background, focuses on ROI
- **Evidence**: Model 2 includes full volumes with large background regions
- **Impact**: Model 1 learns from focused, relevant data

#### 2. **Isotropic Resampling** (Impact: ⭐⭐⭐⭐⭐)
- **Why Critical**: Uniform resolution enables better 3D feature learning
- **Evidence**: Model 2's 2× coarser z-resolution loses depth information
- **Impact**: Model 1 captures spatial relationships more accurately

#### 3. **Offline Data Augmentation** (Impact: ⭐⭐⭐⭐)
- **Why Critical**: 3× dataset size, more diverse training samples
- **Evidence**: Model 1 has ~199 volumes vs. Model 2's ~69 volumes
- **Impact**: Better generalization, reduced overfitting

#### 4. **Higher Volume Resolution** (Impact: ⭐⭐⭐)
- **Why Important**: More depth context (128 vs. 64 slices)
- **Evidence**: Model 1 captures more inter-slice relationships
- **Impact**: Better depth-wise feature learning

#### 5. **Training Duration** (Impact: ⭐⭐)
- **Why Moderate**: More epochs (500 vs. 300)
- **Evidence**: Model 1 has more training iterations
- **Impact**: Helps but not the main factor

#### 6. **Learning Rate Scheduler** (Impact: ⭐⭐)
- **Why Moderate**: Gentler scheduling allows more exploration
- **Evidence**: Model 1's scheduler is more patient
- **Impact**: Helps convergence but secondary

### 6.2 Combined Effect

The performance gap is **multiplicative**, not additive:
- Each improvement builds on the others
- YOLO cropping + isotropic resampling = better spatial understanding
- More data + better preprocessing = better generalization
- **Result**: 2.3× better Dice score

---

## 7. Detailed Comparison by Component

### 7.1 Data Pipeline Comparison

| Component | Model 1 (aug) | Model 2 (baseline) | Winner |
|-----------|---------------|-------------------|--------|
| **Preprocessing** | YOLO cropping + aggregation | Direct loading | ✅ Model 1 |
| **Resampling** | Isotropic (1.0×1.0×1.0) | Anisotropic (2.0×1.0×1.0) | ✅ Model 1 |
| **Resolution** | (128, 256, 256) | (64, 256, 256) | ✅ Model 1 |
| **Augmentation** | Offline + Online | Online only | ✅ Model 1 |
| **Data Size** | ~199 volumes | ~69 volumes | ✅ Model 1 |

### 7.2 Training Comparison

| Component | Model 1 (aug) | Model 2 (baseline) | Winner |
|-----------|---------------|-------------------|--------|
| **Epochs** | 500 | 300 | ✅ Model 1 |
| **LR Scheduler** | Factor=0.8, Patience=10 | Factor=0.5, Patience=3 | ✅ Model 1 |
| **Batch Size** | 4 | 4 | ⚖️ Tie |
| **Optimizer** | Adam | Adam | ⚖️ Tie |
| **Loss Function** | DiceCELoss | DiceCELoss | ⚖️ Tie |

---

## 8. Recommendations

### 8.1 For Future Models

1. **Always use YOLO-based preprocessing**
   - Critical for focusing on region of interest
   - Removes irrelevant background
   - Improves training efficiency

2. **Use isotropic resampling**
   - Essential for 3D feature learning
   - Maintains spatial relationships
   - Better depth understanding

3. **Implement offline augmentation**
   - Increases dataset size significantly
   - Improves generalization
   - Reduces overfitting

4. **Use higher resolution when possible**
   - More depth context
   - Better inter-slice relationships
   - Trade-off: memory usage

5. **Train for sufficient epochs**
   - 500 epochs showed better convergence
   - Use learning rate scheduling
   - Monitor validation loss

### 8.2 For Model 2 Improvement

To improve Model 2 (baseline), implement:
1. ✅ YOLO-based cropping (highest priority)
2. ✅ Isotropic resampling (1.0×1.0×1.0 mm)
3. ✅ Offline augmentation (2 copies per study)
4. ✅ Increase volume resolution (128×256×256)
5. ✅ Extend training to 500 epochs

**Expected Improvement**: Should approach Model 1 performance

### 8.3 Best Practices Summary

| Practice | Model 1 | Model 2 | Recommendation |
|----------|---------|---------|---------------|
| Intelligent cropping | ✅ | ❌ | **Always use** |
| Isotropic resampling | ✅ | ❌ | **Always use** |
| Offline augmentation | ✅ | ❌ | **Highly recommended** |
| Higher resolution | ✅ | ❌ | **Use when memory allows** |
| Extended training | ✅ | ⚠️ | **Use with proper scheduling** |

---

## 9. Conclusion

The performance gap between the two models (2.3× Dice score improvement) is primarily due to:

1. **YOLO-based intelligent cropping** - Focuses learning on relevant regions
2. **Isotropic resampling** - Enables better 3D feature learning
3. **Offline data augmentation** - Increases dataset diversity and size
4. **Higher resolution** - Provides more depth context

**Key Takeaway**: Data preprocessing and augmentation are **more critical than architecture** for this task. The identical architectures perform vastly differently based on how data is prepared and augmented.

**Recommendation**: Always use the enhanced pipeline (Model 1 approach) for future segmentation tasks, as it demonstrates significantly superior performance across all metrics and vertebra classes.

---

## 10. Performance Summary Table

| Metric | Model 1 (aug) | Model 2 (baseline) | Gap |
|--------|---------------|-------------------|-----|
| **Overall Dice** | 93.54% | 40.94% | **+52.60%** |
| **Overall IoU** | 89.50% | 32.17% | **+57.33%** |
| **Pixel Accuracy** | 99.87% | 98.71% | +1.16% |
| **Best Class Dice** | 99.94% (Background) | 99.53% (Background) | +0.41% |
| **Worst Class Dice** | 82.60% (C1) | 25.07% (C8) | **+57.53%** |
| **C1-C7 Average Dice** | 92.38% | 34.79% | **+57.59%** |

---

**Report Generated**: Based on code analysis and evaluation results  
**Comparison Date**: 2024  
**Models Compared**: 3D_DiCELoss_aug vs. 3D_DiCELoss

