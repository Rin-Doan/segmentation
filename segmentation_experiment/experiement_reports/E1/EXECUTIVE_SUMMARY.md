# Experiment 1: Executive Summary
## Pre-processing and Data Augmentation Impact on Vertebral Segmentation

---

## Quick Results Overview

| Metric | Baseline (E1-no) | With Augmentation (E1-aug) | Change |
|--------|------------------|----------------------------|---------|
| **Dice Score** | 0.2251 | 0.2208 | **-1.9%** ⬇️ |
| **IoU Score** | 0.2137 | 0.2087 | **-2.3%** ⬇️ |
| **Training Time** | 14.99 hrs | 16.11 hrs | **+7.4%** ⬆️ |
| **CO2 Emissions** | 1.416 kg | 1.460 kg | **+3.1%** ⬆️ |
| **CPU Power** | 29.37 W | 28.31 W | **-3.6%** ⬇️ |
| **GPU Power** | 77.66 W | 75.98 W | **-2.2%** ⬇️ |

---

## Key Finding

**Data augmentation decreased performance while increasing computational cost.**

- ❌ Worse segmentation accuracy (-1.9% Dice)
- ❌ Longer training time (+1.1 hours)
- ❌ Higher carbon footprint (+44g CO2)
- ✅ Similar GPU/CPU power usage (efficient pipeline)
- ✅ Faster convergence (13 vs 17-19 epochs)

---

## Why Did Augmentation Fail?

### 1. Severe Class Imbalance
- **Background**: 104,860 samples
- **Vertebrae**: 0 samples in test set metrics
- Model achieves 99.3% accuracy by just predicting background

### 2. Noisy Annotations
- C3, C4, C5 vertebrae: <10% Dice score
- High variability (±0.24-0.40 std dev)
- Augmentation amplifies annotation errors

### 3. Overly Aggressive Augmentation
- 80% augmentation probability
- Multiple transformations applied simultaneously
- Creates domain shift from test data

---

## Why GPU/CPU Usage Was Similar

**Despite augmentation adding operations, power consumption stayed nearly identical:**

1. **Parallel Data Loading**: 8 worker processes perform CPU augmentation in background
2. **Pipeline Overlap**: Augmentation happens while GPU trains on previous batch
3. **GPU Bottleneck**: Both models limited by same GPU operations (75W GPU >> 28W CPU)
4. **Efficient Implementation**: NumPy/SciPy operations don't block GPU

**Result**: Only 7% increase in total time, but no increase in peak power draw. This demonstrates excellent pipeline optimization.

---

## Top Recommendations

### Immediate Actions

1. **Address Class Imbalance First**
   - Use Focal Loss or Dice Loss instead of Cross-Entropy
   - Oversample vertebrae-containing slices
   - Weighted sampling to balance classes

2. **Reduce Augmentation Aggressiveness**
   - Lower probability: 80% → 40%
   - Keep only anatomically valid transforms (flip, small rotations)
   - Remove or reduce elastic deformation

3. **Validate Annotation Quality**
   - Review samples where Dice < 0.1
   - Check boundary consistency
   - Consider annotation refinement

### Long-Term Solutions

- Collect more balanced dataset
- Implement 3D segmentation (use volumetric context)
- Apply transfer learning from larger medical imaging datasets

---

## Bottom Line

**For this vertebral segmentation task, the baseline model without augmentation is superior.**

The problem is not augmentation itself, but the **combination of severe class imbalance and noisy annotations**. Fix data quality issues before reintroducing carefully designed, minimal augmentation.

### Trade-off Assessment

| Aspect | Cost | Benefit | Verdict |
|--------|------|---------|---------|
| Accuracy | -1.9% Dice | None | ❌ Negative |
| Training Time | +1.1 hours | None | ❌ Negative |
| Carbon Impact | +44g CO2 | None | ❌ Negative |
| Convergence Speed | None | -31% epochs | ✅ Positive |
| Pipeline Efficiency | None | No power increase | ✅ Positive |

**Overall**: Augmentation provides no accuracy benefit while increasing cost and environmental impact. The efficient pipeline implementation is the only positive outcome.

---

## Performance by Class

| Class | Baseline Dice | Augmentation Dice | Change | Status |
|-------|--------------|-------------------|---------|---------|
| **C1** | 0.1081 | 0.1039 | -3.9% | ⬇️ Worse |
| **C2** | 0.1827 | 0.1665 | -8.9% | ⬇️ Worse |
| **C3** | 0.0897 | 0.0807 | -10.0% | ⬇️ Worse |
| **C4** | 0.0924 | 0.0906 | -1.9% | ⬇️ Worse |
| **C5** | 0.0907 | 0.0886 | -2.3% | ⬇️ Worse |
| **C6** | 0.1065 | 0.1069 | +0.4% | ⬆️ Better |
| **C7** | 0.1207 | 0.1234 | +2.2% | ⬆️ Better |
| **C8** | 0.2375 | 0.2300 | -3.2% | ⬇️ Worse |

Only C6 and C7 showed marginal improvement; all other classes degraded.

---

## Files Generated

- `EXPERIMENT_1_COMPREHENSIVE_REPORT.md` - Full detailed analysis
- `EXECUTIVE_SUMMARY.md` - This quick reference (you are here)

**Date Generated**: December 19, 2025

