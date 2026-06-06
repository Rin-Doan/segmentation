# Experiment 1: Detailed Comparison Tables

## Table 1: Overall Segmentation Performance

| Metric | E1-no (Baseline) | E1-augmentation | Absolute Diff | Relative Change | Winner |
|--------|------------------|-----------------|---------------|-----------------|---------|
| Mean Dice Score | 0.225106 | 0.220838 | -0.004268 | -1.90% | **E1-no** |
| Mean IoU Score | 0.213705 | 0.208728 | -0.004977 | -2.33% | **E1-no** |
| Mean Pixel Accuracy | 0.993892 | 0.992869 | -0.001023 | -0.10% | **E1-no** |
| Std Dev (Dice) | ±0.007192 | ±0.007631 | +0.000439 | +6.10% | E1-no |
| Std Dev (IoU) | ±0.213705 | ±0.208728 | -0.004977 | -2.33% | E1-augmentation |

---

## Table 2: Per-Class Dice Score Comparison

| Class | Description | E1-no | E1-aug | Abs. Diff | Rel. Change | Winner |
|-------|-------------|-------|--------|-----------|-------------|---------|
| 0 | Background | **0.9976** | 0.9970 | -0.0006 | -0.06% | **E1-no** |
| 1 | C1 (Atlas) | **0.1081** | 0.1039 | -0.0042 | -3.88% | **E1-no** |
| 2 | C2 (Axis) | **0.1827** | 0.1665 | -0.0162 | -8.87% | **E1-no** |
| 3 | C3 | **0.0897** | 0.0807 | -0.0090 | -10.03% | **E1-no** |
| 4 | C4 | **0.0924** | 0.0906 | -0.0018 | -1.95% | **E1-no** |
| 5 | C5 | **0.0907** | 0.0886 | -0.0021 | -2.32% | **E1-no** |
| 6 | C6 | 0.1065 | **0.1069** | +0.0004 | +0.38% | **E1-aug** |
| 7 | C7 | 0.1207 | **0.1234** | +0.0027 | +2.24% | **E1-aug** |
| 8 | Other vertebrae | **0.2375** | 0.2300 | -0.0075 | -3.16% | **E1-no** |
| **Overall** | **Macro Average** | **0.2251** | **0.2208** | **-0.0043** | **-1.91%** | **E1-no** |

**Summary**: E1-no wins on 7/9 classes. E1-augmentation wins on only C6 and C7 (both marginally).

---

## Table 3: Per-Class IoU Score Comparison

| Class | Description | E1-no | E1-aug | Abs. Diff | Rel. Change | Winner |
|-------|-------------|-------|--------|-----------|-------------|---------|
| 0 | Background | **0.9953** | 0.9940 | -0.0013 | -0.13% | **E1-no** |
| 1 | C1 (Atlas) | **0.0985** | 0.0933 | -0.0052 | -5.28% | **E1-no** |
| 2 | C2 (Axis) | **0.1701** | 0.1526 | -0.0175 | -10.29% | **E1-no** |
| 3 | C3 | **0.0778** | 0.0686 | -0.0092 | -11.83% | **E1-no** |
| 4 | C4 | **0.0796** | 0.0771 | -0.0025 | -3.14% | **E1-no** |
| 5 | C5 | **0.0785** | 0.0773 | -0.0012 | -1.53% | **E1-no** |
| 6 | C6 | 0.0933 | **0.0943** | +0.0010 | +1.07% | **E1-aug** |
| 7 | C7 | 0.1095 | **0.1109** | +0.0014 | +1.28% | **E1-aug** |
| 8 | Other vertebrae | **0.2206** | 0.2103 | -0.0103 | -4.67% | **E1-no** |
| **Overall** | **Macro Average** | **0.2137** | **0.2087** | **-0.0050** | **-2.34%** | **E1-no** |

**Summary**: E1-no wins on 7/9 classes. Performance degradation is more pronounced in IoU than Dice.

---

## Table 4: Training Time and Convergence

| Metric | E1-no (Baseline) | E1-augmentation | Absolute Diff | Relative Change |
|--------|------------------|-----------------|---------------|-----------------|
| **Total Training Time (avg)** | 14.99 hours | 16.11 hours | +1.12 hours | +7.47% |
| **Total Training Time (avg)** | 53,946 seconds | 58,014 seconds | +4,068 seconds | +7.54% |
| **Average Epoch Time** | 2,794 seconds | 2,945 seconds | +150 seconds | +5.37% |
| **Convergence Epoch** | 17-19 epochs | 13 epochs | -5 epochs | -31.25% |
| **Time to Convergence** | 13.2 hours | 10.5 hours | -2.7 hours | -20.45% |
| **Average RAM Usage** | 1.249 GB | 1.245 GB | -0.004 GB | -0.32% |
| **Average GPU Memory** | 0.373 GB | 0.373 GB | 0.000 GB | 0.00% |

**Key Insight**: Augmentation converged faster (13 vs 17-19 epochs) but each epoch took longer, resulting in 7.5% more total training time.

---

## Table 5: Energy Consumption and Carbon Emissions (Training)

| Metric | E1-no (Baseline) | E1-augmentation | Absolute Diff | Relative Change |
|--------|------------------|-----------------|---------------|-----------------|
| **CO2 Emissions (avg)** | 1.416 kg | 1.460 kg | +0.044 kg | +3.11% |
| **Energy Consumed (avg)** | 2.576 kWh | 2.598 kWh | +0.022 kWh | +0.85% |
| **CO2 per Epoch** | 0.0745 kg | 0.0730 kg | -0.0015 kg | -2.01% |
| **Energy per Epoch** | 0.1288 kWh | 0.1299 kWh | +0.0011 kWh | +0.85% |
| **Training Runs Analyzed** | 5 runs | 5 runs | - | - |

**Environmental Impact**: Over 100 training runs, augmentation would produce an additional **4.4 kg of CO2** compared to baseline.

---

## Table 6: Energy Consumption and Carbon Emissions (Inference)

| Metric | E1-no (Baseline) | E1-augmentation | Absolute Diff | Relative Change |
|--------|------------------|-----------------|---------------|-----------------|
| **CO2 Emissions (avg)** | 0.0467 kg | 0.0583 kg | +0.0116 kg | +24.84% |
| **Energy Consumed (avg)** | 0.0852 kWh | 0.0878 kWh | +0.0026 kWh | +3.05% |
| **Total Inference Time** | 2,257.85 s | 2,330.26 s | +72.41 s | +3.21% |
| **Throughput** | 2.79 samples/s | 2.70 samples/s | -0.09 samples/s | -3.23% |
| **Latency per Sample** | 358.39 ms | 369.88 ms | +11.49 ms | +3.20% |
| **Test Samples** | 6,300 | 6,300 | 0 | 0% |

**Key Finding**: Inference with the augmented model is **24.8% more expensive in CO2** despite only 3.2% slower. This suggests power spikes or less efficient GPU utilization.

---

## Table 7: Hardware Power Consumption

### CPU Power

| Experiment | Min | Max | Average | Std Dev | Samples |
|------------|-----|-----|---------|---------|---------|
| **E1-no** | 27.01 W | 32.28 W | 29.37 W | ±2.05 W | 5 runs |
| **E1-augmentation** | 27.02 W | 31.60 W | 28.31 W | ±1.84 W | 5 runs |
| **Difference** | +0.01 W | -0.68 W | **-1.06 W** | - | - |
| **Relative Change** | +0.04% | -2.11% | **-3.61%** | - | - |

### GPU Power

| Experiment | Min | Max | Average | Std Dev | Samples |
|------------|-----|-----|---------|---------|---------|
| **E1-no** | 64.79 W | 108.27 W | 77.66 W | ±13.36 W | 5 runs |
| **E1-augmentation** | 63.37 W | 117.55 W | 75.98 W | ±19.95 W | 5 runs |
| **Difference** | -1.42 W | +9.28 W | **-1.68 W** | - | - |
| **Relative Change** | -2.19% | +8.57% | **-2.16%** | - | - |

### RAM Power

| Experiment | Power |
|------------|-------|
| **E1-no** | 38.0 W (constant) |
| **E1-augmentation** | 38.0 W (constant) |
| **Difference** | 0.0 W (0%) |

**Critical Observation**: Despite augmentation adding computational operations, average CPU and GPU power consumption actually **decreased slightly**. This is explained by:
1. Efficient parallel data loading pipeline
2. Augmentation in background workers doesn't block GPU
3. Same model architecture and batch size
4. Similar GPU utilization patterns

---

## Table 8: Class-wise Standard Deviation Analysis

### Dice Score Standard Deviation

| Class | E1-no | E1-augmentation | Change | Interpretation |
|-------|-------|-----------------|--------|----------------|
| 0 | ±0.0029 | ±0.0031 | +0.0002 | Background: Stable |
| 1 | ±0.2914 | ±0.2839 | -0.0075 | C1: Slightly more stable |
| 2 | ±0.3665 | ±0.3495 | -0.0170 | C2: More stable |
| 3 | ±0.2575 | ±0.2422 | -0.0153 | C3: More stable |
| 4 | ±0.2594 | ±0.2545 | -0.0049 | C4: Slightly more stable |
| 5 | ±0.2575 | ±0.2580 | +0.0005 | C5: Similar |
| 6 | ±0.2803 | ±0.2829 | +0.0026 | C6: Slightly less stable |
| 7 | ±0.3045 | ±0.3053 | +0.0008 | C7: Similar |
| 8 | ±0.4014 | ±0.3923 | -0.0091 | Other: More stable |

**Insight**: Augmentation produced slightly more stable predictions (lower variance) across most classes, but this didn't translate to better average performance.

---

## Table 9: Computational Efficiency Metrics

| Metric | E1-no | E1-augmentation | Change | Winner |
|--------|-------|-----------------|--------|---------|
| **Samples per kWh (Training)** | 2,094 | 2,035 | -59 (-2.8%) | **E1-no** |
| **Samples per kg CO2 (Training)** | 4,450 | 4,315 | -135 (-3.0%) | **E1-no** |
| **Samples per kWh (Inference)** | 73,944 | 71,731 | -2,213 (-3.0%) | **E1-no** |
| **Samples per kg CO2 (Inference)** | 134,904 | 108,080 | -26,824 (-19.9%) | **E1-no** |
| **Training Time per Dice Point** | 66.6 hours | 73.0 hours | +6.4 hrs (+9.6%) | **E1-no** |
| **CO2 per Dice Point** | 6.29 kg | 6.61 kg | +0.32 kg (+5.1%) | **E1-no** |

**Key Takeaway**: E1-no is more efficient across all metrics. The augmented model requires more resources to achieve worse performance.

---

## Table 10: Cost-Benefit Summary

| Factor | E1-no (Baseline) | E1-augmentation | Trade-off Assessment |
|--------|------------------|-----------------|----------------------|
| **Accuracy** | 0.2251 Dice | 0.2208 Dice | ❌ -1.9%, Negative |
| **Training Time** | 14.99 hrs | 16.11 hrs | ❌ +7.5%, Negative |
| **CO2 (Training)** | 1.416 kg | 1.460 kg | ❌ +3.1%, Negative |
| **CO2 (Inference)** | 0.0467 kg | 0.0583 kg | ❌ +24.8%, Negative |
| **GPU Power** | 77.66 W | 75.98 W | ✅ -2.2%, Positive |
| **Convergence Speed** | 17-19 epochs | 13 epochs | ✅ -31%, Positive |
| **Prediction Stability** | Higher variance | Lower variance | ✅ More stable, Positive |
| **Implementation** | Simple | Complex | ❌ More code, Negative |

**Verdict**: **3 Positives, 5 Negatives → Use E1-no (Baseline)**

---

## Table 11: Recommended Next Experiments

| Priority | Experiment | Expected Impact | Rationale |
|----------|-----------|-----------------|-----------|
| 🔴 **High** | Focal Loss / Dice Loss | +5-15% Dice | Address severe class imbalance |
| 🔴 **High** | Class-balanced sampling | +3-10% Dice | Oversample vertebrae slices |
| 🟡 **Medium** | Reduced augmentation (p=0.4) | +2-5% Dice | Less aggressive transforms |
| 🟡 **Medium** | 3D U-Net architecture | +10-20% Dice | Use volumetric context |
| 🟢 **Low** | Transfer learning | +3-8% Dice | Pre-train on larger datasets |
| 🟢 **Low** | Annotation refinement | +5-10% Dice | Fix noisy labels |

---

## Table 12: Augmentation Techniques Applied (E1-aug only)

| Technique | Probability | Parameter Range | Anatomical Validity | Keep/Remove |
|-----------|-------------|-----------------|---------------------|-------------|
| Overall augmentation | 80% | - | - | Reduce to 40% |
| Horizontal flip | 50% | Binary | ✅ Valid (symmetry) | **Keep** |
| Rotation | 70% | -15° to +15° | ⚠️ Questionable | Reduce to ±5° |
| Scaling | 60% | 0.9 to 1.1 | ✅ Valid | **Keep** |
| Translation | 60% | ±10% | ⚠️ Questionable | **Remove** |
| Intensity scaling/shift | 70% | ±10% | ⚠️ May conflict | Reduce to 30% |
| Gaussian noise | 50% | σ=0.02 | ⚠️ May conflict | Reduce to 20% |
| Gamma correction | 30% | 0.8 to 1.2 | ✅ Valid | **Keep** |
| Elastic deformation | 20% | α=150, σ=12 | ❌ Too aggressive | Reduce to 5% |

**Recommended Pipeline**: Keep flip, scaling, gamma. Reduce rotation. Remove/reduce others.

---

## Quick Reference: Which Model to Use?

### Use E1-no (Baseline) if:
- ✅ Accuracy is the top priority
- ✅ Minimizing training time matters
- ✅ Reducing carbon footprint is important
- ✅ You want simpler, more maintainable code
- ✅ You need faster inference (2.79 vs 2.70 samples/s)

### Use E1-augmentation if:
- ⚠️ You specifically need faster convergence (13 vs 17-19 epochs)
- ⚠️ You're experimenting with augmentation strategies
- ⚠️ You value prediction stability over accuracy

**Overall Recommendation**: **Use E1-no (Baseline)** for production. Fix data quality issues before revisiting augmentation.

---

**Report Generated**: December 19, 2025  
**Experiments Compared**: E1-no vs E1-augmentation  
**Model**: 2D U-Net with ResNet34 encoder  
**Dataset**: Vertebral CT scans (6,300 test samples, 9 classes)

