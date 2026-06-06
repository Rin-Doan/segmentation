# Experiment 1 Reports - Navigation Guide

This directory contains comprehensive analysis reports comparing two experimental configurations for vertebral CT segmentation:

- **E1-no**: Baseline model without data augmentation
- **E1-augmentation**: Enhanced model with comprehensive augmentation pipeline

---

## 📁 Report Files

### 1. **EXPERIMENT_1_COMPREHENSIVE_REPORT.md** (Main Report)
📄 **Full detailed analysis** - 40+ pages

**Contents:**
- Executive summary with key findings
- Overall accuracy metrics performance comparison
- Detailed per-class performance analysis (all 9 classes)
- Computational trade-offs analysis (GPU, CPU, time, CO2)
- Deep dive into why augmentation decreased performance
- Root cause analysis (class imbalance, noisy annotations)
- Hardware resource utilization explanation
- Recommendations for immediate actions and long-term solutions
- Complete methodology and experimental setup

**Use this when:**
- Writing papers or thesis chapters
- Need detailed explanations and rationale
- Preparing comprehensive technical documentation
- Understanding the "why" behind results

---

### 2. **EXECUTIVE_SUMMARY.md** (Quick Overview)
📊 **One-page summary** - 5 minutes read

**Contents:**
- Quick results table (Dice, IoU, time, CO2)
- Key finding highlighted
- Top 3 reasons augmentation failed
- GPU/CPU usage explanation (why they're similar)
- Top 3 recommendations
- Performance by class at a glance
- Bottom line verdict

**Use this when:**
- Presenting to supervisors/colleagues
- Need quick reference during discussions
- Preparing slide presentations
- Making decisions without deep technical details

---

### 3. **COMPARISON_TABLES.md** (Data Tables)
📈 **12 comprehensive tables** - Easy to copy into papers/slides

**Contents:**
- **Table 1**: Overall segmentation performance
- **Table 2**: Per-class Dice score comparison
- **Table 3**: Per-class IoU score comparison
- **Table 4**: Training time and convergence
- **Table 5**: Energy consumption (training)
- **Table 6**: Energy consumption (inference)
- **Table 7**: Hardware power consumption
- **Table 8**: Class-wise standard deviation analysis
- **Table 9**: Computational efficiency metrics
- **Table 10**: Cost-benefit summary
- **Table 11**: Recommended next experiments
- **Table 12**: Augmentation techniques applied

**Use this when:**
- Creating figures for papers
- Need specific numbers for citations
- Comparing multiple metrics side-by-side
- Planning next experiments based on data

---

### 4. **README.md** (This File)
📖 **Navigation guide** - You are here!

---

## 🎯 Quick Start Guide

### For Different Audiences:

**For Your Supervisor:**
→ Start with `EXECUTIVE_SUMMARY.md`  
→ Reference specific tables from `COMPARISON_TABLES.md` as needed

**For Paper Writing:**
→ Use `EXPERIMENT_1_COMPREHENSIVE_REPORT.md` for methodology and discussion  
→ Copy tables from `COMPARISON_TABLES.md` for results section

**For Presentation Slides:**
→ Extract key points from `EXECUTIVE_SUMMARY.md`  
→ Use Table 1, 2, 4, 10 from `COMPARISON_TABLES.md`  
→ Include existing visualization PNGs from parent directories

**For Debugging/Next Steps:**
→ Read Section 3 and 4 of `EXPERIMENT_1_COMPREHENSIVE_REPORT.md`  
→ Check Table 11 in `COMPARISON_TABLES.md` for experiment priorities

---

## 🔍 Key Findings at a Glance

### Performance
- ❌ Augmentation **decreased Dice by 1.9%** (0.2251 → 0.2208)
- ❌ Augmentation **decreased IoU by 2.3%** (0.2137 → 0.2087)
- ❌ Only 2/9 classes improved (C6, C7 by <3%)

### Computational Cost
- ⏱️ Training time **increased 7.5%** (+1.1 hours)
- 🌍 CO2 emissions **increased 3.1%** during training
- 🌍 CO2 emissions **increased 24.8%** during inference
- ⚡ GPU/CPU power **remained similar** (efficient pipeline)

### Root Causes
1. Severe class imbalance (104,860 background vs 0 vertebrae samples)
2. Noisy annotations (C3-C5 with <10% Dice)
3. Overly aggressive augmentation (80% probability, multiple transforms)

### Recommendation
**Use E1-no (baseline) for production.** Fix class imbalance with weighted loss functions before reintroducing minimal augmentation.

---

## 📊 Existing Visualizations

These PNG files were generated during training/evaluation and are in parent directories:

**From E1-augmentation/**
- `training_curves.png` - Loss curves over 20 epochs
- `evaluation_report/confusion_matrix.png` - 9×9 confusion matrix
- `evaluation_report/metrics_per_class.png` - Bar charts for Dice/IoU by class
- `evaluation_report/sample_predictions.png` - Visual predictions vs ground truth

**From E1-no/**
- `training_curves.png` - Loss curves over 20 epochs
- `evaluation_report/confusion_matrix.png` - 9×9 confusion matrix
- `evaluation_report/metrics_per_class.png` - Bar charts for Dice/IoU by class
- `evaluation_report/sample_predictions.png` - Visual predictions vs ground truth

**From experiement_reports/E1/**
- `convergence_comparison.png` - Convergence speed comparison
- `inference_performance_comparison.png` - Inference metrics comparison
- `training_performance_comparison.png` - Training metrics comparison

---

## 🔬 Methodology Summary

### Experimental Setup
- **Model**: 2D U-Net with ResNet34 encoder (ImageNet pre-trained)
- **Training**: 20 epochs, batch size 64, Adam optimizer (LR=0.001)
- **Loss**: Cross-Entropy Loss
- **Hardware**: Tesla V100-SXM2-32GB GPU
- **Data**: 6,300 test samples, 9 classes

### Augmentation Pipeline (E1-augmentation only)
Applied with 80% overall probability:
- Horizontal flip (50%)
- Rotation ±15° (70%)
- Scaling 0.9-1.1 (60%)
- Translation ±10% (60%)
- Intensity variations (70%)
- Gaussian noise σ=0.02 (50%)
- Gamma correction 0.8-1.2 (30%)
- Elastic deformation α=150, σ=12 (20%)

### Pre-processing (Both)
- HU windowing: [-200, 1800] HU
- Normalization to [0, 1]
- Spatial standardization to 1.0mm spacing
- Resize to 512×512 pixels

---

## 🚀 Recommended Next Steps

Based on the analysis, prioritize these experiments:

### Priority 1: Address Class Imbalance
**Estimated Impact**: +5-15% Dice improvement

**Actions:**
1. Replace Cross-Entropy with Focal Loss or Dice Loss
2. Implement class-balanced batch sampling
3. Oversample vertebrae-containing slices

**Code changes needed:**
```python
# In training.py, replace:
criterion = nn.CrossEntropyLoss()

# With:
from segmentation_models_pytorch.losses import DiceLoss
criterion = DiceLoss(mode='multiclass')
```

### Priority 2: Validate Annotation Quality
**Estimated Impact**: +5-10% Dice improvement

**Actions:**
1. Visualize all samples where Dice < 0.1
2. Check boundary consistency across slices
3. Consider annotation refinement pipeline

### Priority 3: Reduce Augmentation Aggressiveness
**Estimated Impact**: +2-5% Dice improvement

**Actions:**
1. Lower augmentation probability: 80% → 40%
2. Reduce rotation range: ±15° → ±5°
3. Remove translation augmentation
4. Reduce elastic deformation: 20% → 5%

**Code changes needed:**
```python
# In training.py:
AUGMENT_P = 0.4  # Changed from 0.8

# In data_process.py, CTVertebralAugmentation.__call__:
# Reduce rotation
if np.random.random() > 0.3:
    angle = np.random.uniform(-5, 5)  # Changed from ±15

# Reduce elastic deformation
if np.random.random() > 0.95:  # Changed from 0.8
    image, label = self._elastic_transform(image, label)
```

---

## 📧 Questions or Issues?

If you need clarification on any results or recommendations:

1. Check the relevant section in `EXPERIMENT_1_COMPREHENSIVE_REPORT.md`
2. Look up specific numbers in `COMPARISON_TABLES.md`
3. Review the methodology in Appendix section

For implementation help, refer to:
- Training code: `E1-augmentation/training.py` and `E1-no/training.py`
- Data processing: `E1-augmentation/data_process.py` and `E1-no/data_process.py`
- Evaluation: `E1-augmentation/evaluation.py` and `E1-no/evaluation.py`

---

## 📅 Report Information

- **Generated**: December 19, 2025
- **Experiments**: E1-no vs E1-augmentation
- **Data Sources**:
  - `evaluation_report/evaluation_report.txt`
  - `evaluation_report/metrics_per_class.csv`
  - `training_reports/training_efficiency.csv`
  - `training_reports/convergence.csv`
  - `emissions.csv` and `inference_emissions.csv`
  
- **Total Training Runs Analyzed**: 10 runs (5 per experiment)
- **Total Test Samples**: 6,300 samples per experiment

---

## 📝 Citation

If using these results in publications, please cite:

```
Experiment 1: Impact of Pre-processing and Data Augmentation on Vertebral Segmentation
Model: 2D U-Net with ResNet34 encoder
Hardware: Tesla V100-SXM2-32GB GPU
Dataset: Vertebral CT scans (9 classes)
Date: December 2025
```

---

**End of Navigation Guide**

Choose the appropriate report based on your needs and audience. All reports are written in Markdown format and can be converted to PDF, HTML, or integrated into LaTeX documents.


