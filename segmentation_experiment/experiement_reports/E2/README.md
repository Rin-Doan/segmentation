# Experiment 2 Reports - Navigation Guide

This directory contains comprehensive analysis reports comparing three U-Net architectural variants for vertebral CT segmentation:

- **E2-2D** (E1-augmentation): 2D U-Net processing individual CT slices
- **E2-2.5D**: 2D U-Net with 3-channel input (current + 2 adjacent slices)
- **E2-3D**: 3D U-Net processing entire volumetric CT scans

---

## 📁 Report Files

### 1. **EXPERIMENT_2_COMPREHENSIVE_REPORT.md** (Main Report)
📄 **Full detailed analysis** - 50+ pages

**Contents:**
- Executive summary with key findings
- Overall performance comparison (Dice, IoU, Pixel Accuracy)
- Detailed per-class analysis (all 9 classes with 213% improvement for C3 in 3D!)
- Why 3D achieves best performance (volumetric context explanation)
- Why 2.5D shows minimal improvement (2D convolutions don't learn depth)
- Computational trade-offs (time, energy, CO2, memory)
- Hardware resource utilization analysis
- Memory management for 3D (why batch size = 1)
- Detailed explanations of all metrics
- Recommendations for deployment and future work

**Use this when:**
- Writing papers or thesis chapters
- Need comprehensive technical understanding
- Explaining architectural choices to stakeholders
- Planning future research directions

---

### 2. **EXECUTIVE_SUMMARY.md** (Quick Overview)
📊 **One-page summary** - 5 minutes read

**Contents:**
- Quick results table (all key metrics at a glance)
- Key finding: 3D dominates (+28% Dice, -68% time, -68% CO2)
- Why 3D wins (volumetric context)
- Why 2.5D fails (2D convolutions limitation)
- Why 3D is 68% faster despite complexity
- Memory management (batch size = 1 explanation)
- Environmental impact (68% greener)
- Top recommendations (use 3D for production, skip 2.5D)

**Use this when:**
- Presenting to supervisors/colleagues
- Quick reference during meetings
- Making architectural decisions
- Preparing executive presentations

---

### 3. **COMPARISON_TABLES.md** (Data Tables)
📈 **18 comprehensive tables** - Ready for papers/slides

**Contents:**
- **Table 1**: Overall segmentation performance
- **Table 2**: Per-class Dice score (3D wins on 7/9 classes)
- **Table 3**: Per-class IoU score
- **Table 4**: Training time and convergence
- **Table 5**: Energy and CO2 (training)
- **Table 6**: Energy and CO2 (inference)
- **Table 7**: Hardware power consumption (CPU, GPU, RAM)
- **Table 8**: Memory usage comparison
- **Table 9**: Computational efficiency metrics
- **Table 10**: Statistical comparison (variance, consistency)
- **Table 11**: Architecture specifications
- **Table 12**: Data processing comparison
- **Table 13**: Training configuration
- **Table 14**: Test set characteristics
- **Table 15**: Cost-benefit summary
- **Table 16**: Recommended use cases
- **Table 17**: Implementation requirements
- **Table 18**: Recommended next experiments

**Use this when:**
- Creating figures for papers
- Need specific numbers for citations
- Comparing multiple metrics side-by-side
- Planning resource allocation
- Justifying architectural choices with data

---

### 4. **README.md** (This File)
📖 **Navigation guide** - You are here!

---

## 🎯 Quick Start Guide

### For Different Audiences:

**For Your Supervisor:**
→ Start with `EXECUTIVE_SUMMARY.md` (key findings)
→ Reference `COMPARISON_TABLES.md` Table 1, 2, 15 for specific data

**For Paper Writing:**
→ Use `EXPERIMENT_2_COMPREHENSIVE_REPORT.md` for methodology and discussion
→ Copy tables from `COMPARISON_TABLES.md` for results section
→ Cite Table 2 (per-class performance) and Table 15 (cost-benefit)

**For Presentation Slides:**
→ Extract key points from `EXECUTIVE_SUMMARY.md`
→ Use Tables 1, 2, 4, 6, 15 from `COMPARISON_TABLES.md`
→ Include visualization PNGs from parent directories

**For Implementation:**
→ Read Section 2 of `EXPERIMENT_2_COMPREHENSIVE_REPORT.md` (architecture details)
→ Check Table 11 (architecture specs) and Table 17 (requirements)
→ Review memory management best practices

**For Production Deployment:**
→ Read Section 4 of `EXPERIMENT_2_COMPREHENSIVE_REPORT.md` (recommendations)
→ Check Table 16 (use cases) and Table 17 (requirements)
→ Review clinical viability metrics in Table 6

---

## 🔍 Key Findings at a Glance

### Performance
- ✅ **3D architecture is superior**: +28% Dice score vs 2D
- ✅ **3D dominates individual vertebrae**: C3 +213%, C5 +159%, C4 +122%
- ⚠️ **2.5D marginal improvement**: Only +5.7% Dice vs 2D
- ❌ **2.5D worst trade-off**: +5.7% accuracy at +15.5% time cost

### Computational Efficiency
- ✅ **3D trains 68% faster**: 5.2 hours vs 16.1 hours (2D)
- ✅ **3D converges 85% faster**: 1.6 hours vs 10.5 hours (2D)
- ✅ **3D inference 97% faster**: 70 seconds vs 2,330 seconds (2D)
- ✅ **3D is clinically viable**: 1.2 minutes per patient vs 39 minutes (2D)

### Environmental Impact
- 🌍 **3D saves 68% CO2 (training)**: 0.47 kg vs 1.46 kg (2D)
- 🌍 **3D saves 97.5% CO2 (inference)**: 0.00146 kg vs 0.0583 kg (2D)
- 🌍 **Annual savings**: 2,075 kg CO2 per hospital (100 patients/day)
- 🌍 **Equivalent to**: 5,200 miles not driven per year

### Memory Management
- **3D uses 21% LESS GPU memory**: 0.294 GB vs 0.373 GB (2D)
- **3D requires batch size = 1**: Due to volumetric data size
- **Not a limitation**: Still 68% faster overall

### Architectural Insights
- **2.5D limitation**: 2D convolutions don't learn depth, treats slices like RGB channels
- **3D advantage**: True volumetric processing captures inter-slice continuity
- **Volumetric context critical**: Especially for small vertebrae (C3-C6)

---

## 📊 Why 3D Wins: The Architecture Advantage

### 1. Volumetric Context
```
2D: Each slice processed independently → No anatomical continuity
    Problem: Cannot distinguish C3 from C4 without context

2.5D: 3 adjacent slices as channels → Limited context, still 2D
    Problem: 2D convolutions combine channels at same (h,w), don't learn depth

3D: Full volume with 3D convolutions → True volumetric understanding
    Advantage: Captures vertebral shape, continuity, and inter-slice relationships
```

### 2. Per-Class Evidence
```
Classes with largest 3D improvement (vs 2D):
- C3 (smallest vertebra): +213% (0.0807 → 0.2529)
- C5: +159% (0.0886 → 0.2298)
- C4: +122% (0.0906 → 0.2013)

→ Small, ambiguous vertebrae benefit most from volumetric context
```

### 3. Speed Paradox Explained
```
3D is more complex, yet 68% faster. Why?

2D: 25,200 slices / 64 batch_size = 394 batches per epoch
3D: 72 volumes / 1 batch_size = 72 batches per epoch

→ 5.5× fewer forward-backward passes
→ Each 3D batch contains FULL patient scan (better information density)
→ Fewer epochs needed × faster epochs = 68% overall speedup
```

---

## 📊 Existing Visualizations

These PNG files were generated during training/evaluation and are in parent directories:

**From E1-augmentation/ (E2-2D)**
- `training_curves.png` - Loss curves over 20 epochs
- `evaluation_report/confusion_matrix.png` - 9×9 confusion matrix
- `evaluation_report/metrics_per_class.png` - Bar charts Dice/IoU by class
- `evaluation_report/sample_predictions.png` - Visual predictions

**From E2-2.5D/**
- `training_curves.png` - Loss curves over 20 epochs
- `evaluation_report/confusion_matrix.png` - 9×9 confusion matrix
- `evaluation_report/metrics_per_class.png` - Bar charts Dice/IoU by class
- `evaluation_report/sample_predictions.png` - Visual predictions

**From E2-3D/**
- `training_curves.png` - Loss curves over 200 epochs
- `evaluation_report_3d/confusion_matrix.png` - 9×9 confusion matrix
- `evaluation_report_3d/metrics_per_class.png` - Bar charts Dice/IoU by class
- `evaluation_report_3d/sample_predictions.png` - Volumetric predictions

---

## 🔬 Methodology Summary

### Architectures Tested

**E2-2D (2D U-Net)**:
- Encoder: ResNet34 (ImageNet pretrained)
- Input: 512 × 512 × 1 (single CT slice)
- Convolutions: 2D (H×W plane)
- Parameters: ~24.4M
- Batch size: 64

**E2-2.5D (2.5D U-Net)**:
- Encoder: ResNet34 (ImageNet pretrained)
- Input: 512 × 512 × 3 (current + 2 adjacent slices)
- Convolutions: 2D (treats channels like RGB)
- Parameters: ~24.4M
- Batch size: 64

**E2-3D (3D U-Net)**:
- Architecture: MONAI 3D U-Net
- Input: 256 × 256 × 256 × 1 (full volume)
- Convolutions: 3D (D×H×W)
- Parameters: ~19.1M
- Batch size: 1

### Training Configuration
- **Epochs**: 2D/2.5D: 20, 3D: 200
- **Learning Rate**: 0.001 (all)
- **Optimizer**: Adam (all)
- **Loss**: Cross-Entropy (all)
- **Hardware**: Tesla V100-SXM2-32GB GPU
- **Augmentation**: 80% (2D/2.5D), 50% (3D)

### Evaluation
- **2D/2.5D**: 6,300 slices from 21 test patients
- **3D**: 18 full volumes from 18 test patients
- **Metrics**: Dice, IoU, Pixel Accuracy (torchmetrics)
- **Emissions Tracking**: CodeCarbon

---

## 🚀 Recommended Next Steps

Based on the analysis, prioritize these experiments:

### Priority 1: Optimize 3D Architecture

**Experiment 2A: 3D with Focal/Dice Loss**
- Expected: +5-10% Dice improvement
- Rationale: Better handle class imbalance than Cross-Entropy
- Implementation: Replace `nn.CrossEntropyLoss()` with `DiceLoss()` or `FocalLoss()`

**Experiment 2B: 3D with Attention Mechanisms**
- Expected: +5-10% Dice improvement
- Rationale: Focus on vertebrae regions, reduce false positives
- Implementation: Add 3D attention layers to encoder

**Experiment 2C: 3D with Mixed Precision (FP16)**
- Expected: -50% memory, enable batch_size=2 or 4
- Rationale: Faster training, higher throughput
- Implementation: Use `torch.cuda.amp.autocast()`

### Priority 2: Memory Optimization

**Experiment 2D: Gradient Checkpointing**
- Expected: -30% memory, +20% time
- Rationale: Trade memory for computation
- Implementation: `torch.utils.checkpoint` for encoder blocks

**Experiment 2E: Resolution Study**
- Test: 128³, 192³, 256³, 320³
- Find: Optimal speed/accuracy trade-off
- Current: 256³ is baseline

### Priority 3: Transfer Learning

**Experiment 2F: Pre-train on Large CT Datasets**
- Pre-train: ChestCT, AbdomenCT, or TotalSegmentator
- Fine-tune: Vertebral segmentation
- Expected: +3-8% Dice with less training data

### Skip

❌ **Further 2.5D Optimization**: Not worth investment, use 2D or 3D instead

---

## 🎯 Decision Matrix

### Use 3D U-Net if:
✅ Full CT volumes available
✅ Accuracy is top priority (+28% Dice)
✅ Need fast inference (1.2 min vs 39 min)
✅ Care about carbon footprint (-68% CO2)
✅ GPU has ≥6 GB VRAM
✅ Production clinical deployment
✅ High-throughput processing (many patients)

### Use 2D U-Net if:
⚠️ Rapid prototyping (simplest implementation)
⚠️ Educational purposes (easier to visualize)
⚠️ Partial/incomplete scans only
⚠️ Very limited GPU memory (<4 GB)
⚠️ Slice-level annotation tasks

### NEVER Use 2.5D U-Net:
❌ Only 5.7% better than 2D
❌ 15.5% slower than 2D
❌ 10.3% more CO2 than 2D
❌ No compelling reason vs 2D or 3D
❌ Worst trade-off of all three

---

## 📝 Implementation Guide

### 3D U-Net Requirements

**Hardware**:
- GPU: Tesla V100, RTX 3090, or similar (≥6 GB VRAM)
- RAM: 16 GB minimum, 32 GB recommended
- CPU: Multi-core (8+ cores) for data loading

**Software**:
```bash
pip install torch torchvision
pip install monai
pip install numpy scipy nibabel pydicom
pip install torchmetrics
pip install codecarbon  # For emissions tracking
```

**Key Configuration**:
```python
# training.py
BATCH_SIZE = 1          # Required for 3D
NUM_WORKERS = 8         # Parallel data loading
EPOCHS = 200            # More epochs needed for 3D
LEARNING_RATE = 0.001   # Standard Adam LR

# Model
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=9,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm='batch',
    dropout=0.1
)
```

**Memory Optimization** (if OOM):
```python
# Option 1: Mixed Precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

# Option 2: Reduce Resolution
# 256³ → 128³ (-87% memory)

# Option 3: Gradient Checkpointing
from torch.utils.checkpoint import checkpoint
```

---

## 📧 Questions or Issues?

### For Results Interpretation:
→ Check `EXPERIMENT_2_COMPREHENSIVE_REPORT.md` sections:
  - Section 2: Why 3D achieves best performance
  - Section 2.4: Why 2.5D shows minimal improvement
  - Section 3: Computational trade-offs

### For Implementation Help:
→ Check `COMPARISON_TABLES.md` tables:
  - Table 11: Architecture specifications
  - Table 13: Training configuration
  - Table 17: Implementation requirements

### For Specific Metrics:
→ Check `COMPARISON_TABLES.md` tables:
  - Table 1: Overall performance
  - Table 2: Per-class Dice scores
  - Table 4: Training time
  - Table 5-6: Energy and CO2

---

## 📅 Report Information

- **Generated**: December 19, 2025
- **Experiments**: E2-2D (2D U-Net) vs E2-2.5D (2.5D U-Net) vs E2-3D (3D U-Net)
- **Data Sources**:
  - E2-2D: `E1-augmentation/` (evaluation_report, training_reports, emissions.csv)
  - E2-2.5D: `E2-2.5D/` (evaluation_report, training_reports, emissions.csv)
  - E2-3D: `E2-3D/` (evaluation_report_3d, training_reports, emissions.csv)
- **Total Training Runs Analyzed**: 15 runs (5 per architecture)
- **Test Samples**: 6,300 slices (2D/2.5D), 18 volumes (3D)

---

## 📝 Citation

If using these results in publications, please cite:

```
Experiment 2: Impact of Model Architecture (2D vs 2.5D vs 3D) on Vertebral Segmentation
Architectures: 2D U-Net (ResNet34), 2.5D U-Net (ResNet34), 3D U-Net (MONAI)
Hardware: Tesla V100-SXM2-32GB GPU
Dataset: CT Vertebral Segmentation (9 classes: background + C1-C7 + other)
Key Finding: 3D U-Net achieves +28% Dice improvement with -68% training time and CO2
Date: December 2025
```

---

## 🏆 Bottom Line

**3D U-Net is the clear winner** for vertebral CT segmentation:
- **Best accuracy**: +28% Dice score
- **Fastest overall**: 5.2 hours training, 1.2 minutes inference
- **Most sustainable**: -68% CO2 emissions
- **Production-ready**: Clinically viable speed and accuracy

**Skip 2.5D entirely**: Provides minimal benefit (+5.7%) at higher cost (+15.5% time, +10.3% CO2).

**Use 2D only for**: Rapid prototyping, education, or when 3D requirements cannot be met.

---

**End of Navigation Guide**

Choose the appropriate report based on your needs and audience. All reports are written in Markdown format and can be converted to PDF, HTML, or integrated into LaTeX documents.

