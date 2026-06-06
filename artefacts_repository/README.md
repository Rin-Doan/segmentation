# Artefacts repository (handover)

This folder is a **handover bundle** for work on **vertebral (spine) segmentation** on the **RSNA 2022 cervical spine fracture** imaging setup: literature, manuscript drafts, pointers to data and analysis notes, and runnable experiment code together with **logged metrics and reports**. Raw DICOM/NIfTI volumes and trained checkpoints are **not** included here unless added separately.

---

## Top-level layout

| Folder | Contents |
|--------|----------|
| `1.Literature_review` | Bibliography (`Literature_review.bib`), review document (`Literature_review.docx`), and PDFs under `paper/`. |
| `2.Research_paper` | Versioned drafts of the research paper (`versions/*.docx`), titled along the lines of *A Comparative Analysis of Training Techniques for Vertebral Segmentation Using RSNA2022*. |
| `3.Data_and_Analysis` | `data_link.docx` (dataset / access notes) and `analysis_script.docx` (analysis description). |
| `4.Artefacts` | Code and experiment outputs for segmentation experiments (see below). |

---

## Segmentation experiments (`4.Artefacts/segmentation_experiment/`)

Each subdirectory is a **self-contained experiment**: training script, data loading, evaluation, optional loss definitions, **CodeCarbon** emissions logging, and **CSV / text reports** under `training_reports/` and `evaluation_report/` (E2-3D uses `evaluation_report_3d/`).

**Model:** U-Net with **ResNet34** encoder (`segmentation_models_pytorch`), **9 classes**, grayscale input.

| ID | Focus |
|----|--------|
| **E1-no** | Baseline training **without** heavy augmentation (`AUGMENT = False`). |
| **E1-augmentation** | Same setup **with** augmentation enabled. |
| **E2-2.5D** | **2.5D** slice configuration (multi-slice context as used in that run). |
| **E2-3D** | **3D** variant; includes `convergence.csv` in training reports where generated. |
| **E3-Dice** | **Dice**-type segmentation loss. |
| **E3-CE-Dice** | **Cross-entropy + Dice** combined loss. |
| **E3-Focal** | **Focal** loss variant. |

Typical artefacts per run:

- `training_reports/` — e.g. `training_efficiency.csv`, `inference.csv`, `inference_emissions.csv` (and `convergence.csv` for some configs).
- `evaluation_report/` — `evaluation_report.txt`, `metrics_per_class.csv`.
- `emissions.csv` — training-related carbon / energy trace from CodeCarbon (when enabled).

Training scripts save `best_unet_model.pth` and plots (e.g. `training_curves.png`) **when you run training**; those binary artefacts are **not** checked into this handover tree by default.

---

## Reproducing or extending runs

1. **Data paths** — In each experiment’s `training.py` (and matching `evaluation.py` / `data_process.py` if paths differ), `DATA_PATH` and related variables point to a **machine-specific** location (e.g. a shared `vast/...` directory). Update these to your local paths for `training_images` and `segmentations` (NIfTI masks).

2. **Environment** — Python with **PyTorch** (CUDA recommended), `segmentation_models_pytorch`, `torchmetrics`, `scikit-learn`, `matplotlib`, `tqdm`, `numpy`, and **CodeCarbon** for emission logs. Align versions with your cluster or workstation policy.

3. **Order of use** — From the experiment folder: run `training.py` to train (and emit reports), then `evaluation.py` for test metrics and figures. `training_report.py` / `computational_metrics.py` / `analyze_class_distribution.py` support reporting and diagnostics as wired in each folder.

4. **Duplicates** — Some folders contain a `training copy.py` file; treat `training.py` as the primary entry point unless your notes say otherwise.

---
