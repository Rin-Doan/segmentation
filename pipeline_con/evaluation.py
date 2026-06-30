"""
Evaluate the full pipeline (pipeline.py) against ground-truth segmentations.

Predictions are the multiclass vertebra-level label maps written to ML_DATA_PATH
by pipeline.py (0=background, 1-7=C1-C7, 8=other), shaped (256, 256, 256).

To compare fairly, the raw NIfTI ground truth in SEGMENTATION_PATH is pushed
through the *same* spatial transform the prediction went through:

    load + orientation correction  (aggregate_data.load_nifti_volume)
    -> resample to TARGET_SPACING   (order=0, label-preserving)
    -> trim [first_slice:]          (the YOLO offset pipeline.py saved per study)
    -> front z-crop/pad to TARGET_SHAPE (same as preprocess_image)
    -> label remap (>7 -> 8)        (matches bm_segmentation_z/data_process.py)

Only the held-out validation split is scored (test_size=0.2, random_state=42),
mirroring bm_segmentation_z/evaluation_ml.py so we evaluate studies the models
were not trained on.

Run on a GPU node via the project venv (from the pipeline/ directory):

    uv run evaluation.py
"""

import csv
import os
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassJaccardIndex,
)
from tqdm import tqdm

from aggregate.aggregate_data import (
    TARGET_SPACING,
    load_nifti_volume,
    resample_to_standard_spacing_3d,
)
from aggregate.data_process import front_crop_or_pad_to_size

warnings.filterwarnings("ignore")

# Ground-truth source (raw, un-resampled segmentations).
DATA_PATH = "/vast/s222440401"
SEGMENTATION_PATH = DATA_PATH + "/segmentations"

# Pipeline predictions (multiclass label maps from pipeline.py).
PIPELINE_DATA_PATH = DATA_PATH + "/pipeline"
ML_DATA_PATH = PIPELINE_DATA_PATH + "/ml_data"

# Per-study YOLO trim offset written by pipeline.py.
FIRST_SLICE_CSV = "./first_slices.csv"
RESULTS_CSV = "./evaluation_results.csv"

TARGET_SHAPE = (256, 256, 256)
NUM_CLASSES = 9
RANDOM_STATE = 42

# Class names: 0=background, 1-7=C1-C7, 8=other (T1 and below).
CLASS_NAMES = ["background", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "other"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dice_per_class = MulticlassF1Score(num_classes=NUM_CLASSES, average="none").to(device)
iou_per_class = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average="none").to(device)
acc_per_class = MulticlassAccuracy(num_classes=NUM_CLASSES, average="none").to(device)
dice_macro = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro").to(device)
iou_macro = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average="macro").to(device)
acc_micro = MulticlassAccuracy(num_classes=NUM_CLASSES, average="micro").to(device)


def get_study_id(filename: str) -> str:
    return filename.replace(".nii.gz", "").replace(".nii", "")


def find_file(directory: str, study_id: str) -> str | None:
    """Return the .nii / .nii.gz path for a study in a directory, if present."""
    for ext in (".nii", ".nii.gz"):
        path = os.path.join(directory, f"{study_id}{ext}")
        if os.path.exists(path):
            return path
    return None


def load_first_slices(csv_path: str = FIRST_SLICE_CSV) -> dict[str, int]:
    """Load the per-study YOLO trim offset saved by pipeline.py."""
    offsets: dict[str, int] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            offsets[row["study_id"]] = int(row["first_slice"])
    return offsets


def discover_val_studies() -> tuple[list[str], list[str]]:
    """Studies with both a prediction and ground truth; same split as training."""
    pred_ids = {
        get_study_id(f)
        for f in os.listdir(ML_DATA_PATH)
        if f.endswith((".nii", ".nii.gz"))
    }
    gt_ids = {
        get_study_id(f)
        for f in os.listdir(SEGMENTATION_PATH)
        if f.endswith((".nii", ".nii.gz"))
    }
    overlapping = sorted(pred_ids & gt_ids)
    _, val_studies = train_test_split(
        overlapping, test_size=0.2, random_state=RANDOM_STATE
    )
    return overlapping, val_studies


def load_prediction(study_id: str) -> torch.Tensor:
    """Load the multiclass prediction (256, 256, 256) written by pipeline.py."""
    path = find_file(ML_DATA_PATH, study_id)
    if path is None:
        raise FileNotFoundError(f"Prediction not found for {study_id}")
    pred = nib.load(path).get_fdata().astype(np.int64)
    return torch.from_numpy(pred).to(device)


def load_ground_truth(study_id: str, first_slice: int) -> torch.Tensor:
    """Transform the raw GT segmentation into the prediction's grid.

    Mirrors the spatial pipeline applied to the image: orientation correction +
    resample to TARGET_SPACING + the same YOLO z-trim + front z-crop/pad, then the
    label remap used at training time.
    """
    seg_path = find_file(SEGMENTATION_PATH, study_id)
    if seg_path is None:
        raise FileNotFoundError(f"Ground truth not found for {study_id}")

    # Orientation correction + header spacing, then resample to model spacing.
    seg_volume, spacing = load_nifti_volume(seg_path)
    seg_volume = resample_to_standard_spacing_3d(
        seg_volume, spacing, TARGET_SPACING, order=0
    )

    # Apply the same z-trim YOLO applied to the image before segmentation.
    if 0 < first_slice < seg_volume.shape[0]:
        seg_volume = seg_volume[first_slice:]

    seg_volume = front_crop_or_pad_to_size(seg_volume, TARGET_SHAPE)

    # Remap: keep 0-7 (background + C1-C7), group 8+ -> 8 (matches data_process.py).
    seg_volume = np.where(seg_volume > 7, 8, seg_volume).astype(np.int64)
    return torch.from_numpy(seg_volume).to(device)


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Per-volume metrics; pred/target shaped (D, H, W) with class indices."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    dice_scores = dice_per_class(pred_flat, target_flat).cpu().numpy()
    iou_scores = iou_per_class(pred_flat, target_flat).cpu().numpy()
    class_acc = acc_per_class(pred_flat, target_flat).cpu().numpy()
    dice_avg = dice_macro(pred_flat, target_flat).item()
    iou_avg = iou_macro(pred_flat, target_flat).item()
    pixel_acc = acc_micro(pred_flat, target_flat).item()

    dice_per_class.reset()
    iou_per_class.reset()
    acc_per_class.reset()
    dice_macro.reset()
    iou_macro.reset()
    acc_micro.reset()

    metrics = {
        "accuracy": pixel_acc,
        "dice_score": dice_avg,
        "iou": iou_avg,
        # Mean over foreground classes (1-8), ignoring background.
        "foreground_accuracy": float(class_acc[1:].mean()),
        "foreground_dice_score": float(dice_scores[1:].mean()),
        "foreground_iou": float(iou_scores[1:].mean()),
    }

    for cls in range(NUM_CLASSES):
        name = CLASS_NAMES[cls]
        metrics[f"acc_{cls}_{name}"] = float(class_acc[cls])
        metrics[f"dice_{cls}_{name}"] = float(dice_scores[cls])
        metrics[f"iou_{cls}_{name}"] = float(iou_scores[cls])

    return metrics


def evaluate(val_studies: list[str], first_slices: dict[str, int]) -> list[dict]:
    """Compare each validation prediction against its aligned ground truth."""
    rows = []
    for study_id in tqdm(val_studies, desc="Evaluating volumes"):
        if study_id not in first_slices:
            print(f"  Skipping {study_id}: no first-slice offset recorded")
            continue
        try:
            pred = load_prediction(study_id)
            target = load_ground_truth(study_id, first_slices[study_id])
        except FileNotFoundError as e:
            print(f"  Skipping {study_id}: {e}")
            continue

        metrics = calculate_metrics(pred, target)
        metrics["study_id"] = study_id
        rows.append(metrics)
    return rows


def aggregate_metrics(rows: list[dict]) -> dict:
    """Mean overall, foreground, and per-class metrics over all evaluated volumes."""
    df = pd.DataFrame(rows)
    summary = {
        "num_volumes": len(df),
        "accuracy": df["accuracy"].mean(),
        "dice_score": df["dice_score"].mean(),
        "iou": df["iou"].mean(),
        "foreground_accuracy": df["foreground_accuracy"].mean(),
        "foreground_dice_score": df["foreground_dice_score"].mean(),
        "foreground_iou": df["foreground_iou"].mean(),
    }
    for cls in range(NUM_CLASSES):
        name = CLASS_NAMES[cls]
        summary[f"acc_{cls}_{name}"] = df[f"acc_{cls}_{name}"].mean()
        summary[f"dice_{cls}_{name}"] = df[f"dice_{cls}_{name}"].mean()
        summary[f"iou_{cls}_{name}"] = df[f"iou_{cls}_{name}"].mean()
    return summary


def save_results_csv(rows: list[dict], save_path: str = RESULTS_CSV) -> None:
    summary = aggregate_metrics(rows)
    pd.DataFrame([summary]).to_csv(save_path, index=False, float_format="%.6f")


def print_summary(rows: list[dict]) -> None:
    agg = aggregate_metrics(rows)
    print("\n" + "=" * 60)
    print("PIPELINE EVALUATION SUMMARY (validation set)")
    print("=" * 60)
    print(f"Volumes evaluated: {agg['num_volumes']}")
    print("\nOverall (macro Dice/IoU, pixel accuracy):")
    print(f"  Accuracy:   {agg['accuracy']:.4f}")
    print(f"  Dice Score: {agg['dice_score']:.4f}")
    print(f"  IoU:        {agg['iou']:.4f}")
    print("\nForeground / vertebrae (mean over classes 1-8):")
    print(f"  Accuracy:   {agg['foreground_accuracy']:.4f}")
    print(f"  Dice Score: {agg['foreground_dice_score']:.4f}")
    print(f"  IoU:        {agg['foreground_iou']:.4f}")
    print("\nPer-class:")
    print(f"  {'Class':<14} {'Acc':>8} {'Dice':>8} {'IoU':>8}")
    for cls in range(NUM_CLASSES):
        name = CLASS_NAMES[cls]
        label = f"{cls} ({name})"
        print(
            f"  {label:<14}"
            f" {agg[f'acc_{cls}_{name}']:>8.4f}"
            f" {agg[f'dice_{cls}_{name}']:>8.4f}"
            f" {agg[f'iou_{cls}_{name}']:>8.4f}"
        )
    print("=" * 60)


def main() -> None:
    print("=" * 60)
    print("Pipeline Evaluation — ml_data vs ground-truth segmentations")
    print("=" * 60)
    print(f"Device:        {device}")
    print(f"Predictions:   {ML_DATA_PATH}")
    print(f"Ground truth:  {SEGMENTATION_PATH}")
    print(f"First slices:  {FIRST_SLICE_CSV}")
    print(f"Shape:         {TARGET_SHAPE}")
    print()

    if not os.path.isfile(FIRST_SLICE_CSV):
        raise FileNotFoundError(
            f"First-slice CSV not found: {FIRST_SLICE_CSV} (run pipeline.py first)"
        )

    first_slices = load_first_slices()
    overlapping, val_studies = discover_val_studies()
    print(f"Overlapping studies: {len(overlapping)}")
    print(
        f"Validation studies:  {len(val_studies)} "
        f"(test_size=0.2, random_state={RANDOM_STATE})\n"
    )

    rows = evaluate(val_studies, first_slices)
    if not rows:
        print("No volumes evaluated; nothing to report.")
        return

    save_results_csv(rows, RESULTS_CSV)
    print(f"\nAverage metrics saved to: {RESULTS_CSV}")
    print_summary(rows)


if __name__ == "__main__":
    main()
