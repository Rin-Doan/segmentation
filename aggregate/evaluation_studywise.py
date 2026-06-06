"""
Study-wise 3D U-Net evaluation for the aggregate pipeline.

For each validation volume, computes pixel accuracy, macro mean Dice, and macro mean IoU
on that study alone. Results are kept in a Python list and saved to CSV and JSON.

Pixel-wise pooling across all voxels remains in evaluation.py.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_process import Medical3DSegmentationDataset
from evaluation import (
    MODEL_PATH,
    NUM_CLASSES,
    SEGMENTATION_PATH,
    TARGET_SHAPE,
    TRAINING_PATH,
    calculate_metrics,
    device,
    discover_val_studies,
    load_model,
)

EVAL_DIR_STUDY = "evaluation_report_studywise"


def evaluate_studywise(model, val_dataset):
    """
    Run inference once per dataset index and record metrics for that study.

    Returns
    -------
    study_metrics_list : list of dict
        Each dict has study_id, pixel_accuracy, mean_dice, mean_iou (floats).
    """
    study_metrics_list = []
    model.eval()

    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset)), desc="Studies"):
            study_id = val_dataset.samples[idx]["study_id"]
            image, label = val_dataset[idx]
            image = image.unsqueeze(0).to(device)
            label = label.to(device)

            outputs = model(image)
            pred = torch.argmax(outputs, dim=1).squeeze(0)

            m = calculate_metrics(pred, label)
            study_metrics_list.append(
                {
                    "study_id": study_id,
                    "pixel_accuracy": float(m["pixel_accuracy"]),
                    "mean_dice": float(m["mean_dice"]),
                    "mean_iou": float(m["mean_iou"]),
                }
            )

    return study_metrics_list


def print_studywise_summary(study_metrics_list):
    acc = np.array([r["pixel_accuracy"] for r in study_metrics_list])
    dice = np.array([r["mean_dice"] for r in study_metrics_list])
    iou = np.array([r["mean_iou"] for r in study_metrics_list])

    print("=" * 80)
    print("STUDY-WISE SUMMARY (mean ± std over studies)")
    print("=" * 80)
    print(f"Studies: {len(study_metrics_list)}")
    print(f"Pixel accuracy: {acc.mean():.4f} ± {acc.std():.4f}")
    print(f"Mean Dice:      {dice.mean():.4f} ± {dice.std():.4f}")
    print(f"Mean IoU:       {iou.mean():.4f} ± {iou.std():.4f}")
    print("=" * 80)


def main():
    os.makedirs(EVAL_DIR_STUDY, exist_ok=True)

    print("=" * 80)
    print("3D U-NET — STUDY-WISE EVALUATION (one row per volume)")
    print("=" * 80)
    print()

    model = load_model(MODEL_PATH)

    print(f"Training images path: {TRAINING_PATH}")
    print(f"Segmentation path: {SEGMENTATION_PATH}")

    _, val_studies = discover_val_studies()
    print(f"Validation studies (same split as segmentation.py): {len(val_studies)}")

    val_dataset = Medical3DSegmentationDataset(
        study_ids=val_studies,
        training_path=TRAINING_PATH,
        segmentation_path=SEGMENTATION_PATH,
        target_shape=TARGET_SHAPE,
        augment=False,
    )
    print(f"✓ Validation dataset: {len(val_dataset)} volumes, shape {TARGET_SHAPE}\n")

    study_metrics_list = evaluate_studywise(model, val_dataset)

    csv_path = os.path.join(EVAL_DIR_STUDY, "studywise_metrics.csv")
    json_path = os.path.join(EVAL_DIR_STUDY, "studywise_metrics.json")

    pd.DataFrame(study_metrics_list).to_csv(csv_path, index=False, float_format="%.6f")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(study_metrics_list, f, indent=2)

    print(f"✓ Study-wise list saved: {csv_path}")
    print(f"✓ Study-wise list saved: {json_path}\n")

    print_studywise_summary(study_metrics_list)

    summary_path = os.path.join(EVAL_DIR_STUDY, "studywise_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"studies: {len(study_metrics_list)}\n")
        f.write(f"num_classes: {NUM_CLASSES}\n")
        acc = np.array([r["pixel_accuracy"] for r in study_metrics_list])
        dice = np.array([r["mean_dice"] for r in study_metrics_list])
        iou = np.array([r["mean_iou"] for r in study_metrics_list])
        f.write(f"pixel_accuracy_mean: {acc.mean():.6f}\n")
        f.write(f"pixel_accuracy_std: {acc.std():.6f}\n")
        f.write(f"mean_dice_mean: {dice.mean():.6f}\n")
        f.write(f"mean_dice_std: {dice.std():.6f}\n")
        f.write(f"mean_iou_mean: {iou.mean():.6f}\n")
        f.write(f"mean_iou_std: {iou.std():.6f}\n")
    print(f"✓ Summary written: {summary_path}")


if __name__ == "__main__":
    main()
