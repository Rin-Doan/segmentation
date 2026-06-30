"""
Evaluate best_unet3d_model.pth on the validation split.

Preprocessing matches Medical3DSegmentationDataset in data_process.py (NIfTI from
agg_data_1, HU windowing, crop/pad, binary labels). Study discovery and train/val
split match segmentation.py.

For each validation volume, runs one forward pass, then writes mean overall and
foreground (vertebrae) Accuracy, Dice Score, and IoU to a CSV (one summary row).
"""

import os

import pandas as pd
import torch
from monai.networks.nets import UNet
from sklearn.model_selection import train_test_split
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassJaccardIndex,
)
from tqdm import tqdm
import warnings

import nibabel as nib

from data_process import Medical3DSegmentationDataset, crop_or_pad_to_size

warnings.filterwarnings("ignore")

# Configuration — align with aggregate_work_1/segmentation.py
DATA_PATH = "/vast/s222440401"
TRAINING_PATH = DATA_PATH + "/bm_data/bm_images_nii"
SEGMENTATION_PATH = DATA_PATH + "/bm_data/bm_segmentations_nii"
MASK_PATH = DATA_PATH + "/bm_data/bm_masks_nii"
MODEL_PATH = "best_unet3d_model_ml.pth"
TARGET_SHAPE = (256, 256, 256)
NUM_CLASSES = 9
RESULTS_CSV = "evaluation_inference_results_ml.csv"
RANDOM_STATE = 42

# Class names: 0=background, 1-7=C1-C7, 8=other (T1 and below)
CLASS_NAMES = ["background", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "other"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: str = MODEL_PATH) -> torch.nn.Module:
    """Load trained MONAI 3D U-Net (same architecture as segmentation.py)."""
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=NUM_CLASSES,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="batch",
        dropout=0.1,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


dice_per_class = MulticlassF1Score(num_classes=NUM_CLASSES, average="none").to(device)
iou_per_class = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average="none").to(device)
acc_per_class = MulticlassAccuracy(num_classes=NUM_CLASSES, average="none").to(device)
dice_macro = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro").to(device)
iou_macro = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average="macro").to(device)
acc_micro = MulticlassAccuracy(num_classes=NUM_CLASSES, average="micro").to(device)


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Per-volume metrics; pred/target shaped (D, H, W) with class indices."""
    if pred.dim() == 4:
        pred = pred.squeeze(0)
    if target.dim() == 4:
        target = target.squeeze(0)

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
        # Mean over foreground classes (1-8), ignoring background
        "foreground_accuracy": float(class_acc[1:].mean()),
        "foreground_dice_score": float(dice_scores[1:].mean()),
        "foreground_iou": float(iou_scores[1:].mean()),
    }

    # Per-class metrics for every class (0=background, 1-7=C1-C7, 8=other)
    for cls in range(NUM_CLASSES):
        name = CLASS_NAMES[cls]
        metrics[f"acc_{cls}_{name}"] = float(class_acc[cls])
        metrics[f"dice_{cls}_{name}"] = float(dice_scores[cls])
        metrics[f"iou_{cls}_{name}"] = float(iou_scores[cls])

    return metrics


def discover_val_studies() -> tuple[list[str], list[str]]:
    """Same study discovery and train/val split as segmentation.py."""
    image_files = [f for f in os.listdir(TRAINING_PATH) if f.endswith((".nii", ".nii.gz"))]
    seg_files = [f for f in os.listdir(SEGMENTATION_PATH) if f.endswith((".nii", ".nii.gz"))]

    def get_study_id(filename: str) -> str:
        return filename.replace(".nii.gz", "").replace(".nii", "")

    image_study_ids = {get_study_id(f): f for f in image_files}
    seg_study_ids = {get_study_id(f): f for f in seg_files}
    overlapping = sorted(set(image_study_ids.keys()) & set(seg_study_ids.keys()))
    _, val_studies = train_test_split(
        overlapping, test_size=0.2, random_state=RANDOM_STATE
    )
    return overlapping, val_studies


def load_foreground_mask(study_id: str, target_shape=TARGET_SHAPE) -> torch.Tensor:
    """Load the saved binary mask for a study, aligned to the model input grid.

    Uses the same crop/pad as data_process.py so it matches the image and label.
    """
    path_nii = os.path.join(MASK_PATH, f"{study_id}.nii")
    path = path_nii if os.path.exists(path_nii) else os.path.join(MASK_PATH, f"{study_id}.nii.gz")
    mask = nib.load(path).get_fdata()
    mask = crop_or_pad_to_size(mask, target_shape)
    return torch.from_numpy(mask > 0).to(device)


def run_inference(
    model: torch.nn.Module, val_dataset: Medical3DSegmentationDataset
) -> list[dict]:
    """Run one forward pass per validation volume and collect metrics."""
    rows = []
    model.eval()

    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset)), desc="Evaluating volumes"):
            image, label = val_dataset[idx]
            image = image.unsqueeze(0).to(device)
            label = label.to(device)

            outputs = model(image)
            pred = torch.argmax(outputs, dim=1).squeeze(0)

            # Re-impose the known background using the real binary mask (Option 2):
            # force every voxel outside the vertebra mask to class 0.
            study_id = val_dataset.samples[idx]["study_id"]
            foreground = load_foreground_mask(study_id)
            pred[~foreground] = 0

            metrics = calculate_metrics(pred, label)
            rows.append(metrics)

    return rows


def aggregate_metrics(rows: list[dict]) -> dict:
    """Mean overall, foreground, and per-class metrics over all validation volumes."""
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
    # Per-class means (0=background, 1-7=C1-C7, 8=other)
    for cls in range(NUM_CLASSES):
        name = CLASS_NAMES[cls]
        summary[f"acc_{cls}_{name}"] = df[f"acc_{cls}_{name}"].mean()
        summary[f"dice_{cls}_{name}"] = df[f"dice_{cls}_{name}"].mean()
        summary[f"iou_{cls}_{name}"] = df[f"iou_{cls}_{name}"].mean()
    return summary


def save_results_csv(rows: list[dict], save_path: str = RESULTS_CSV) -> pd.DataFrame:
    summary = aggregate_metrics(rows)
    df = pd.DataFrame([summary])
    df.to_csv(save_path, index=False, float_format="%.6f")
    return df


def print_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    agg = aggregate_metrics(rows)
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (validation set)")
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
    print("3D U-Net Evaluation — aggregate_work_1")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model:  {MODEL_PATH}")
    print(f"Images: {TRAINING_PATH}")
    print(f"Labels: {SEGMENTATION_PATH}")
    print(f"Shape:  {TARGET_SHAPE}")
    print()

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

    model = load_model(MODEL_PATH)
    print(f"Loaded model ({sum(p.numel() for p in model.parameters()):,} parameters)\n")

    overlapping, val_studies = discover_val_studies()
    print(f"Overlapping studies: {len(overlapping)}")
    print(f"Validation studies:  {len(val_studies)} (test_size=0.2, random_state={RANDOM_STATE})\n")

    val_dataset = Medical3DSegmentationDataset(
        study_ids=val_studies,
        training_path=TRAINING_PATH,
        segmentation_path=SEGMENTATION_PATH,
        target_shape=TARGET_SHAPE,
        augment=False,
    )
    print(f"Validation samples: {len(val_dataset)}\n")

    rows = run_inference(model, val_dataset)
    save_results_csv(rows, RESULTS_CSV)
    print(f"\nAverage metrics saved to: {RESULTS_CSV}")
    print_summary(rows)


if __name__ == "__main__":
    main()
