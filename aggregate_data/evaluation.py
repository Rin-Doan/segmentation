"""
Evaluate best_unet3d_model.pth on the validation split.

Preprocessing matches Medical3DSegmentationDataset in data_process.py (NIfTI from
agg_data_1, HU windowing, crop/pad, binary labels). Study discovery and train/val
split match segmentation.py.

For each validation volume, runs one forward pass, then writes mean overall and
foreground (vertebrae) Accuracy, Dice Score, and IoU to a CSV (one summary row).
"""

import os

import numpy as np
import pandas as pd
import torch
from monai.networks.nets import UNet
from scipy.ndimage import center_of_mass, label
from sklearn.model_selection import train_test_split
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassJaccardIndex,
)
from tqdm import tqdm
import warnings

from data_process import Medical3DSegmentationDataset

warnings.filterwarnings("ignore")

# Configuration — align with aggregate_work_1/segmentation.py
DATA_PATH = "/vast/s222440401"
TRAINING_PATH = DATA_PATH + "/agg_data/images_nii"
SEGMENTATION_PATH = DATA_PATH + "/agg_data/segmentations_nii"
MODEL_PATH = "best_unet3d_model.pth"
TARGET_SHAPE = (256, 256, 256)
NUM_CLASSES = 2
RESULTS_CSV = "evaluation_inference_results.csv"
RANDOM_STATE = 42

# 6-connectivity post-processing thresholds
MIN_COMPONENT_SIZE = 500    # voxels; components smaller than this are candidates for removal
MAX_ISOLATION_DISTANCE = 50.0  # voxel units; centroid distance beyond which a small component is removed

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

    return {
        "accuracy": pixel_acc,
        "dice_score": dice_avg,
        "iou": iou_avg,
        "foreground_accuracy": float(class_acc[1]),
        "foreground_dice_score": float(dice_scores[1]),
        "foreground_iou": float(iou_scores[1]),
    }


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


def filter_small_isolated_components(
    pred: torch.Tensor,
    min_size: int = MIN_COMPONENT_SIZE,
    max_distance: float = MAX_ISOLATION_DISTANCE,
) -> torch.Tensor:
    """
    6-connectivity post-processing on a 3D prediction volume.

    Foreground (class 1) voxels are labelled into connected components using
    6-connectivity (face-adjacent only, no diagonals).  Any component that is
    BOTH smaller than `min_size` voxels AND whose centroid is farther than
    `max_distance` voxels from the largest component's centroid is relabelled
    as background (class 0).

    Args:
        pred: integer tensor of shape (D, H, W) with class indices.
        min_size: voxel count below which a component is a removal candidate.
        max_distance: Euclidean centroid distance above which a small component
                      is considered isolated and removed.

    Returns:
        Filtered prediction tensor with the same shape and device as `pred`.
    """
    pred_np = pred.cpu().numpy().astype(np.uint8)
    foreground = pred_np == 1

    # Face-adjacent 6-connectivity structure (no diagonals)
    struct_6 = np.zeros((3, 3, 3), dtype=np.int32)
    struct_6[1, 1, :] = 1
    struct_6[1, :, 1] = 1
    struct_6[:, 1, 1] = 1

    labeled_arr, num_components = label(foreground, structure=struct_6)

    if num_components <= 1:
        return pred

    # Component sizes (index 0 = background, excluded)
    component_sizes = np.bincount(labeled_arr.ravel())
    component_sizes[0] = 0

    largest_label = int(component_sizes.argmax())
    largest_centroid = np.array(center_of_mass(foreground, labeled_arr, largest_label))

    remove_mask = np.zeros_like(foreground, dtype=bool)
    for comp_label in range(1, num_components + 1):
        if comp_label == largest_label:
            continue
        if component_sizes[comp_label] < min_size:
            centroid = np.array(center_of_mass(foreground, labeled_arr, comp_label))
            if np.linalg.norm(centroid - largest_centroid) > max_distance:
                remove_mask |= labeled_arr == comp_label

    result = pred.clone()
    result[torch.from_numpy(remove_mask).to(pred.device)] = 0
    return result


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
            pred = filter_small_isolated_components(pred)
            metrics = calculate_metrics(pred, label)
            rows.append(metrics)

    return rows


def aggregate_metrics(rows: list[dict]) -> dict:
    """Mean overall and foreground metrics over all validation volumes."""
    df = pd.DataFrame(rows)
    return {
        "num_volumes": len(df),
        "accuracy": df["accuracy"].mean(),
        "dice_score": df["dice_score"].mean(),
        "iou": df["iou"].mean(),
        "foreground_accuracy": df["foreground_accuracy"].mean(),
        "foreground_dice_score": df["foreground_dice_score"].mean(),
        "foreground_iou": df["foreground_iou"].mean(),
    }


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
    print("\nForeground / vertebrae (class 1):")
    print(f"  Accuracy:   {agg['foreground_accuracy']:.4f}")
    print(f"  Dice Score: {agg['foreground_dice_score']:.4f}")
    print(f"  IoU:        {agg['foreground_iou']:.4f}")
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
