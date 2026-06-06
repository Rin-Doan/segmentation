"""
Pixel-wise 3D segmentation evaluation for the aggregate training pipeline.

Uses the same backbones as ``segmentation_tune.py`` (see ``segmentation_models.py``).
Pools predictions across all validation voxels for aggregate metrics, plots, and reports.
For one row per study (volume), use evaluation_studywise.py instead.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import pandas as pd
import warnings

from data_process import Medical3DSegmentationDataset
from segmentation_models import ARCH_CHOICES, build_segmentation_model, default_checkpoint_path

warnings.filterwarnings("ignore")

from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassAccuracy,
)

# Configuration — align with aggregate/segmentation.py
DATA_PATH = "../../../../../vast/s222440401"
TRAINING_PATH = DATA_PATH + "/agg_data/images_nii"
SEGMENTATION_PATH = DATA_PATH + "/agg_data/segmentations_nii"
BATCH_SIZE = 1
NUM_CLASSES = 9
TARGET_SHAPE = (128, 256, 256)
# Defaults when running as a script (overridden by CLI / run_evaluation).
DEFAULT_MODEL_ARCH = "3dunet"
LEGACY_UNET_CHECKPOINT = "best_unet3d_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


def load_model(arch: str, model_path: str):
    """Load checkpoint for ``arch`` (same builder as ``segmentation_tune.py``)."""
    print(f"Loading {arch} model...")
    model = build_segmentation_model(
        arch, spatial_dims=3, in_channels=1, out_channels=NUM_CLASSES
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✓ {arch} loaded from {model_path}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    return model


dice_per_class = MulticlassF1Score(num_classes=NUM_CLASSES, average="none").to(device)
iou_per_class = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average="none").to(device)
acc_per_class = MulticlassAccuracy(num_classes=NUM_CLASSES, average="none").to(device)

dice_macro = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro").to(device)
iou_macro = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average="macro").to(device)
acc_micro = MulticlassAccuracy(num_classes=NUM_CLASSES, average="micro").to(device)


def calculate_metrics(pred, target):
    """Per-volume metrics (torchmetrics); pred/target as in training loop."""
    if pred.dim() == 4:
        pred = pred.squeeze(0)
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
        "dice_per_class": dice_scores,
        "iou_per_class": iou_scores,
        "class_accuracy": class_acc,
        "mean_dice": dice_avg,
        "mean_iou": iou_avg,
        "pixel_accuracy": pixel_acc,
    }


def evaluate_model(model, dataloader, num_classes=NUM_CLASSES):
    model.eval()

    all_dice_scores = []
    all_iou_scores = []
    all_pixel_acc = []
    all_class_acc = []

    all_preds = []
    all_targets = []

    print("Evaluating 3D model on validation set...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Processing 3D volumes"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for pred, target in zip(preds, labels):
                metrics = calculate_metrics(pred, target)

                all_dice_scores.append(metrics["dice_per_class"])
                all_iou_scores.append(metrics["iou_per_class"])
                all_pixel_acc.append(metrics["pixel_accuracy"])
                all_class_acc.append(metrics["class_accuracy"])

                if len(all_preds) < 100000:
                    all_preds.extend(pred.cpu().numpy().flatten()[::50])
                    all_targets.extend(target.cpu().numpy().flatten()[::50])

    results = {
        "dice_scores": np.array(all_dice_scores),
        "iou_scores": np.array(all_iou_scores),
        "pixel_accuracy": np.array(all_pixel_acc),
        "class_accuracy": np.array(all_class_acc),
        "predictions": np.array(all_preds),
        "targets": np.array(all_targets),
    }

    print(f"✓ Evaluation complete on {len(all_pixel_acc)} 3D volumes\n")
    return results


def plot_sample_predictions(model, dataset, num_samples=6, save_path="sample_predictions.png"):
    print(f"Generating sample predictions from 3D volumes ({num_samples} samples)...")
    model.eval()

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            image, label = dataset[sample_idx]
            image_input = image.unsqueeze(0).to(device)

            output = model(image_input)
            pred = torch.argmax(output, dim=1).squeeze(0)

            metrics = calculate_metrics(pred, label.to(device))

            mid_slice = image.shape[1] // 2

            img_np = image[0, mid_slice].cpu().numpy()
            label_np = label.squeeze(0)[mid_slice].cpu().numpy()
            pred_np = pred[mid_slice].cpu().numpy()

            axes[idx, 0].imshow(img_np, cmap="gray")
            axes[idx, 0].set_title(f"Input (slice {mid_slice})", fontsize=12, fontweight="bold")
            axes[idx, 0].axis("off")

            axes[idx, 1].imshow(label_np, cmap="tab20", vmin=0, vmax=8)
            axes[idx, 1].set_title("Ground Truth", fontsize=12, fontweight="bold")
            axes[idx, 1].axis("off")

            axes[idx, 2].imshow(pred_np, cmap="tab20", vmin=0, vmax=8)
            axes[idx, 2].set_title(
                f'Prediction\nAcc: {metrics["pixel_accuracy"]:.3f} | Dice: {metrics["mean_dice"]:.3f}',
                fontsize=12,
                fontweight="bold",
            )
            axes[idx, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"✓ Sample predictions saved to '{save_path}'\n")
    plt.close()


def plot_metrics_per_class(dice_scores, iou_scores, class_acc, save_path="metrics_per_class.png"):
    print("Generating per-class metrics plot...")

    mean_dice = np.nanmean(dice_scores, axis=0)
    mean_iou = np.nanmean(iou_scores, axis=0)
    mean_class_acc = np.nanmean(class_acc, axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    classes = range(NUM_CLASSES)

    axes[0].bar(classes, mean_dice, color="steelblue", alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Class", fontsize=12)
    axes[0].set_ylabel("Dice Score", fontsize=12)
    axes[0].set_title("Mean Dice Score per Class", fontsize=14, fontweight="bold")
    axes[0].set_xticks(classes)
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].set_ylim([0, 1])
    for i, v in enumerate(mean_dice):
        if not np.isnan(v):
            axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    axes[1].bar(classes, mean_iou, color="coral", alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("Class", fontsize=12)
    axes[1].set_ylabel("IoU Score", fontsize=12)
    axes[1].set_title("Mean IoU Score per Class", fontsize=14, fontweight="bold")
    axes[1].set_xticks(classes)
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].set_ylim([0, 1])
    for i, v in enumerate(mean_iou):
        if not np.isnan(v):
            axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    axes[2].bar(classes, mean_class_acc, color="seagreen", alpha=0.7, edgecolor="black")
    axes[2].set_xlabel("Class", fontsize=12)
    axes[2].set_ylabel("Accuracy", fontsize=12)
    axes[2].set_title("Mean Accuracy per Class", fontsize=14, fontweight="bold")
    axes[2].set_xticks(classes)
    axes[2].grid(True, alpha=0.3, axis="y")
    axes[2].set_ylim([0, 1])
    for i, v in enumerate(mean_class_acc):
        if not np.isnan(v):
            axes[2].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"✓ Per-class metrics saved to '{save_path}'\n")
    plt.close()


def plot_confusion_matrix(targets, predictions, save_path="confusion_matrix.png"):
    print("Generating confusion matrix...")

    cm = confusion_matrix(targets, predictions, labels=range(NUM_CLASSES))
    cm_normalized = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    im1 = axes[0].imshow(cm, cmap="Blues", aspect="auto")
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=16, fontweight="bold")
    axes[0].set_xlabel("Predicted Label", fontsize=12)
    axes[0].set_ylabel("True Label", fontsize=12)
    axes[0].set_xticks(range(NUM_CLASSES))
    axes[0].set_yticks(range(NUM_CLASSES))
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(cm_normalized, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=16, fontweight="bold")
    axes[1].set_xlabel("Predicted Label", fontsize=12)
    axes[1].set_ylabel("True Label", fontsize=12)
    axes[1].set_xticks(range(NUM_CLASSES))
    axes[1].set_yticks(range(NUM_CLASSES))
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"✓ Confusion matrix saved to '{save_path}'\n")
    plt.close()


def save_evaluation_report(results, save_path="evaluation_report.txt", architecture: str = "3dunet"):
    print("Generating evaluation report...")

    dice_scores = results["dice_scores"]
    iou_scores = results["iou_scores"]
    pixel_acc = results["pixel_accuracy"]
    class_acc = results["class_accuracy"]

    mean_dice = np.nanmean(dice_scores, axis=0)
    std_dice = np.nanstd(dice_scores, axis=0)
    mean_iou = np.nanmean(iou_scores, axis=0)
    std_iou = np.nanstd(iou_scores, axis=0)
    mean_pixel_acc = np.mean(pixel_acc)
    std_pixel_acc = np.std(pixel_acc)
    mean_class_acc = np.nanmean(class_acc, axis=0)

    with open(save_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("3D SEGMENTATION — PIXEL-WISE EVALUATION (aggregate pipeline)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Model architecture: {architecture}\n")
        f.write("Data: NIfTI volumes (Medical3DSegmentationDataset)\n")
        f.write("Metrics: torchmetrics (F1Score=Dice, JaccardIndex=IoU)\n")
        f.write(f"Number of test volumes: {len(pixel_acc)}\n")
        f.write(f"Number of classes: {NUM_CLASSES} (0=background, 1-7=C1-C7, 8=other)\n\n")

        f.write("OVERALL METRICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean Pixel Accuracy:     {mean_pixel_acc:.6f} ± {std_pixel_acc:.6f}\n")
        f.write(f"Mean Dice Score (macro): {np.nanmean(mean_dice):.6f}\n")
        f.write(f"Mean IoU Score (macro):  {np.nanmean(mean_iou):.6f}\n\n")

        f.write("PER-CLASS METRICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<8} {'Dice':<20} {'IoU':<20} {'Accuracy':<12}\n")
        f.write("-" * 80 + "\n")

        for cls in range(NUM_CLASSES):
            dice_str = (
                f"{mean_dice[cls]:.4f}±{std_dice[cls]:.4f}"
                if not np.isnan(mean_dice[cls])
                else "N/A"
            )
            iou_str = (
                f"{mean_iou[cls]:.4f}±{std_iou[cls]:.4f}"
                if not np.isnan(mean_iou[cls])
                else "N/A"
            )
            acc_str = f"{mean_class_acc[cls]:.4f}" if not np.isnan(mean_class_acc[cls]) else "N/A"
            f.write(f"{cls:<8} {dice_str:<20} {iou_str:<20} {acc_str:<12}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY:\n")
        f.write("=" * 80 + "\n")
        valid_dice = [d for d in mean_dice if not np.isnan(d)]
        valid_iou = [i for i in mean_iou if not np.isnan(i)]
        if valid_dice:
            f.write(
                f"Best class (Dice):  Class {np.nanargmax(mean_dice)} ({np.nanmax(mean_dice):.4f})\n"
            )
            f.write(
                f"Worst class (Dice): Class {np.nanargmin(mean_dice)} ({np.nanmin(mean_dice):.4f})\n"
            )
        if valid_iou:
            f.write(
                f"Best class (IoU):   Class {np.nanargmax(mean_iou)} ({np.nanmax(mean_iou):.4f})\n"
            )
            f.write(
                f"Worst class (IoU):  Class {np.nanargmin(mean_iou)} ({np.nanmin(mean_iou):.4f})\n"
            )

    print(f"✓ Evaluation report saved to '{save_path}'\n")

    print("=" * 80)
    print("EVALUATION SUMMARY (pixel-wise / mean over volumes)")
    print("=" * 80)
    print(f"Samples evaluated:     {len(pixel_acc)}")
    print(f"Mean Pixel Accuracy:   {mean_pixel_acc:.4f} ± {std_pixel_acc:.4f}")
    print(f"Mean Dice Score:       {np.nanmean(mean_dice):.4f}")
    print(f"Mean IoU Score:        {np.nanmean(mean_iou):.4f}")
    print("=" * 80)
    print()


def save_metrics_csv(results, save_path="metrics_per_class.csv"):
    print("Saving metrics to CSV...")

    mean_dice = np.nanmean(results["dice_scores"], axis=0)
    mean_iou = np.nanmean(results["iou_scores"], axis=0)
    mean_class_acc = np.nanmean(results["class_accuracy"], axis=0)

    class_counts = [0] * NUM_CLASSES
    for target in results["targets"]:
        if 0 <= target < NUM_CLASSES:
            class_counts[int(target)] += 1

    df = pd.DataFrame(
        {
            "Class": range(NUM_CLASSES),
            "Dice_Score": mean_dice,
            "IoU_Score": mean_iou,
            "Accuracy": mean_class_acc,
            "Sample_Count": class_counts,
        }
    )

    df.to_csv(save_path, index=False, float_format="%.6f")
    print(f"✓ Metrics CSV saved to '{save_path}'\n")


def discover_val_studies():
    """Same study discovery and train/val split as aggregate/segmentation.py."""
    image_files = [f for f in os.listdir(TRAINING_PATH) if f.endswith((".nii", ".nii.gz"))]
    seg_files = [f for f in os.listdir(SEGMENTATION_PATH) if f.endswith((".nii", ".nii.gz"))]

    def get_study_id(filename):
        return filename.replace(".nii.gz", "").replace(".nii", "")

    image_study_ids = {get_study_id(f): f for f in image_files}
    seg_study_ids = {get_study_id(f): f for f in seg_files}

    overlapping_studies = sorted(
        list(set(image_study_ids.keys()).intersection(set(seg_study_ids.keys())))
    )
    _, val_studies = train_test_split(overlapping_studies, test_size=0.2, random_state=42)
    return overlapping_studies, val_studies


def parse_eval_cli():
    parser = argparse.ArgumentParser(
        description="Pixel-wise validation evaluation (same train/val study split as segmentation_tune.py)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ARCH,
        choices=ARCH_CHOICES,
        help=f"Backbone matching training: {', '.join(ARCH_CHOICES)}.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Weights path. Default: best_<model>_3d_seg.pth from segmentation_tune. "
        f"Legacy aggregate/segmentation.py U-Net: use --model 3dunet --checkpoint {LEGACY_UNET_CHECKPOINT}.",
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=None,
        help="Output directory for plots and reports. Default: evaluation_report_<model>.",
    )
    return parser.parse_args()


def run_evaluation(
    arch: str,
    checkpoint_path: str | None = None,
    eval_dir: str | None = None,
):
    """
    Run full pixel-wise evaluation for one backbone.

    Args:
        arch: One of ``ARCH_CHOICES`` (same as ``--model`` in segmentation_tune.py).
        checkpoint_path: Path to ``state_dict``. Defaults to ``default_checkpoint_path(arch)``.
        eval_dir: Output directory; default ``evaluation_report_<arch>``.
    """
    arch = arch.strip().lower()
    if arch not in ARCH_CHOICES:
        raise ValueError(f"Unknown architecture {arch!r}. Choose one of: {ARCH_CHOICES}")

    ckpt = checkpoint_path or default_checkpoint_path(arch)
    out_dir = eval_dir or f"evaluation_report_{arch}"

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print(f"3D SEGMENTATION — PIXEL-WISE EVALUATION ({arch})")
    print("=" * 80)
    print()

    model = load_model(arch, ckpt)

    print(f"Training images path: {TRAINING_PATH}")
    print(f"Segmentation path: {SEGMENTATION_PATH}")

    overlapping_studies, val_studies = discover_val_studies()
    print(f"Found {len(overlapping_studies)} overlapping studies")
    print(f"Validation studies (same split as segmentation_tune.py): {len(val_studies)}")

    val_dataset = Medical3DSegmentationDataset(
        study_ids=val_studies,
        training_path=TRAINING_PATH,
        segmentation_path=SEGMENTATION_PATH,
        target_shape=TARGET_SHAPE,
        augment=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"✓ Validation dataset: {len(val_dataset)} volumes, shape {TARGET_SHAPE}")
    print()

    results = evaluate_model(model, val_loader, num_classes=NUM_CLASSES)

    plot_sample_predictions(
        model,
        val_dataset,
        num_samples=8,
        save_path=os.path.join(out_dir, "sample_predictions.png"),
    )
    plot_metrics_per_class(
        results["dice_scores"],
        results["iou_scores"],
        results["class_accuracy"],
        save_path=os.path.join(out_dir, "metrics_per_class.png"),
    )
    plot_confusion_matrix(
        results["targets"],
        results["predictions"],
        save_path=os.path.join(out_dir, "confusion_matrix.png"),
    )

    save_evaluation_report(
        results,
        save_path=os.path.join(out_dir, "evaluation_report.txt"),
        architecture=arch,
    )
    save_metrics_csv(results, save_path=os.path.join(out_dir, "metrics_per_class.csv"))

    np.savez(
        os.path.join(out_dir, "evaluation_metrics.npz"),
        dice_scores=results["dice_scores"],
        iou_scores=results["iou_scores"],
        pixel_accuracy=results["pixel_accuracy"],
        class_accuracy=results["class_accuracy"],
    )
    print(f"✓ Raw metrics saved to '{os.path.join(out_dir, 'evaluation_metrics.npz')}'\n")

    print("=" * 80)
    print("PIXEL-WISE EVALUATION COMPLETED")
    print("=" * 80)
    print(f"Output directory: {out_dir}/")
    print("=" * 80)


def main():
    args = parse_eval_cli()
    run_evaluation(args.model, args.checkpoint, args.eval_dir)


if __name__ == "__main__":
    main()
