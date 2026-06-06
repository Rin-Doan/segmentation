"""
Train a YOLOv8n-Pose model for vertebra detection.

The preparation step (prepare_dataset.py) produced a YOLO-format dataset
at ``$VAST/yolo_first_slice_dataset/dataset.yaml`` with one class
(``vertebra``) and a single keypoint per box (segmentation centroid).

This script fine-tunes the official ``yolov8n-pose.pt`` checkpoint on
that dataset and copies the best weights to ``yolov8.pth`` in the
current directory.
"""

import argparse
import os
import shutil
from pathlib import Path

from ultralytics import YOLO


DATA_PATH = "../../../../../vast/s222440401"
DATASET_YAML = os.path.join(DATA_PATH, "yolo_first_slice_dataset", "dataset.yaml")


def parse_args():
    ap = argparse.ArgumentParser(description="Train YOLOv8n-Pose for vertebra first-slice detection")
    ap.add_argument("--data", default=DATASET_YAML,
                    help="Path to dataset.yaml produced by prepare_dataset.py")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0",
                    help="CUDA device id, 'cpu', or comma-separated list for multi-GPU")
    ap.add_argument("--project", default="runs_first_slice",
                    help="Ultralytics project directory")
    ap.add_argument("--name", default="yolov8n_pose_vertebra",
                    help="Ultralytics run name")
    ap.add_argument("--weights", default="yolov8n-pose.pt",
                    help="Starting checkpoint")
    ap.add_argument("--output", default="yolov8.pth",
                    help="Final weights file copied into this directory")
    return ap.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Dataset YAML not found at {args.data}. "
            "Run prepare_dataset.py first."
        )

    print("=" * 60)
    print("Training YOLOv8n-Pose for vertebra detection")
    print("=" * 60)
    print(f"Data       : {args.data}")
    print(f"Weights    : {args.weights}")
    print(f"Image size : {args.imgsz}")
    print(f"Epochs     : {args.epochs}")
    print(f"Batch      : {args.batch}")
    print(f"Device     : {args.device}")

    model = YOLO(args.weights)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=25,
        optimizer="AdamW",
        lr0=1e-3,
        cos_lr=True,
        pose=12.0,
        kobj=2.0,
        # CT slices are anatomically symmetric along vertical axis only.
        fliplr=0.5,
        flipud=0.0,
        degrees=10.0,
        translate=0.05,
        scale=0.2,
        mosaic=0.0,
        close_mosaic=0,
        plots=True,
    )

    # Locate the best checkpoint produced by Ultralytics
    save_dir = Path(getattr(results, "save_dir",
                            os.path.join(args.project, args.name)))
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        # Fallback: last weights if best wasn't written
        best = save_dir / "weights" / "last.pt"

    if not best.exists():
        raise RuntimeError(f"Could not find trained weights under {save_dir}")

    dest = Path(args.output).resolve()
    shutil.copy2(best, dest)
    print(f"\nBest weights   : {best}")
    print(f"Copied to      : {dest}")
    print("=" * 60)
    print("Training complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
