"""Train a YOLOv11 pose model to localise vertebrae on axial CT slices.

Dataset: /vast/s222440401/yolo_first_slice_dataset
    images/{train,val}/*.png   512x512 grayscale axial CT slices
    labels/{train,val}/*.txt   YOLO-pose labels, one line per object:
        <class> <cx> <cy> <w> <h> <kpt_x> <kpt_y> <visibility>

    - class:        always 0 ("vertebra"); single class.
    - cx cy w h:    bounding box of the vertebra (normalised 0-1).
    - kpt_x kpt_y:  the vertebra centre keypoint (normalised 0-1).
    - visibility:   2 = visible.

    dataset.yaml declares `kpt_shape: [1, 3]`, i.e. 1 keypoint with (x, y, vis),
    so this is a pose task: the model learns to both detect the vertebra's
    bounding box and regress its centre point. Most slices contain a single
    vertebra; ~19% are background (empty label files) and act as negatives.

Run on the cluster with the project venv (ultralytics + torch are already
installed there):

    cd first_slice_detection
    uv run train.py                       # sensible defaults
    uv run train.py --model yolo11m-pose.pt --epochs 150 --batch 32

Submit via SLURM with a `--gres=gpu:v100:1` job (see the other *.sh scripts).
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

# dataset.yaml lives at the dataset root and already points `path` at the
# mounted dataset directory, so we default to it directly.
DEFAULT_DATA = "/vast/s222440401/yolo_first_slice_dataset/dataset.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 pose to localise vertebrae on CT slices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help="Path to the YOLO dataset.yaml.",
    )
    parser.add_argument(
        "--model",
        default="yolo11s-pose.pt",
        help="Pretrained YOLOv11 pose checkpoint to fine-tune "
        "(e.g. yolo11n-pose.pt, yolo11s-pose.pt, yolo11m-pose.pt).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=512,
        help="Training image size; matches the native 512x512 slices.",
    )
    parser.add_argument(
        "--batch",
        type=float,
        default=16,
        help="Batch size (use -1 for ultralytics auto-batch by GPU memory).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="CUDA device(s), e.g. '0' or '0,1'. Defaults to auto-detect.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early-stopping patience (epochs without improvement).",
    )
    parser.add_argument(
        "--project",
        default=str(Path(__file__).resolve().parent / "runs"),
        help="Directory to store training runs.",
    )
    parser.add_argument("--name", default="vertebra_pose")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the most recent run with this name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset config not found: {data_path}\n"
            "Pass the correct path with --data /path/to/dataset.yaml"
        )

    print(f"Loading model:   {args.model}")
    print(f"Dataset config:  {data_path}")
    model = YOLO(args.model)

    # CT slices are single-channel and contain a single small vertebra centred
    # near the spine, so we keep colour/geometry augmentation conservative:
    # no hue/saturation shifts (grayscale), no mosaic (it fragments the single
    # object), small rotations/translations only, and no vertical flip.
    model.train(
        data=str(data_path),
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        resume=args.resume,
        pretrained=True,
        optimizer="auto",
        cos_lr=True,
        # augmentation tuned for grayscale single-vertebra CT slices
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,
        degrees=10.0,
        translate=0.1,
        scale=0.3,
        shear=0.0,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.0,
        plots=True,
        verbose=True,
    )

    # Evaluate the best weights on the validation split and report metrics.
    print("\nRunning validation on the best checkpoint...")
    metrics = model.val(split="val")
    print("\nValidation results:")
    print(f"  Box  mAP50    : {metrics.box.map50:.4f}")
    print(f"  Box  mAP50-95 : {metrics.box.map:.4f}")
    print(f"  Pose mAP50    : {metrics.pose.map50:.4f}")
    print(f"  Pose mAP50-95 : {metrics.pose.map:.4f}")

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nBest weights saved to: {best}")


if __name__ == "__main__":
    main()
