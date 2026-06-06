"""
Run the trained YOLOv8n-Pose model over every study and report the
first slice in which a vertebra is detected with confidence >= 0.80.

Output
------
A CSV file with columns ``StudyInstanceUID,slice_number`` where
``slice_number`` is 1-indexed and follows the InstanceNumber-sorted
DICOM order used throughout the project. If no slice exceeds the
confidence threshold the row records ``slice_number = 0``.
"""

import argparse
import csv
import os

import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm

from ultralytics import YOLO

import warnings
warnings.filterwarnings("ignore")


DATA_PATH = "../../../../../vast/s222440401"
TRAINING_PATH = os.path.join(DATA_PATH, "training_images")

HU_MIN, HU_MAX = -200.0, 1800.0


def parse_args():
    ap = argparse.ArgumentParser(description="YOLOv8n-Pose first slice inference")
    ap.add_argument("--weights", default="yolov8.pth",
                    help="Trained YOLO weights (produced by train_yolo.py)")
    ap.add_argument("--training-path", default=TRAINING_PATH,
                    help="Directory containing one subfolder of DICOMs per study")
    ap.add_argument("--output", default="yolo_inference_results.csv",
                    help="CSV to write: StudyInstanceUID,slice_number")
    ap.add_argument("--conf", type=float, default=0.80,
                    help="Confidence threshold (default 0.80)")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--device", default="0",
                    help="CUDA device id or 'cpu'")
    ap.add_argument("--batch", type=int, default=32,
                    help="Number of slices processed per YOLO forward pass")
    return ap.parse_args()


def sort_dicom_files(study_dir):
    """Return DICOM paths sorted by InstanceNumber (same as training)."""
    dicom_files = []
    for root, _, files in os.walk(study_dir):
        for f in files:
            if f.lower().endswith(".dcm"):
                dicom_files.append(os.path.join(root, f))

    meta = []
    for path in dicom_files:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            meta.append((int(getattr(ds, "InstanceNumber", 0)), path))
        except Exception:
            meta.append((0, path))
    meta.sort(key=lambda x: x[0])
    return [p for _, p in meta]


def dicom_slice_to_uint8(path):
    """Load one DICOM slice and return an HxWx3 uint8 image (YOLO expects RGB)."""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr * slope + intercept
    hu = np.clip(hu, HU_MIN, HU_MAX)
    hu = (hu - HU_MIN) / (HU_MAX - HU_MIN)
    img8 = (hu * 255.0).astype(np.uint8)
    # Replicate grayscale to 3 channels for YOLO's predictor
    return np.stack([img8, img8, img8], axis=-1)


def first_slice_for_study(model, study_dir, conf, imgsz, device, batch):
    """Return the 1-indexed first slice exceeding ``conf``; 0 if none."""
    paths = sort_dicom_files(study_dir)
    if not paths:
        return 0

    for start in range(0, len(paths), batch):
        chunk_paths = paths[start : start + batch]
        images = []
        valid_idx = []
        for k, p in enumerate(chunk_paths):
            try:
                images.append(dicom_slice_to_uint8(p))
                valid_idx.append(k)
            except Exception as e:
                print(f"  [warn] could not load {p}: {e}")

        if not images:
            continue

        results = model.predict(
            source=images,
            conf=conf,
            imgsz=imgsz,
            device=device,
            max_det=1,
            verbose=False,
        )

        for local_k, res in zip(valid_idx, results):
            boxes = getattr(res, "boxes", None)
            if boxes is None or boxes.conf is None or len(boxes.conf) == 0:
                continue
            if float(boxes.conf.max().item()) >= conf:
                # 1-indexed global slice number
                return start + local_k + 1
    return 0


def main():
    args = parse_args()

    if not os.path.exists(args.weights):
        raise FileNotFoundError(
            f"Weights not found: {args.weights}. "
            "Run train_yolo.py first."
        )
    if not os.path.isdir(args.training_path):
        raise FileNotFoundError(f"Training directory not found: {args.training_path}")

    studies = sorted([d for d in os.listdir(args.training_path)
                      if os.path.isdir(os.path.join(args.training_path, d))])
    print(f"Loaded {len(studies)} studies from {args.training_path}")

    print(f"Loading YOLO model from {args.weights}")
    model = YOLO(args.weights)

    with open(args.output, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["StudyInstanceUID", "slice_number"])

        for sid in tqdm(studies, desc="Inferring first slice"):
            study_dir = os.path.join(args.training_path, sid)
            try:
                slice_num = first_slice_for_study(
                    model, study_dir, args.conf, args.imgsz, args.device, args.batch
                )
            except Exception as e:
                print(f"  [error] {sid}: {e}")
                slice_num = 0
            writer.writerow([sid, slice_num])

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
