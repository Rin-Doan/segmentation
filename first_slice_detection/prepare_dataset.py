"""
Build a YOLOv8n-Pose dataset for vertebra detection from CT scans.

Pipeline
--------
1.  Find overlapping studies that have both a DICOM series in
    `TRAINING_PATH` and a segmentation mask in `SEGMENTATION_PATH`.
2.  For every study, load the DICOM volume sorted by InstanceNumber and
    the NIfTI segmentation with the same coordinate correction used in
    `aggregate/data_process.py` (transpose + axis flips).
3.  For every 2D axial slice:
      - apply HU windowing and convert to an 8-bit PNG image,
      - if the segmentation slice contains vertebra pixels, compute a
        tight bounding box (left/right/top/bottom most pixels) and the
        segmentation centroid as a single keypoint,
      - write the YOLO-Pose label file (empty for negative slices).
4.  Split studies (not slices) 80/20 into train/val so there is no
    leakage between splits.
5.  Emit a `dataset.yaml` that YOLOv8 can consume directly.
"""

import os
import sys
import random
import shutil
from pathlib import Path

import numpy as np
import nibabel as nib
import pydicom
from PIL import Image
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------
DATA_PATH = "../../../../../vast/s222440401"
TRAINING_PATH = os.path.join(DATA_PATH, "training_images")
SEGMENTATION_PATH = os.path.join(DATA_PATH, "segmentations")

# Dataset output directory (kept local so YOLO can read it quickly)
DATASET_ROOT = os.path.join(DATA_PATH, "yolo_first_slice_dataset")

IMG_DIRS = {
    "train": os.path.join(DATASET_ROOT, "images", "train"),
    "val":   os.path.join(DATASET_ROOT, "images", "val"),
}
LBL_DIRS = {
    "train": os.path.join(DATASET_ROOT, "labels", "train"),
    "val":   os.path.join(DATASET_ROOT, "labels", "val"),
}

# HU window used everywhere else in the project
HU_MIN, HU_MAX = -200.0, 1800.0

# Train/val split on study level
VAL_FRACTION = 0.2
SEED = 42

# Single-class vertebra detector with one centroid keypoint
NUM_KEYPOINTS = 1


# ---------------------------------------------------------------------------
# Volume loading helpers (mirrors aggregate/data_process.py + aggregate_data.py)
# ---------------------------------------------------------------------------
def load_dicom_volume(study_dir):
    """Return a (D, H, W) float32 volume sorted by InstanceNumber."""
    dicom_files = []
    for root, _, files in os.walk(study_dir):
        for f in files:
            if f.lower().endswith(".dcm"):
                dicom_files.append(os.path.join(root, f))

    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files under {study_dir}")

    meta = []
    for path in dicom_files:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            meta.append((int(getattr(ds, "InstanceNumber", 0)), path))
        except Exception:
            meta.append((0, path))
    meta.sort(key=lambda x: x[0])

    slices = []
    for _, path in meta:
        try:
            ds = pydicom.dcmread(path)
            arr = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            slices.append(arr * slope + intercept)
        except Exception as e:
            print(f"  [warn] could not load {path}: {e}")

    if not slices:
        raise RuntimeError(f"No readable DICOM slices under {study_dir}")

    # Some studies have slices with different shapes; pad to the max in-plane.
    max_h = max(s.shape[0] for s in slices)
    max_w = max(s.shape[1] for s in slices)
    if any(s.shape != (max_h, max_w) for s in slices):
        padded = []
        for s in slices:
            canvas = np.full((max_h, max_w), HU_MIN, dtype=np.float32)
            canvas[: s.shape[0], : s.shape[1]] = s
            padded.append(canvas)
        slices = padded

    return np.stack(slices, axis=0)


def load_seg_volume(seg_path):
    """Return a (D, H, W) int32 segmentation aligned to the DICOM volume."""
    nii = nib.load(seg_path)
    seg = nii.get_fdata()
    # Same convention used in aggregate/data_process.py and aggregate_data.py
    seg = seg[:, ::-1, ::-1].transpose(2, 1, 0)
    return seg.astype(np.int32)


def hu_to_uint8(slice_hu):
    """HU -> [0,255] uint8 image using the project-wide window."""
    s = np.clip(slice_hu, HU_MIN, HU_MAX)
    s = (s - HU_MIN) / (HU_MAX - HU_MIN)
    return (s * 255.0).astype(np.uint8)


def bbox_and_centroid_from_mask(mask):
    """
    Given a 2D binary mask, return ((x_min, y_min, x_max, y_max), (cx, cy)).
    Coordinates are in pixel units. Returns None if the mask is empty.
    """
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    cx = float(xs.mean())
    cy = float(ys.mean())
    return (x_min, y_min, x_max, y_max), (cx, cy)


def yolo_pose_label_line(bbox, centroid, img_w, img_h):
    """Format a single YOLOv8-Pose line: class cx cy w h kx ky v."""
    x_min, y_min, x_max, y_max = bbox
    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    bw = (x_max - x_min + 1) / img_w
    bh = (y_max - y_min + 1) / img_h
    kx = centroid[0] / img_w
    ky = centroid[1] / img_h
    # class id 0 (vertebra), visibility 2 (labeled + visible)
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {kx:.6f} {ky:.6f} 2"


# ---------------------------------------------------------------------------
# Main preparation routine
# ---------------------------------------------------------------------------
def find_overlapping_studies():
    studies_img = {d for d in os.listdir(TRAINING_PATH)
                   if os.path.isdir(os.path.join(TRAINING_PATH, d))}
    studies_seg = set()
    for f in os.listdir(SEGMENTATION_PATH):
        if f.endswith(".nii.gz"):
            studies_seg.add(f[:-7])
        elif f.endswith(".nii"):
            studies_seg.add(f[:-4])
    return sorted(studies_img & studies_seg)


def seg_path_for(study_id):
    p1 = os.path.join(SEGMENTATION_PATH, f"{study_id}.nii")
    p2 = os.path.join(SEGMENTATION_PATH, f"{study_id}.nii.gz")
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    return None


def prepare_splits(studies):
    random.Random(SEED).shuffle(studies)
    n_val = max(1, int(round(len(studies) * VAL_FRACTION)))
    val = sorted(studies[:n_val])
    train = sorted(studies[n_val:])
    return {"train": train, "val": val}


def reset_dataset_dirs():
    if os.path.exists(DATASET_ROOT):
        shutil.rmtree(DATASET_ROOT)
    for d in list(IMG_DIRS.values()) + list(LBL_DIRS.values()):
        os.makedirs(d, exist_ok=True)


def process_study(study_id, split):
    """Write images + label files for every 2D slice of one study."""
    study_dir = os.path.join(TRAINING_PATH, study_id)
    sp = seg_path_for(study_id)
    if sp is None:
        return 0, 0

    try:
        volume = load_dicom_volume(study_dir)
        seg = load_seg_volume(sp)
    except Exception as e:
        print(f"  [skip] {study_id}: {e}")
        return 0, 0

    D = min(volume.shape[0], seg.shape[0])
    H, W = volume.shape[1], volume.shape[2]

    # If the seg volume's in-plane size differs from the CT, skip (rare).
    if seg.shape[1] != H or seg.shape[2] != W:
        print(f"  [skip] {study_id}: shape mismatch img={volume.shape}, seg={seg.shape}")
        return 0, 0

    positives, negatives = 0, 0
    for i in range(D):
        img8 = hu_to_uint8(volume[i])
        mask = seg[i] > 0

        stem = f"{study_id}__slice_{i:04d}"
        img_path = os.path.join(IMG_DIRS[split], stem + ".png")
        lbl_path = os.path.join(LBL_DIRS[split], stem + ".txt")

        Image.fromarray(img8, mode="L").save(img_path, optimize=True)

        if mask.any():
            bbox_cent = bbox_and_centroid_from_mask(mask)
            if bbox_cent is None:
                # Technically unreachable because mask.any() is True
                open(lbl_path, "w").close()
                negatives += 1
                continue
            bbox, centroid = bbox_cent
            line = yolo_pose_label_line(bbox, centroid, W, H)
            with open(lbl_path, "w") as f:
                f.write(line + "\n")
            positives += 1
        else:
            open(lbl_path, "w").close()
            negatives += 1

    return positives, negatives


def write_yaml(splits):
    yaml_path = os.path.join(DATASET_ROOT, "dataset.yaml")
    # flip: ultralytics pose expects keypoint indices for horizontal flip
    # remapping. A single keypoint stays at index 0.
    content = (
        f"path: {os.path.abspath(DATASET_ROOT)}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: vertebra\n"
        f"kpt_shape: [{NUM_KEYPOINTS}, 3]\n"
        f"flip_idx: [0]\n"
    )
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"Wrote {yaml_path}")
    return yaml_path


def main():
    print("=" * 60)
    print("Preparing YOLOv8n-Pose first-slice detection dataset")
    print("=" * 60)
    print(f"Training images : {os.path.abspath(TRAINING_PATH)}")
    print(f"Segmentations   : {os.path.abspath(SEGMENTATION_PATH)}")
    print(f"Dataset output  : {os.path.abspath(DATASET_ROOT)}")

    studies = find_overlapping_studies()
    print(f"Overlapping studies: {len(studies)}")
    if not studies:
        print("No overlapping studies found; aborting.")
        sys.exit(1)

    splits = prepare_splits(studies)
    print(f"  train: {len(splits['train'])} studies")
    print(f"  val  : {len(splits['val'])} studies")

    reset_dataset_dirs()

    totals = {"train": [0, 0], "val": [0, 0]}
    for split, study_ids in splits.items():
        for sid in tqdm(study_ids, desc=f"{split} studies"):
            pos, neg = process_study(sid, split)
            totals[split][0] += pos
            totals[split][1] += neg

    print("\nDataset built:")
    for split, (pos, neg) in totals.items():
        print(f"  {split}: {pos} positive slices, {neg} negative slices")

    write_yaml(splits)
    print("=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
