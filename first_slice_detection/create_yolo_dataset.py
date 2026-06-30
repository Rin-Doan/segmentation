"""Build the YOLO-pose dataset for vertebra localisation on axial CT slices.

For every study present in both ``training_images`` and ``segmentations``:

    1. Load the DICOM series as a (D, H, W) HU volume (sorted by InstanceNumber).
    2. Load the matching NIfTI segmentation with the same coordinate correction
       used in ``aggregate_data.py``.
    3. Export every axial slice as a 512x512 grayscale PNG:
           pixel = (clip(HU, -200, 1800) + 200) / 2000 * 255
    4. Derive a YOLO-pose label from the segmentation mask on that slice:
           - bounding box = axis-aligned extent (left, right, top, bottom)
           - keypoint     = mask centroid
       Slices with no foreground voxels get an empty label file (background).

Output layout (compatible with ``train.py``):

    /vast/s222440401/yolo_first_slice_dataset/
        dataset.yaml
        images/{train,val}/*.png
        labels/{train,val}/*.txt

Run:

    cd first_slice_detection
    uv run create_yolo_dataset.py
    uv run create_yolo_dataset.py --val-frac 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATA_PATH = "/vast/s222440401"
TRAINING_PATH = DATA_PATH + "/training_images"
SEGMENTATION_PATH = DATA_PATH + "/segmentations"
OUTPUT_PATH = DATA_PATH + "/yolo_first_slice_dataset"

HU_LOW, HU_HIGH = -200.0, 1800.0
IMGSZ = 512
CLASS_ID = 0
KEYPOINT_VISIBILITY = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the YOLO-pose vertebra dataset from DICOM + segmentations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--training-path", default=None)
    parser.add_argument("--segmentation-path", default=None)
    parser.add_argument("--output-path", default=OUTPUT_PATH)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Fraction of studies held out for validation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing images/ and labels/ under the output path first.",
    )
    return parser.parse_args()


def discover_overlapping_studies(training_path: Path, segmentation_path: Path) -> list[str]:
    """Return study IDs present in both training_images and segmentations."""
    training_studies = {p.name for p in training_path.iterdir() if p.is_dir()}
    segmentation_studies = set()
    for seg_path in segmentation_path.glob("*.nii*"):
        name = seg_path.name
        if name.endswith(".nii.gz"):
            segmentation_studies.add(name[: -len(".nii.gz")])
        elif name.endswith(".nii"):
            segmentation_studies.add(name[: -len(".nii")])
    return sorted(training_studies & segmentation_studies)


def load_dicom_volume(study_dir: Path) -> np.ndarray:
    """Load a DICOM series as a (D, H, W) HU volume sorted by InstanceNumber."""
    dicom_files = [path for path in study_dir.rglob("*.dcm") if path.is_file()]
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {study_dir}")

    try:
        dicom_meta = []
        for dcm_path in dicom_files:
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
            instance_num = getattr(ds, "InstanceNumber", 0)
            dicom_meta.append((instance_num, dcm_path))
        dicom_meta.sort(key=lambda x: x[0])
        sorted_dicom_paths = [path for _, path in dicom_meta]
    except Exception:
        sorted_dicom_paths = sorted(dicom_files)

    slices = []
    for dcm_path in sorted_dicom_paths:
        ds = pydicom.dcmread(str(dcm_path))
        slice_data = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        slices.append(slice_data * slope + intercept)

    if not slices:
        raise ValueError(f"No valid DICOM slices found in {study_dir}")

    return np.stack(slices, axis=0)


def load_nifti_segmentation(seg_path: Path) -> np.ndarray:
    """Load a NIfTI segmentation volume with aggregate_data coordinate correction."""
    seg_volume = nib.load(str(seg_path)).get_fdata()
    seg_volume = seg_volume[:, ::-1, ::-1].transpose(2, 1, 0)
    return seg_volume.astype(np.int32)


def resolve_segmentation_path(segmentation_path: Path, study_id: str) -> Path | None:
    for name in (f"{study_id}.nii", f"{study_id}.nii.gz"):
        candidate = segmentation_path / name
        if candidate.is_file():
            return candidate
    return None


def hu_slice_to_png(slice_2d: np.ndarray, imgsz: int) -> Image.Image:
    """Window-normalise an axial HU slice to a grayscale 512x512 PNG."""
    windowed = np.clip(slice_2d, HU_LOW, HU_HIGH)
    scaled = (windowed - HU_LOW) / (HU_HIGH - HU_LOW) * 255.0
    return Image.fromarray(scaled.astype(np.uint8)).resize((imgsz, imgsz))


def resize_mask(mask_2d: np.ndarray, imgsz: int) -> np.ndarray:
    """Resize a boolean mask with nearest-neighbour interpolation."""
    image = Image.fromarray((mask_2d.astype(np.uint8) * 255))
    resized = np.array(image.resize((imgsz, imgsz), resample=Image.NEAREST))
    return resized > 0


def mask_to_pose_label(mask: np.ndarray, imgsz: int) -> str | None:
    """Return a YOLO-pose label line for the mask, or None if empty."""
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None

    left, right = int(cols.min()), int(cols.max())
    top, bottom = int(rows.min()), int(rows.max())

    cx = (left + right) / 2.0 / imgsz
    cy = (top + bottom) / 2.0 / imgsz
    w = (right - left + 1) / imgsz
    h = (bottom - top + 1) / imgsz
    kpt_x = float(cols.mean()) / imgsz
    kpt_y = float(rows.mean()) / imgsz

    return (
        f"{CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} "
        f"{kpt_x:.6f} {kpt_y:.6f} {KEYPOINT_VISIBILITY}"
    )


def write_dataset_yaml(output_path: Path) -> None:
    yaml_path = output_path / "dataset.yaml"
    content = (
        f"path: {output_path}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: vertebra\n"
        "kpt_shape: [1, 3]\n"
        "flip_idx: [0]\n"
    )
    yaml_path.write_text(content)


def prepare_output_dirs(output_path: Path, overwrite: bool) -> None:
    for subdir in ("images/train", "images/val", "labels/train", "labels/val"):
        path = output_path / subdir
        if overwrite and path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def process_study(
    study_id: str,
    split: str,
    training_path: Path,
    segmentation_path: Path,
    output_path: Path,
    imgsz: int,
) -> tuple[int, int]:
    """Export all slices for one study. Returns (num_slices, num_labelled_slices)."""
    image_volume = load_dicom_volume(training_path / study_id)
    seg_path = resolve_segmentation_path(segmentation_path, study_id)
    if seg_path is None:
        raise FileNotFoundError(f"No segmentation found for {study_id}")

    seg_volume = load_nifti_segmentation(seg_path)
    if image_volume.shape[0] != seg_volume.shape[0]:
        raise ValueError(
            f"Depth mismatch for {study_id}: "
            f"image {image_volume.shape[0]} vs segmentation {seg_volume.shape[0]}"
        )

    images_dir = output_path / "images" / split
    labels_dir = output_path / "labels" / split
    labelled = 0

    for z in range(image_volume.shape[0]):
        stem = f"{study_id}__slice_{z:04d}"
        png = hu_slice_to_png(image_volume[z], imgsz)
        png.save(images_dir / f"{stem}.png")

        mask = resize_mask(seg_volume[z] > 0, imgsz)
        label_line = mask_to_pose_label(mask, imgsz)
        label_path = labels_dir / f"{stem}.txt"
        if label_line is None:
            label_path.write_text("")
        else:
            label_path.write_text(label_line + "\n")
            labelled += 1

    return image_volume.shape[0], labelled


def main() -> None:
    args = parse_args()

    training_path = Path(args.training_path or args.data_path + "/training_images")
    segmentation_path = Path(args.segmentation_path or args.data_path + "/segmentations")
    output_path = Path(args.output_path)

    if not training_path.is_dir():
        raise FileNotFoundError(f"Training path not found: {training_path}")
    if not segmentation_path.is_dir():
        raise FileNotFoundError(f"Segmentation path not found: {segmentation_path}")

    study_ids = discover_overlapping_studies(training_path, segmentation_path)
    if not study_ids:
        raise RuntimeError("No overlapping studies found between training images and segmentations")

    train_ids, val_ids = train_test_split(
        study_ids,
        test_size=args.val_frac,
        random_state=args.seed,
    )

    prepare_output_dirs(output_path, overwrite=args.overwrite)
    write_dataset_yaml(output_path)

    print(f"Found {len(study_ids)} studies")
    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}")
    print(f"Writing dataset to: {output_path}")

    totals = {"slices": 0, "labelled": 0}
    for split, ids in (("train", train_ids), ("val", val_ids)):
        for study_id in tqdm(ids, desc=f"{split}"):
            num_slices, num_labelled = process_study(
                study_id,
                split,
                training_path,
                segmentation_path,
                output_path,
                args.imgsz,
            )
            totals["slices"] += num_slices
            totals["labelled"] += num_labelled

    background = totals["slices"] - totals["labelled"]
    print("\nDataset creation complete.")
    print(f"  Total slices : {totals['slices']}")
    print(f"  With labels  : {totals['labelled']}")
    print(f"  Background   : {background}")
    print(f"  Config       : {output_path / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
