"""Find the first vertebra slice of each study with the trained YOLOv11 model.

For every study that has both DICOM images in ``training_images`` and a
segmentation in ``segmentations``, this script:

    1. Loads the DICOM series as a 3D volume (shape (D, H, W); axis 0 is the
       z / axial axis, sorted by InstanceNumber). Pixel values are converted
       to HU via RescaleSlope / RescaleIntercept.
    2. Converts each axial slice to a 512x512 uint8 image using exactly the
       same windowing/normalisation used to build the YOLO training set
           pixel = (clip(HU, -200, 1800) + 200) / 2000 * 255
       (verified to reproduce the training PNGs to the byte).
    3. Runs the trained YOLOv11-pose detector over the slices and marks a slice
       as "vertebra present" when it contains a detection with confidence
       above the threshold (default 0.80).
    4. Scans down the z axis for the longest run of present slices (tolerating
       a few isolated sub-threshold slices, see --max-gap, so the spine isn't
       split by momentary confidence dips) and takes the first slice of that
       run as the study's "first slice". Axis 0 is ordered superior->inferior,
       so a small first-slice index means the top of the spine (C1 region).
    5. Writes one row per study to a CSV in this directory.

Run on a GPU node via the project venv:

    cd first_slice_detection
    uv run find_first_slice.py
    uv run find_first_slice.py --conf 0.8 --device 0
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from ultralytics import YOLO

TRAINING_PATH = "/vast/s222440401/training_images"
SEGMENTATION_PATH = "/vast/s222440401/segmentations"

# HU window used to build the YOLO training PNGs (reproduces them exactly).
HU_LOW, HU_HIGH = -200.0, 1800.0


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Locate the first vertebra slice per study with YOLOv11.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default=TRAINING_PATH,
        help="Root directory of DICOM study folders.",
    )
    parser.add_argument(
        "--seg",
        default=SEGMENTATION_PATH,
        help="Root directory of NIfTI segmentation files (.nii / .nii.gz).",
    )
    parser.add_argument(
        "--weights",
        default=str(here / "models" / "yolo11s-pose.pt"),
        help="Trained YOLOv11 pose weights.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.70,
        help="Minimum detection confidence to count a slice as a vertebra.",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=3,
        help="Allowed number of consecutive sub-threshold slices inside a run "
        "(bridges isolated confidence dips so the spine stays one sequence; "
        "0 = strictly consecutive).",
    )
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Number of slices per inference batch.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="CUDA device, e.g. '0'. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--out",
        default=str(here / "first_slices.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def discover_overlapping_studies(training_path: Path, segmentation_path: Path) -> list[str]:
    """Return study IDs present in both training_images and segmentations."""
    training_studies = {
        p.name for p in training_path.iterdir() if p.is_dir()
    }
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
    dicom_files = [
        path
        for path in study_dir.rglob("*.dcm")
        if path.is_file()
    ]
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


def slice_to_image(slice_2d: np.ndarray, imgsz: int) -> np.ndarray:
    """Window-normalise an axial HU slice to a 512x512 3-channel uint8 image."""
    windowed = np.clip(slice_2d, HU_LOW, HU_HIGH)
    scaled = (windowed - HU_LOW) / (HU_HIGH - HU_LOW) * 255.0
    img = Image.fromarray(scaled.astype(np.uint8)).resize((imgsz, imgsz))
    gray = np.asarray(img, dtype=np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def slice_confidences(model: YOLO, volume: np.ndarray, args) -> np.ndarray:
    """Return the max detection confidence for every axial slice of a volume."""
    num_slices = volume.shape[0]
    confs = np.zeros(num_slices, dtype=np.float32)
    for start in range(0, num_slices, args.batch):
        stop = min(start + args.batch, num_slices)
        batch = [slice_to_image(volume[z], args.imgsz) for z in range(start, stop)]
        results = model.predict(
            batch,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )
        for offset, res in enumerate(results):
            if res.boxes is not None and len(res.boxes) > 0:
                confs[start + offset] = float(res.boxes.conf.max())
    return confs


def longest_present_run(present: np.ndarray, max_gap: int):
    """Find the longest run of True values, tolerating up to ``max_gap`` gaps.

    Returns (start_index, length) of the longest run, or (-1, 0) if none.
    The start index ignores leading gap slices so it points at a real
    detection.
    """
    best_start, best_len = -1, 0
    run_start, run_present, gap = -1, 0, 0
    for i, val in enumerate(present):
        if val:
            if run_start == -1:
                run_start = i
            run_present += 1
            gap = 0
        elif run_start != -1:
            gap += 1
            if gap > max_gap:
                if run_present > best_len:
                    best_start, best_len = run_start, run_present
                run_start, run_present, gap = -1, 0, 0
    if run_present > best_len:
        best_start, best_len = run_start, run_present
    return best_start, best_len


def find_first_slice() -> None:
    """Find the first vertebra slice of each study with the trained YOLOv11 model."""
    args = parse_args()

    training_path = Path(args.data)
    segmentation_path = Path(args.seg)
    if not training_path.is_dir():
        raise FileNotFoundError(f"Training path not found: {training_path}")
    if not segmentation_path.is_dir():
        raise FileNotFoundError(f"Segmentation path not found: {segmentation_path}")
    if not Path(args.weights).is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    study_ids = discover_overlapping_studies(training_path, segmentation_path)
    print(
        f"Found {len(study_ids)} overlapping studies "
        f"({training_path} ∩ {segmentation_path})"
    )
    if not study_ids:
        raise RuntimeError("No overlapping studies found between data and segmentations")

    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    rows = []
    for idx, study_id in enumerate(study_ids, 1):
        study_dir = training_path / study_id
        volume = load_dicom_volume(study_dir)

        confs = slice_confidences(model, volume, args)
        present = confs >= args.conf
        start, length = longest_present_run(present, args.max_gap)

        first_conf = float(confs[start]) if start >= 0 else 0.0
        print(
            f"[{idx}/{len(study_ids)}] {study_id}: slices={volume.shape[0]} "
            f"present={int(present.sum())} first_slice={start} "
            f"run_len={length} conf={first_conf:.3f}"
        )
        rows.append(
            {
                "study_id": study_id,
                "first_slice": start,
                "sequence_length": length,
                "first_slice_confidence": round(first_conf, 4),
                "num_slices": int(volume.shape[0]),
                "num_present_slices": int(present.sum()),
            }
        )

    out_path = Path(args.out)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "study_id",
                "first_slice",
                "sequence_length",
                "first_slice_confidence",
                "num_slices",
                "num_present_slices",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    found = sum(1 for r in rows if r["first_slice"] >= 0)
    print(f"\nWrote {len(rows)} rows ({found} with a detected first slice) to {out_path}")