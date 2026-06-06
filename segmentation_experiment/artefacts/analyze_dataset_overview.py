"""
Extra dataset statistics: training/segmentation overlap, volume geometry, spacing.

Does not require DICOM decoding for core stats (uses NIfTI headers and folder counts).
Optional: sample HU statistics from a limited number of DICOM slices (slow on large data).

Outputs: console summary and CSV under ./output/
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_paths import resolve_data_path

DATA_PATH = resolve_data_path()
TRAINING_PATH = DATA_PATH / "training_images"
SEGMENTATION_PATH = DATA_PATH / "segmentations"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Optional DICOM sampling (set to 0 to skip)
MAX_STUDIES_FOR_HU_SAMPLE = 5
MAX_SLICES_PER_STUDY_HU = 20


def _list_seg_studies() -> dict[str, Path]:
    out = {}
    if not SEGMENTATION_PATH.is_dir():
        return out
    for f in os.listdir(SEGMENTATION_PATH):
        if f.endswith(".nii.gz"):
            sid = f.replace(".nii.gz", "")
            out[sid] = SEGMENTATION_PATH / f
        elif f.endswith(".nii"):
            sid = f.replace(".nii", "")
            out[sid] = SEGMENTATION_PATH / f
    return out


def _list_training_studies() -> set[str]:
    if not TRAINING_PATH.is_dir():
        return set()
    return {d for d in os.listdir(TRAINING_PATH) if (TRAINING_PATH / d).is_dir()}


def _nifti_geometry(path: Path):
    nii = nib.load(str(path))
    shape = nii.shape
    zooms = nii.header.get_zooms()[:3] if nii.header.get_zooms() else (1.0, 1.0, 1.0)
    return shape, tuple(float(z) for z in zooms)


def _sample_hu_from_dicoms(study_dir: Path, max_slices: int) -> list[float] | None:
    try:
        import pydicom
    except ImportError:
        return None
    files = sorted(f for f in os.listdir(study_dir) if f.endswith(".dcm"))
    if not files:
        return None
    idxs = np.linspace(0, len(files) - 1, num=min(max_slices, len(files)), dtype=int)
    hu_vals = []
    for i in idxs:
        ds = pydicom.dcmread(study_dir / files[i], force=True)
        arr = ds.pixel_array.astype(np.float64)
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        hu = arr * slope + intercept
        hu_vals.extend(hu.ravel().tolist())
    return hu_vals


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"DATA_PATH (resolved): {DATA_PATH}")
    if not DATA_PATH.is_dir():
        print(
            "Hint: set SEGMENTATION_DATA_ROOT to the directory that contains "
            "training_images/ and segmentations/, e.g. export SEGMENTATION_DATA_ROOT=/vast/s222440401"
        )

    train_studies = _list_training_studies()
    seg_map = _list_seg_studies()
    seg_studies = set(seg_map.keys())

    overlap = train_studies & seg_studies
    only_train = train_studies - seg_studies
    only_seg = seg_studies - train_studies

    print("=== Folder / ID overlap ===")
    print(f"training_images/ study dirs: {len(train_studies):,}")
    print(f"segmentations/ NIfTI volumes: {len(seg_studies):,}")
    print(f"Intersection (both):         {len(overlap):,}")
    print(f"Only in training_images:     {len(only_train):,}")
    print(f"Only in segmentations:       {len(only_seg):,}")

    pd.DataFrame(
        {
            "metric": [
                "n_training_dirs",
                "n_segmentation_files",
                "n_intersection",
                "n_only_training",
                "n_only_segmentation",
            ],
            "value": [
                len(train_studies),
                len(seg_studies),
                len(overlap),
                len(only_train),
                len(only_seg),
            ],
        }
    ).to_csv(OUTPUT_DIR / "dataset_overlap_counts.csv", index=False)

    if not seg_map:
        print("\nNo segmentations found; skipping geometry stats.")
        return

    rows = []
    for sid, path in tqdm(sorted(seg_map.items()), desc="NIfTI geometry"):
        try:
            shape, zooms = _nifti_geometry(path)
            depth, h, w = int(shape[0]), int(shape[1]), int(shape[2])
            vox_vol = float(zooms[0] * zooms[1] * zooms[2])
            rows.append(
                {
                    "study_id": sid,
                    "dim_0": depth,
                    "dim_1": h,
                    "dim_2": w,
                    "spacing_0": zooms[0],
                    "spacing_1": zooms[1],
                    "spacing_2": zooms[2],
                    "voxel_volume_mm3": vox_vol,
                    "in_training_images": sid in train_studies,
                }
            )
        except Exception as e:
            rows.append(
                {
                    "study_id": sid,
                    "error": str(e),
                }
            )

    geo_df = pd.DataFrame(rows)
    geo_df.to_csv(OUTPUT_DIR / "segmentation_volume_geometry.csv", index=False)

    ok = geo_df.dropna(subset=["dim_0"])
    if len(ok):
        print("\n=== Segmentation volume shape (NIfTI, raw on-disk order before training transpose) ===")
        for col, label in [("dim_0", "axis 0"), ("dim_1", "axis 1"), ("dim_2", "axis 2")]:
            print(f"  {label}: min={int(ok[col].min())}  max={int(ok[col].max())}  median={float(ok[col].median()):.1f}")
        print("\n=== Voxel spacing (mm, from header) ===")
        for i in range(3):
            c = f"spacing_{i}"
            print(f"  spacing[{i}]: min={ok[c].min():.4f}  max={ok[c].max():.4f}  median={ok[c].median():.4f}")
        print(f"\nVoxel volume (mm³): median={ok['voxel_volume_mm3'].median():.6f}")

    # Histograms of depth and voxel volume
    if len(ok):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(ok["dim_0"].values, bins=min(40, max(10, len(ok) // 10)), color="#2E86AB", edgecolor="black")
        axes[0].set_xlabel("NIfTI dimension 0 (slice count proxy)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Distribution of depth (dim 0)")

        axes[1].hist(ok["voxel_volume_mm3"].values, bins=min(40, max(10, len(ok) // 10)), color="#6A994E", edgecolor="black")
        axes[1].set_xlabel("Voxel volume (mm³)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Distribution of voxel volume")
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "volume_geometry_histograms.png", dpi=200, bbox_inches="tight")
        plt.close()

    # Optional HU sampling
    if MAX_STUDIES_FOR_HU_SAMPLE > 0 and overlap:
        sample_studies = sorted(overlap)[:MAX_STUDIES_FOR_HU_SAMPLE]
        all_hu = []
        for sid in sample_studies:
            hu = _sample_hu_from_dicoms(TRAINING_PATH / sid, MAX_SLICES_PER_STUDY_HU)
            if hu:
                all_hu.extend(hu)
        if all_hu:
            arr = np.array(all_hu, dtype=np.float64)
            print("\n=== HU sample (subset of DICOMs; pydicom) ===")
            print(f"  Studies sampled: {len(sample_studies)}, pixels: {arr.size:,}")
            print(f"  min={arr.min():.1f}  max={arr.max():.1f}  mean={arr.mean():.1f}  std={arr.std():.1f}")
            print(f"  p1={np.percentile(arr,1):.1f}  p50={np.percentile(arr,50):.1f}  p99={np.percentile(arr,99):.1f}")
            pd.DataFrame([{"hu_min": arr.min(), "hu_max": arr.max(), "hu_mean": arr.mean(), "hu_std": arr.std()}]).to_csv(
                OUTPUT_DIR / "hu_sample_summary.csv", index=False
            )
        else:
            print("\n(HU sampling skipped: no DICOMs read or pydicom missing.)")

    print(f"\nSaved outputs under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
