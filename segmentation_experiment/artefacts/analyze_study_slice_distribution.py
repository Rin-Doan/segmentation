"""
Histogram of slice counts per study.

- n_dicom_slices: number of *.dcm files under training_images/<study_id>/ (source stack depth).
- n_seg_slices: segmentation depth in the same axis order as E1-augmentation/data_process
  (on-disk NIfTI shape dimension 2 after transpose(2,1,0) becomes slice axis).
- n_effective_slices: min(n_dicom_slices, n_seg_slices) when both exist (what training pairs).

Outputs: console summary, CSV, and figure under ./output/
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


def _seg_path_for_study(study_id: str) -> Path | None:
    gz = SEGMENTATION_PATH / f"{study_id}.nii.gz"
    if gz.is_file():
        return gz
    nii = SEGMENTATION_PATH / f"{study_id}.nii"
    if nii.is_file():
        return nii
    return None


def _n_seg_slices_training_order(seg_path: Path) -> int:
    """Slice depth along axis 0 of seg_corrected in data_process (transpose(2,1,0) of raw volume)."""
    shape = nib.load(str(seg_path)).shape
    if len(shape) < 3:
        return int(shape[0])
    return int(shape[2])


def _count_dcm(study_dir: Path) -> int:
    if not study_dir.is_dir():
        return 0
    return sum(1 for f in os.listdir(study_dir) if f.endswith(".dcm"))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"DATA_PATH (resolved): {DATA_PATH}")
    if not DATA_PATH.is_dir():
        print(
            "Hint: set SEGMENTATION_DATA_ROOT to the directory that contains "
            "training_images/ and segmentations/, e.g. export SEGMENTATION_DATA_ROOT=/vast/s222440401"
        )

    if not TRAINING_PATH.is_dir():
        print(f"No training_images directory: {TRAINING_PATH}")
        return

    study_dirs = sorted(d for d in TRAINING_PATH.iterdir() if d.is_dir())
    rows = []
    for study_dir in tqdm(study_dirs, desc="Slice counts"):
        sid = study_dir.name
        n_dcm = _count_dcm(study_dir)
        seg_path = _seg_path_for_study(sid)
        n_seg: int | None = None
        n_eff: int | None = None
        if seg_path is not None:
            try:
                n_seg = _n_seg_slices_training_order(seg_path)
                n_eff = min(n_dcm, n_seg)
            except Exception as e:
                rows.append(
                    {
                        "study_id": sid,
                        "n_dicom_slices": n_dcm,
                        "n_seg_slices": None,
                        "n_effective_slices": None,
                        "error": str(e),
                    }
                )
                continue
        rows.append(
            {
                "study_id": sid,
                "n_dicom_slices": n_dcm,
                "n_seg_slices": n_seg,
                "n_effective_slices": n_eff,
                "error": None,
            }
        )

    df = pd.DataFrame(rows)
    out_csv = OUTPUT_DIR / "study_slice_counts.csv"
    df.to_csv(out_csv, index=False)

    with_dcm = df[df["n_dicom_slices"] > 0]
    both = df.dropna(subset=["n_effective_slices"])

    print("\n=== Slice counts (training_images) ===")
    print(f"Study directories: {len(df):,}")
    print(f"With at least one .dcm: {len(with_dcm):,}")
    if len(with_dcm):
        s = with_dcm["n_dicom_slices"]
        print(
            f"DICOM slices — min={int(s.min())}  max={int(s.max())}  "
            f"median={float(s.median()):.1f}  mean={float(s.mean()):.1f}"
        )
    if len(both):
        e = both["n_effective_slices"]
        print(
            f"Effective slices (min DICOM, seg), {len(both):,} studies — "
            f"min={int(e.min())}  max={int(e.max())}  median={float(e.median()):.1f}"
        )
        mismatch = both[both["n_dicom_slices"] != both["n_seg_slices"]]
        if len(mismatch):
            print(f"Studies where DICOM count ≠ seg depth: {len(mismatch):,}")

    if len(with_dcm) == 0:
        print("No DICOM slices found; skipping figure.")
        print(f"Saved: {out_csv}")
        return

    n = len(with_dcm)
    bins = min(40, max(10, n // 10))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(
        with_dcm["n_dicom_slices"].values,
        bins=bins,
        color="#2E86AB",
        edgecolor="black",
        alpha=0.9,
    )
    ax.set_xlabel("Number of DICOM slices per study")
    ax.set_ylabel("Number of studies")
    ax.set_title("Distribution of slice count per study (training_images/*.dcm)")
    plt.tight_layout()
    fig_path = OUTPUT_DIR / "study_slice_count_histogram.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
