"""
Per-class pixel counts and percentages over all NIfTI segmentations.

Uses the same volume orientation and label remapping as E1-augmentation (data_process / training).
Outputs: console summary, CSV, and figures under ./output/
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
SEGMENTATION_PATH = DATA_PATH / "segmentations"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SKIP_SLICE = 1
NUM_CLASSES = 9

CLASS_LABELS = {
    0: "Background",
    1: "C1",
    2: "C2",
    3: "C3",
    4: "C4",
    5: "C5",
    6: "C6",
    7: "C7",
    8: "Other Vertebrae",
}

CLASS_COLORS = [
    "#2E86AB",
    "#A23B72",
    "#F18F01",
    "#C73E1D",
    "#6A994E",
    "#BC4749",
    "#F77F00",
    "#FCBF49",
    "#D62828",
]


def _prepare_seg_slice(seg_slice: np.ndarray) -> np.ndarray:
    s = seg_slice.astype(np.int64)
    s = np.where(s > 8, 8, s)
    return np.clip(s, 0, 8)


def analyze_single_segmentation(seg_path: Path, skip_slice: int = 1):
    try:
        nii = nib.load(str(seg_path))
        seg_volume = nii.get_fdata()
        seg_corrected = seg_volume[:, ::-1, ::-1].transpose(2, 1, 0)

        class_counts = {i: 0 for i in range(NUM_CLASSES)}
        total_pixels = 0
        num_slices = 0

        for i in range(0, seg_corrected.shape[0], skip_slice):
            seg_slice = _prepare_seg_slice(seg_corrected[i])
            u, c = np.unique(seg_slice, return_counts=True)
            for cls, count in zip(u, c):
                if 0 <= cls < NUM_CLASSES:
                    class_counts[int(cls)] += int(count)
            total_pixels += seg_slice.size
            num_slices += 1

        return class_counts, total_pixels, num_slices, None
    except Exception as e:
        return None, 0, 0, str(e)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SEGMENTATION_PATH.is_dir():
        print(f"Segmentation path not found: {SEGMENTATION_PATH}")
        print("Set up data under vast/s222440401/segmentations or adjust _ROOT in this script.")
        return

    files = sorted(
        f
        for f in os.listdir(SEGMENTATION_PATH)
        if f.endswith(".nii.gz") or f.endswith(".nii")
    )
    if not files:
        print(f"No NIfTI files in {SEGMENTATION_PATH}")
        return

    aggregated = {i: 0 for i in range(NUM_CLASSES)}
    total_pixels = 0
    per_file_rows = []
    errors = []

    for fname in tqdm(files, desc="Segmentations"):
        study_id = fname.replace(".nii.gz", "").replace(".nii", "")
        path = SEGMENTATION_PATH / fname
        cc, tp, ns, err = analyze_single_segmentation(path, SKIP_SLICE)
        if err:
            errors.append((fname, err))
            continue
        total_pixels += tp
        for k in range(NUM_CLASSES):
            aggregated[k] += cc[k]
        row = {"study_id": study_id, "total_pixels_analyzed": tp, "slices_counted": ns}
        for k in range(NUM_CLASSES):
            row[f"class_{k}_pixels"] = cc[k]
        per_file_rows.append(row)

    if errors:
        print(f"Warnings: {len(errors)} file(s) failed (see CSV).")
        pd.DataFrame(errors, columns=["file", "error"]).to_csv(
            OUTPUT_DIR / "segmentation_load_errors.csv", index=False
        )

    if total_pixels == 0:
        print("No pixels counted.")
        return

    rows = []
    for cls in range(NUM_CLASSES):
        c = aggregated[cls]
        pct = 100.0 * c / total_pixels
        rows.append(
            {
                "class_id": cls,
                "class_name": CLASS_LABELS[cls],
                "pixel_count": c,
                "percent_of_all_pixels": round(pct, 6),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "class_pixel_counts.csv", index=False)
    pd.DataFrame(per_file_rows).to_csv(OUTPUT_DIR / "per_study_class_pixels.csv", index=False)

    print("\n=== Pixel counts per class (all segmentations, skip_slice=%d) ===\n" % SKIP_SLICE)
    print(f"Total pixels analyzed: {total_pixels:,}")
    print(f"Volumes: {len(per_file_rows)}\n")
    for _, r in df.iterrows():
        print(f"  {r['class_name']:<18} (id {int(r['class_id'])}): {int(r['pixel_count']):>14,}  ({r['percent_of_all_pixels']:.4f}%)")

    bg = aggregated[0]
    vert = sum(aggregated[i] for i in range(1, NUM_CLASSES))
    print("\n=== Foreground vs background ===")
    print(f"  Background:  {bg:>14,}  ({100.0 * bg / total_pixels:.4f}%)")
    print(f"  Vertebrae+: {vert:>14,}  ({100.0 * vert / total_pixels:.4f}%)")

    # Figures
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    classes = list(range(NUM_CLASSES))
    counts = [aggregated[c] for c in classes]
    pcts = [100.0 * counts[i] / total_pixels for i in classes]
    labels = [CLASS_LABELS[c] for c in classes]

    axes[0].bar(classes, counts, color=CLASS_COLORS, edgecolor="black", linewidth=0.5)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Class id")
    axes[0].set_ylabel("Pixel count (log)")
    axes[0].set_title("Pixels per class")
    axes[0].set_xticks(classes)

    axes[1].bar(classes, pcts, color=CLASS_COLORS, edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Class id")
    axes[1].set_ylabel("Percent of all pixels")
    axes[1].set_title("Class distribution (%)")
    axes[1].set_xticks(classes)
    for i, p in enumerate(pcts):
        if p >= 0.05:
            axes[1].text(i, p + 0.1, f"{p:.2f}%", ha="center", fontsize=7, rotation=90)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_pixel_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()

    fig2, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        counts,
        labels=[f"{labels[i]}\n{pcts[i]:.2f}%" for i in range(NUM_CLASSES)],
        colors=CLASS_COLORS,
        autopct="",
        startangle=90,
    )
    ax.set_title("Pixel share per class")
    fig2.savefig(OUTPUT_DIR / "class_pixel_pie.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nSaved CSV and figures under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
