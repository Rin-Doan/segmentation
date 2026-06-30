"""
Visualise sagittal views of segmentation volumes with first-slice markers.

For each study listed in first_slices.csv, loads the NIfTI segmentation from
/vast/s222440401/segmentations (with the same coordinate correction used in
aggregate_data.py), extracts a sagittal slice at the middle x index, and draws
a horizontal line at the detected first_slice z position.

Output layout
-------------
  {OUTPUT_PATH}/{study_id}/sagittal.png   — one sagittal panel per study
  {OUTPUT_PATH}/legend.png                — class colour key
  {OUTPUT_PATH}/montage.png               — all studies in a single grid

Run from first_slice_detection/:
    ../.venv/bin/python visualise_sagittal.py
    ../.venv/bin/python visualise_sagittal.py --csv first_slices.csv --out ./sagittal_vis
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA_PATH = "/vast/s222440401"
SEGMENTATION_PATH = os.path.join(DATA_PATH, "segmentations")
DEFAULT_CSV = HERE / "first_slices.csv"
DEFAULT_OUTPUT = os.path.join(DATA_PATH, "first_slice_detection", "sagittal_vis")

FIRST_SLICE_COLOR = "#00ffff"
FIRST_SLICE_LINEWIDTH = 1.5

# Vertebra label colours (0 background, 1–7 C1–C7, 8+ other)
SEG_COLORS = np.array(
    [
        [0, 0, 0],
        [220, 50, 50],
        [240, 140, 40],
        [230, 210, 50],
        [60, 180, 60],
        [50, 190, 220],
        [60, 100, 220],
        [170, 60, 220],
        [160, 160, 160],
    ],
    dtype=np.float32,
) / 255.0
CLASS_NAMES = ["background", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "other"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise sagittal segmentations with first-slice markers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seg",
        default=SEGMENTATION_PATH,
        help="Directory of NIfTI segmentation files.",
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV),
        help="CSV with study_id and first_slice columns.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT,
        help="Output directory for PNGs.",
    )
    parser.add_argument(
        "--montage-cols",
        type=int,
        default=9,
        help="Number of columns in the overview montage.",
    )
    return parser.parse_args()


def load_first_slice_rows(csv_path: str) -> list[dict[str, str]]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def find_segmentation(seg_dir: str, study_id: str) -> str | None:
    for ext in (".nii", ".nii.gz"):
        path = os.path.join(seg_dir, f"{study_id}{ext}")
        if os.path.exists(path):
            return path
    return None


def load_segmentation(seg_path: str) -> np.ndarray:
    """Load segmentation and apply the same (D, H, W) correction as aggregate_data."""
    seg = nib.load(seg_path).get_fdata()
    seg = seg[:, ::-1, ::-1].transpose(2, 1, 0)
    return seg.astype(np.int64)


def seg_to_rgb(seg_2d: np.ndarray) -> np.ndarray:
    """(H, W) int labels -> (H, W, 3) float in [0, 1]."""
    labels = np.clip(seg_2d.astype(np.int64), 0, 8)
    return SEG_COLORS[labels]


def extract_sagittal(seg_vol: np.ndarray) -> tuple[np.ndarray, int]:
    """Return sagittal RGB image and the x index used for the cut."""
    sag_x = seg_vol.shape[2] // 2
    sag = seg_vol[:, :, sag_x]
    return seg_to_rgb(sag), sag_x


def short_study_id(study_id: str) -> str:
    return study_id.split(".")[-1] if "." in study_id else study_id


def save_figure(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, bbox_inches="tight", facecolor="black", dpi=100)
    plt.close(fig)


def save_study_sagittal(
    study_id: str,
    sag_rgb: np.ndarray,
    first_slice: int,
    confidence: float | None,
    sag_x: int,
    out_dir: str,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sagittal.png")

    z_slices, _ = sag_rgb.shape[:2]
    fig_h = max(4.0, z_slices / 256.0 * 4.0)
    fig, ax = plt.subplots(figsize=(4, fig_h), dpi=100)
    fig.patch.set_facecolor("black")
    ax.imshow(sag_rgb, origin="upper", aspect="auto")

    if 0 <= first_slice < z_slices:
        ax.axhline(
            y=first_slice,
            color=FIRST_SLICE_COLOR,
            linewidth=FIRST_SLICE_LINEWIDTH,
            linestyle="--",
            label=f"first slice ({first_slice})",
        )
        ax.legend(
            loc="upper right",
            facecolor="black",
            edgecolor="white",
            labelcolor="white",
            fontsize=7,
        )

    ax.axis("off")
    title = short_study_id(study_id)
    if confidence is not None:
        title += f"  z={first_slice}  conf={confidence:.2f}"
    else:
        title += f"  z={first_slice}"
    title += f"  x={sag_x}"
    ax.set_title(title, color="white", fontsize=8)

    save_figure(fig, out_path)
    return out_path


def build_legend(path: str) -> None:
    fig, ax = plt.subplots(figsize=(2.8, 4.5), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.axis("off")
    ax.set_title("Segmentation labels", color="white", fontsize=10)

    entries = list(enumerate(CLASS_NAMES)) + [(-1, "first slice (detected)")]
    for i, (idx, name) in enumerate(entries):
        y = 1.0 - (i + 1) / (len(entries) + 1)
        if idx == -1:
            ax.plot(
                [0.05, 0.17],
                [y, y],
                color=FIRST_SLICE_COLOR,
                linewidth=FIRST_SLICE_LINEWIDTH,
                linestyle="--",
            )
            ax.text(0.22, y, name, color="white", fontsize=8, va="center")
        else:
            ax.add_patch(plt.Rectangle((0.05, y - 0.03), 0.12, 0.05, color=SEG_COLORS[idx]))
            ax.text(0.22, y, f"{idx} – {name}", color="white", fontsize=8, va="center")

    save_figure(fig, path)


def build_montage(study_paths: list[tuple[str, str]], output_path: str, cols: int) -> None:
    if not study_paths:
        return

    n = len(study_paths)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2), dpi=100)
    fig.patch.set_facecolor("black")

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.set_facecolor("black")
        ax.axis("off")
        if idx >= n:
            continue
        study_id, img_path = study_paths[idx]
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.set_title(short_study_id(study_id), color="white", fontsize=6)

    montage_path = os.path.join(output_path, "montage.png")
    save_figure(fig, montage_path)
    print(f"\nMontage saved -> {montage_path}  ({n} studies)")


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("First-slice sagittal visualisation")
    print("=" * 60)
    print(f"Segmentations: {args.seg}")
    print(f"First slices:  {args.csv}")
    print(f"Output:        {args.out}")
    print()

    if not os.path.isdir(args.seg):
        raise FileNotFoundError(f"Segmentation directory not found: {args.seg}")
    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    rows = load_first_slice_rows(args.csv)
    print(f"Studies in CSV: {len(rows)}\n")

    os.makedirs(args.out, exist_ok=True)
    legend_path = os.path.join(args.out, "legend.png")
    build_legend(legend_path)
    print(f"Colour legend saved -> {legend_path}\n")

    saved_paths: list[tuple[str, str]] = []
    skipped = 0

    for row in tqdm(rows, desc="Studies"):
        study_id = row["study_id"]
        first_slice = int(row["first_slice"])
        confidence = float(row["first_slice_confidence"]) if row.get("first_slice_confidence") else None

        out_dir = os.path.join(args.out, study_id)
        out_file = os.path.join(out_dir, "sagittal.png")
        if os.path.isfile(out_file):
            saved_paths.append((study_id, out_file))
            continue

        seg_path = find_segmentation(args.seg, study_id)
        if seg_path is None:
            skipped += 1
            tqdm.write(f"  Warning: no segmentation found for {study_id}")
            continue

        seg_vol = load_segmentation(seg_path)
        sag_rgb, sag_x = extract_sagittal(seg_vol)
        path = save_study_sagittal(
            study_id,
            sag_rgb,
            first_slice,
            confidence,
            sag_x,
            out_dir,
        )
        saved_paths.append((study_id, path))

    build_montage(saved_paths, args.out, args.montage_cols)

    print(f"\nDone. Per-study sagittal views saved under {args.out}/")
    if skipped:
        print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
