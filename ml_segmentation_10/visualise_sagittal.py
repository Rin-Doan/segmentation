"""
Visualise sagittal views of bm_segmentations_nii volumes.

For each study, loads the pre-aggregated segmentation (same D, H, W convention
as aggregate_data.py), crops/pads to the training grid (256, 256, 256), and
renders one sagittal slice at the middle x index using the first 256 z slices.

Output layout
-------------
  {OUTPUT_PATH}/{study_id}/sagittal.png   — one sagittal panel per study
  {OUTPUT_PATH}/legend.png                — class colour key
  {OUTPUT_PATH}/montage.png               — all studies in a single grid

Run from ml_segmentation/:
    python visualise_sagittal.py
"""

import math
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from tqdm import tqdm

from data_process import crop_or_pad_to_size

warnings.filterwarnings("ignore")

DATA_PATH = "/vast/s222440401"
SEGMENTATION_PATH = os.path.join(DATA_PATH, "agg_data/segmentations_nii")
OUTPUT_PATH = os.path.join(DATA_PATH, "ml_segmentation")

TARGET_SHAPE = (256, 256, 256)
NUM_Z_SLICES = 256
SAG_X = TARGET_SHAPE[2] // 2  # middle x column for sagittal cut

# Class colours after ml training remap (0 bg, 1–7 C1–C7, 8 other)
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


def get_study_id(filename: str) -> str:
    return filename.replace(".nii.gz", "").replace(".nii", "")


def discover_studies() -> list[str]:
    return sorted(
        get_study_id(f)
        for f in os.listdir(SEGMENTATION_PATH)
        if f.endswith((".nii", ".nii.gz"))
    )


def find_segmentation(study_id: str) -> str | None:
    for ext in (".nii", ".nii.gz"):
        path = os.path.join(SEGMENTATION_PATH, f"{study_id}{ext}")
        if os.path.exists(path):
            return path
    return None


def load_segmentation(study_id: str) -> np.ndarray | None:
    """Load segmentation, crop/pad to TARGET_SHAPE, remap labels like training."""
    seg_path = find_segmentation(study_id)
    if seg_path is None:
        return None

    seg = nib.load(seg_path).get_fdata().astype(np.int64)
    # Coordinate correction was applied upstream in aggregate_data.py before save.
    seg = crop_or_pad_to_size(seg, TARGET_SHAPE)
    seg = np.where(seg > 7, 8, seg)
    return seg


def seg_to_rgb(seg_2d: np.ndarray) -> np.ndarray:
    """(H, W) int labels -> (H, W, 3) float in [0, 1]."""
    return SEG_COLORS[np.clip(seg_2d.astype(np.int64), 0, 8)]


def extract_sagittal(seg_vol: np.ndarray) -> np.ndarray:
    """Return sagittal RGB image for the first NUM_Z_SLICES z positions."""
    sag = seg_vol[:NUM_Z_SLICES, :, SAG_X]
    return seg_to_rgb(sag)


def save_figure(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, bbox_inches="tight", facecolor="black", dpi=100)
    plt.close(fig)


def save_study_sagittal(study_id: str, sag_rgb: np.ndarray, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sagittal.png")

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    fig.patch.set_facecolor("black")
    ax.imshow(sag_rgb, origin="upper")
    ax.axis("off")
    short_id = study_id.split(".")[-1] if "." in study_id else study_id
    ax.set_title(short_id, color="white", fontsize=8)
    save_figure(fig, out_path)
    return out_path


def build_legend(path: str) -> None:
    fig, ax = plt.subplots(figsize=(2.5, 4), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.axis("off")
    ax.set_title("Segmentation labels", color="white", fontsize=10)

    for i, name in enumerate(CLASS_NAMES):
        y = 1.0 - (i + 1) / (len(CLASS_NAMES) + 1)
        ax.add_patch(plt.Rectangle((0.05, y - 0.03), 0.12, 0.05, color=SEG_COLORS[i]))
        ax.text(0.22, y, f"{i} – {name}", color="white", fontsize=8, va="center")

    save_figure(fig, path)


def build_montage(study_paths: list[tuple[str, str]], cols: int = 9) -> None:
    """Tile per-study sagittal PNGs into one overview image."""
    if not study_paths:
        return

    n = len(study_paths)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), dpi=100)
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
        short_id = study_id.split(".")[-1] if "." in study_id else study_id
        ax.set_title(short_id, color="white", fontsize=6)

    montage_path = os.path.join(OUTPUT_PATH, "montage.png")
    save_figure(fig, montage_path)
    print(f"\nMontage saved -> {montage_path}  ({n} studies)")


def main() -> None:
    print("=" * 60)
    print("BM Segmentation Sagittal Visualisation")
    print("=" * 60)
    print(f"Segmentations: {SEGMENTATION_PATH}")
    print(f"Output:        {OUTPUT_PATH}")
    print(f"Grid:          {TARGET_SHAPE}  (first {NUM_Z_SLICES} z slices)")
    print(f"Sagittal x:    {SAG_X}")
    print()

    if not os.path.isdir(SEGMENTATION_PATH):
        raise FileNotFoundError(f"Segmentation directory not found: {SEGMENTATION_PATH}")

    studies = discover_studies()
    print(f"Studies to visualise: {len(studies)}\n")

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    legend_path = os.path.join(OUTPUT_PATH, "legend.png")
    build_legend(legend_path)
    print(f"Colour legend saved -> {legend_path}\n")

    saved_paths: list[tuple[str, str]] = []
    skipped = 0

    for study_id in tqdm(studies, desc="Studies"):
        out_dir = os.path.join(OUTPUT_PATH, study_id)
        out_file = os.path.join(out_dir, "sagittal.png")
        if os.path.isfile(out_file):
            saved_paths.append((study_id, out_file))
            continue

        seg_vol = load_segmentation(study_id)
        if seg_vol is None:
            skipped += 1
            tqdm.write(f"  Warning: no segmentation found for {study_id}")
            continue

        sag_rgb = extract_sagittal(seg_vol)
        path = save_study_sagittal(study_id, sag_rgb, out_dir)
        saved_paths.append((study_id, path))

    build_montage(saved_paths)

    print(f"\nDone. Per-study sagittal views saved under {OUTPUT_PATH}/")
    if skipped:
        print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
