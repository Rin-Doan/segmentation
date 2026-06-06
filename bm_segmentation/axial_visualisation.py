"""
Export axial (z-axis) slice visualisations for a single BM study.

Iterates through every slice in bm_images_nii and bm_segmentations_nii,
saving each axial slice as a PNG under ./axial_vis/<study_id>/.
"""

from __future__ import annotations

import argparse
import os
import random
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

warnings.filterwarnings("ignore")

DATA_PATH = "../../../../../vast/s222440401"
BM_IMAGES_PATH = os.path.join(DATA_PATH, "bm_data/bm_images_nii")
BM_SEG_PATH = os.path.join(DATA_PATH, "bm_data/bm_segmentations_nii")
OUTPUT_DIR = "./axial_vis"

HU_MIN, HU_MAX = -200.0, 1800.0

LABEL_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
    "#999999",
]


def list_studies() -> list[str]:
    """Return study IDs that have both image and segmentation volumes."""
    image_ids = {
        f.replace(".nii.gz", "").replace(".nii", "")
        for f in os.listdir(BM_IMAGES_PATH)
        if f.endswith((".nii", ".nii.gz"))
    }
    seg_ids = {
        f.replace(".nii.gz", "").replace(".nii", "")
        for f in os.listdir(BM_SEG_PATH)
        if f.endswith((".nii", ".nii.gz"))
    }
    return sorted(image_ids & seg_ids)


def pick_random_study(seed: int | None = None) -> str:
    studies = list_studies()
    if not studies:
        raise FileNotFoundError(
            f"No overlapping studies found in {BM_IMAGES_PATH} and {BM_SEG_PATH}"
        )
    rng = random.Random(seed)
    return rng.choice(studies)


def load_nifti(path: str) -> np.ndarray:
    return np.ascontiguousarray(nib.load(path).get_fdata(), dtype=np.float32)


def segmentation_overlay(seg_slice: np.ndarray, labels: list[int]) -> np.ndarray:
    """Build an RGBA overlay for segmentation labels on one axial slice."""
    overlay = np.zeros((*seg_slice.shape, 4), dtype=np.float32)
    for idx, label in enumerate(labels):
        color = mcolors.to_rgba(LABEL_COLORS[idx % len(LABEL_COLORS)], alpha=0.55)
        overlay[seg_slice == label] = color
    return overlay


def save_axial_slice(
    image_slice: np.ndarray,
    seg_slice: np.ndarray,
    labels: list[int],
    z_index: int,
    output_path: str,
    dpi: int = 100,
) -> None:
    """Save one axial slice panel: CT | segmentation | CT + overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white")

    axes[0].imshow(image_slice, cmap="gray", vmin=HU_MIN, vmax=HU_MAX)
    axes[0].set_title(f"CT (z={z_index})")
    axes[0].axis("off")

    seg_rgb = np.zeros((*seg_slice.shape, 3), dtype=np.float32)
    for idx, label in enumerate(labels):
        color = mcolors.to_rgb(LABEL_COLORS[idx % len(LABEL_COLORS)])
        seg_rgb[seg_slice == label] = color
    axes[1].imshow(seg_rgb)
    axes[1].set_title(f"Segmentation (z={z_index})")
    axes[1].axis("off")

    axes[2].imshow(image_slice, cmap="gray", vmin=HU_MIN, vmax=HU_MAX)
    if labels:
        axes[2].imshow(segmentation_overlay(seg_slice, labels))
    axes[2].set_title(f"CT + overlay (z={z_index})")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_axial_slices(
    study_id: str,
    output_dir: str | None = None,
    dpi: int = 100,
) -> list[str]:
    """
    Export all axial slices for a study.

    Returns a list of paths to the saved PNG files.
    """
    image_path = os.path.join(BM_IMAGES_PATH, f"{study_id}.nii")
    seg_path = os.path.join(BM_SEG_PATH, f"{study_id}.nii")

    image = load_nifti(image_path)
    segmentation = load_nifti(seg_path).astype(np.int32)

    if image.shape != segmentation.shape:
        raise ValueError(
            f"Shape mismatch for {study_id}: image {image.shape} vs seg {segmentation.shape}"
        )

    labels = sorted(int(v) for v in np.unique(segmentation) if v > 0)
    study_dir = os.path.join(output_dir or OUTPUT_DIR, study_id)
    os.makedirs(study_dir, exist_ok=True)

    saved_paths: list[str] = []
    num_slices = image.shape[0]

    for z in range(num_slices):
        out_path = os.path.join(study_dir, f"slice_{z:03d}.png")
        save_axial_slice(
            image_slice=image[z],
            seg_slice=segmentation[z],
            labels=labels,
            z_index=z,
            output_path=out_path,
            dpi=dpi,
        )
        saved_paths.append(out_path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export all axial slices for a single BM study"
    )
    parser.add_argument(
        "--study",
        default=None,
        help="Study ID to visualise (default: random study)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed when choosing a study",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Output directory (default: ./axial_vis)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Output image resolution (default: 100)",
    )
    return parser.parse_args()


def main() -> list[str]:
    args = parse_args()
    study_id = args.study or pick_random_study(seed=args.seed)

    print("=" * 60)
    print("BM Axial Visualisation")
    print("=" * 60)
    print(f"Study:          {study_id}")
    print(f"Images:         {os.path.abspath(BM_IMAGES_PATH)}")
    print(f"Segmentations:  {os.path.abspath(BM_SEG_PATH)}")
    print(f"Output dir:     {os.path.abspath(args.output_dir)}")
    print("=" * 60)

    saved_paths = render_axial_slices(
        study_id,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )

    print(f"Saved {len(saved_paths)} axial slices to:")
    print(f"  {os.path.abspath(os.path.join(args.output_dir, study_id))}")
    return saved_paths


if __name__ == "__main__":
    main()
