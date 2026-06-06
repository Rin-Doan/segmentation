"""
Cervical vertebrae slice analysis.

For each study in DATA_PATH, determine the first and last slice (z-axis index)
containing each cervical vertebra (C1..C7) based on the segmentation mask.

Outputs:
    * One CSV per vertebra: ``<OUT_DIR>/c{n}_slices.csv`` with columns
      (study, first_slice, last_slice, num_slice).
    * One histogram per vertebra showing the distribution of ``num_slice``,
      saved under ``<OUT_DIR>/figures/``.
    * A summary CSV (min, max, mean of ``num_slice``) across vertebrae.

Label convention (RSNA cervical spine style):
    1 -> C1, 2 -> C2, ..., 7 -> C7
Axis 0 of the NIfTI array is the slice (z) axis.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label as cc_label
from tqdm import tqdm


DATA_PATH = "../../../../../vast/s222440401/agg_data_1/segmentations_nii"
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
VERTEBRAE = [(i, f"C{i}") for i in range(1, 8)]


def resolve_data_path() -> Path:
    """Resolve DATA_PATH relative to this file (mirrors aggregate_data.py)."""
    p = Path(DATA_PATH)
    if not p.is_absolute():
        p = (OUT_DIR / p).resolve()
    if not p.exists():
        fallback = Path("/vast/s222440401/agg_data_1/segmentations_nii")
        if fallback.exists():
            return fallback
    return p


_CC_STRUCTURE = np.ones((3, 3, 3), dtype=bool)


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Return the binary mask of the largest 26-connected component.

    If ``mask`` is empty, the original (empty) mask is returned.
    """
    if not mask.any():
        return mask
    labels, n = cc_label(mask, structure=_CC_STRUCTURE)
    if n <= 1:
        return labels > 0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # ignore background
    largest = int(sizes.argmax())
    return labels == largest


def slice_range_for_label(
    seg: np.ndarray, label: int, use_largest_cc: bool = True
) -> tuple[int, int, int] | None:
    """Return (first_slice, last_slice, num_slice) along axis 0 for ``label``.

    When ``use_largest_cc`` is True, stray voxels are removed by keeping only
    the largest 3D connected component of the label before computing the
    slice range. Returns ``None`` if the label is absent.
    """
    mask = seg == label
    if use_largest_cc:
        mask = largest_connected_component(mask)
    present = np.any(mask, axis=(1, 2))
    idx = np.where(present)[0]
    if idx.size == 0:
        return None
    first = int(idx.min())
    last = int(idx.max())
    return first, last, last - first + 1


def analyse_studies(
    seg_dir: Path, use_largest_cc: bool = True
) -> dict[int, pd.DataFrame]:
    """Compute per-vertebra slice ranges for every NIfTI under ``seg_dir``."""
    files = sorted(f for f in os.listdir(seg_dir) if f.endswith((".nii", ".nii.gz")))
    if not files:
        raise FileNotFoundError(f"No NIfTI files found in {seg_dir}")

    records: dict[int, list[dict]] = {label: [] for label, _ in VERTEBRAE}

    for fname in tqdm(files, desc="Analysing studies"):
        study = fname.replace(".nii.gz", "").replace(".nii", "")
        seg = nib.load(str(seg_dir / fname)).get_fdata().astype(np.int16)
        for label, _ in VERTEBRAE:
            result = slice_range_for_label(seg, label, use_largest_cc=use_largest_cc)
            if result is None:
                first_slice = last_slice = num_slice = np.nan
            else:
                first_slice, last_slice, num_slice = result
            records[label].append(
                {
                    "study": study,
                    "first_slice": first_slice,
                    "last_slice": last_slice,
                    "num_slice": num_slice,
                }
            )

    return {label: pd.DataFrame(rows) for label, rows in records.items()}


def save_per_vertebra_csvs(dfs: dict[int, pd.DataFrame]) -> None:
    for label, name in VERTEBRAE:
        out_path = OUT_DIR / f"{name.lower()}_slices.csv"
        dfs[label].to_csv(out_path, index=False)
        print(f"Saved {out_path}")


def save_histograms(dfs: dict[int, pd.DataFrame]) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    for label, name in VERTEBRAE:
        df = dfs[label]
        counts = df["num_slice"].dropna().astype(int).to_numpy()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if counts.size:
            bins = max(10, min(30, int(np.sqrt(counts.size) * 2)))
            ax.hist(counts, bins=bins, color="steelblue", edgecolor="black")
            ax.axvline(counts.mean(), color="red", linestyle="--",
                       label=f"mean={counts.mean():.1f}")
            ax.legend()
        ax.set_title(f"{name}: distribution of slice counts "
                     f"(n={counts.size}/{len(df)})", fontweight="bold")
        ax.set_xlabel("Number of slices containing the vertebra")
        ax.set_ylabel("Number of studies")
        fig.tight_layout()
        fig_path = FIG_DIR / f"{name.lower()}_num_slices_hist.png"
        fig.savefig(fig_path, dpi=600)
        plt.close(fig)
        print(f"Saved {fig_path}")


def count_pixel_classes(seg_dir: Path) -> tuple[int, int]:
    """Count total background and vertebrae pixels across all studies.

    Returns (background_pixels, vertebrae_pixels).
    """
    files = sorted(f for f in os.listdir(seg_dir) if f.endswith((".nii", ".nii.gz")))
    if not files:
        raise FileNotFoundError(f"No NIfTI files found in {seg_dir}")

    bg_total = 0
    vert_total = 0
    for fname in tqdm(files, desc="Counting pixels"):
        seg = nib.load(str(seg_dir / fname)).get_fdata().astype(np.int16)
        bg_total += int((seg == 0).sum())
        vert_total += int(((seg >= 1) & (seg <= 7)).sum())

    return bg_total, vert_total


def save_pixel_pie_chart(bg_pixels: int, vert_pixels: int) -> None:
    """Save a pie chart comparing background vs vertebrae pixel counts."""
    FIG_DIR.mkdir(exist_ok=True)

    _MM_PER_INCH = 25.4
    width_in = 84 / _MM_PER_INCH  # 84 mm -> inches
    fig, ax = plt.subplots(figsize=(width_in, width_in))

    sizes = [bg_pixels, vert_pixels]
    labels = ["Background", "Vertebrae"]
    colors = ["#b0c4de", "#e07b54"]
    explode = (0.03, 0.03)

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 0.5},
    )
    for t in texts + autotexts:
        t.set_fontsize(5)

    ax.set_title("Background vs Vertebrae\npixel distribution", fontsize=6, pad=4, fontweight="bold")
    fig.tight_layout()
    fig_path = FIG_DIR / "pixel_class_distribution_pie.png"
    fig.savefig(fig_path, dpi=600)
    plt.close(fig)
    print(f"Saved {fig_path}")


def build_summary(dfs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for label, name in VERTEBRAE:
        counts = dfs[label]["num_slice"].dropna().astype(float)
        if counts.empty:
            rows.append({
                "vertebra": name, "studies_with_label": 0,
                "min": np.nan, "max": np.nan, "mean": np.nan,
            })
            continue
        rows.append({
            "vertebra": name,
            "studies_with_label": int(counts.size),
            "min": int(counts.min()),
            "max": int(counts.max()),
            "mean": float(counts.mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    seg_dir = resolve_data_path()
    print(f"Using segmentation directory: {seg_dir}")

    dfs = analyse_studies(seg_dir)
    save_per_vertebra_csvs(dfs)
    save_histograms(dfs)

    summary = build_summary(dfs)
    summary_path = OUT_DIR / "vertebrae_slice_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}\n")
    print(summary.to_string(index=False))

    bg_pixels, vert_pixels = count_pixel_classes(seg_dir)
    save_pixel_pie_chart(bg_pixels, vert_pixels)


if __name__ == "__main__":
    main()
