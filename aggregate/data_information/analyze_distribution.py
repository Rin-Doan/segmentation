"""
Scan training DICOM studies and summarize data distribution:
  1) In-plane dimensions (rows x columns)
  2) In-plane voxel spacing (PixelSpacing: y, x in mm)
  3) Number of slices (z)
  4) Slice spacing / thickness (SpacingBetweenSlices or mean |Δz| from IPP; SliceThickness tag)

Paths match aggregate/aggregate_data.py (TRAINING_PATH).
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from pydicom.tag import Tag

# Same layout as aggregate/aggregate_data.py (lines 20–22)
_AGG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_PATH = os.path.abspath(os.path.join(_AGG_DIR, "../../../../../vast/s222440401"))
TRAINING_PATH = os.path.join(_DATA_PATH, "training_images")

# Lightweight reads: sort + IPP in one pass per file
_TAGS_SORT_IPP = [
    Tag("InstanceNumber"),
    Tag("ImagePositionPatient"),
]
_TAGS_GEOM = [
    Tag("Rows"),
    Tag("Columns"),
    Tag("PixelSpacing"),
    Tag("SliceThickness"),
    Tag("SpacingBetweenSlices"),
]


def _collect_dicom_paths(study_dir: str) -> list[str]:
    paths: list[str] = []
    for root, _, files in os.walk(study_dir):
        for f in files:
            if f.lower().endswith(".dcm"):
                paths.append(os.path.join(root, f))
    return paths


def _safe_float(x) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def analyze_study(study_dir: str) -> dict | None:
    """Return one row of metadata; None if no valid DICOM."""
    dicom_paths = _collect_dicom_paths(study_dir)
    if not dicom_paths:
        return None

    rows_meta: list[tuple[int, float | None, str]] = []
    for p in dicom_paths:
        try:
            ds = pydicom.dcmread(
                p,
                stop_before_pixels=True,
                force=True,
                specific_tags=_TAGS_SORT_IPP,
            )
            inst = int(getattr(ds, "InstanceNumber", 0) or 0)
            ipp = getattr(ds, "ImagePositionPatient", None)
            z = float(ipp[2]) if ipp is not None and len(ipp) > 2 else None
        except Exception:
            inst = 0
            z = None
        rows_meta.append((inst, z, p))

    rows_meta.sort(key=lambda t: t[0])
    sorted_paths = [t[2] for t in rows_meta]

    try:
        ds0 = pydicom.dcmread(
            sorted_paths[0],
            stop_before_pixels=True,
            force=True,
            specific_tags=_TAGS_GEOM,
        )
    except Exception:
        return None

    r = int(getattr(ds0, "Rows", 0) or 0)
    c = int(getattr(ds0, "Columns", 0) or 0)
    if r <= 0 or c <= 0:
        return None

    if hasattr(ds0, "PixelSpacing") and ds0.PixelSpacing is not None:
        ps = ds0.PixelSpacing
        spacing_y = float(ps[0])
        spacing_x = float(ps[1])
    else:
        spacing_y, spacing_x = np.nan, np.nan

    slice_thickness_tag = _safe_float(getattr(ds0, "SliceThickness", None))
    if slice_thickness_tag is None:
        slice_thickness_tag = np.nan
    else:
        slice_thickness_tag = float(slice_thickness_tag)

    spacing_between = _safe_float(getattr(ds0, "SpacingBetweenSlices", None))
    if spacing_between is None:
        spacing_between = np.nan
    else:
        spacing_between = float(spacing_between)

    zs = [t[1] for t in rows_meta if t[1] is not None]
    zs_sorted = np.array(sorted(zs), dtype=float)

    if not np.isnan(spacing_between) and spacing_between > 0:
        z_spacing_mm = float(spacing_between)
    elif len(zs_sorted) > 1:
        z_spacing_mm = float(np.mean(np.abs(np.diff(zs_sorted))))
    else:
        z_spacing_mm = np.nan

    n_slices = len(sorted_paths)

    return {
        "study_id": os.path.basename(study_dir.rstrip(os.sep)),
        "rows": r,
        "cols": c,
        "spacing_y_mm": spacing_y,
        "spacing_x_mm": spacing_x,
        "n_slices": n_slices,
        "z_spacing_mm": z_spacing_mm,
        "slice_thickness_tag_mm": slice_thickness_tag,
        "spacing_between_slices_mm": spacing_between,
    }


def _worker(study_path: str) -> dict | None:
    return analyze_study(study_path)


def build_dataframe(training_path: str, workers: int, max_studies: int | None) -> pd.DataFrame:
    studies = sorted(
        d
        for d in os.listdir(training_path)
        if os.path.isdir(os.path.join(training_path, d))
    )
    if max_studies is not None:
        studies = studies[:max_studies]

    paths = [os.path.join(training_path, s) for s in studies]
    rows: list[dict] = []

    if workers <= 1:
        for p in paths:
            out = analyze_study(p)
            if out is not None:
                rows.append(out)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_worker, p): p for p in paths}
            for fut in as_completed(futs):
                try:
                    out = fut.result()
                    if out is not None:
                        rows.append(out)
                except Exception:
                    pass

    return pd.DataFrame(rows)


def plot_histograms(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    df = df.copy()
    df["plane_key"] = df["rows"].astype(str) + "x" + df["cols"].astype(str)
    counts = df["plane_key"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(counts))))
    counts.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Number of studies")
    ax.set_ylabel("In-plane size (rows x cols)")
    ax.set_title("Distribution of axial plane dimensions")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_1_in_plane_dimensions.png"), dpi=150)
    plt.close(fig)

    # Round to 2 decimals for grouping similar spacings (full precision remains in CSV)
    sy = df["spacing_y_mm"].round(2)
    sx = df["spacing_x_mm"].round(2)
    df["spacing_key"] = "(" + sy.astype(str) + " x " + sx.astype(str) + ") mm"
    sc = df["spacing_key"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(sc))))
    sc.plot(kind="barh", ax=ax, color="seagreen")
    ax.set_xlabel("Number of studies")
    ax.set_ylabel("PixelSpacing (row x col) mm")
    ax.set_title("Distribution of in-plane voxel spacing (DICOM PixelSpacing)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_2_in_plane_spacing_mm.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df["spacing_y_mm"].dropna(), bins=30, color="coral", edgecolor="black", alpha=0.85)
    axes[0].set_xlabel("spacing_y (mm)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Row spacing (PixelSpacing[0])")
    axes[1].hist(df["spacing_x_mm"].dropna(), bins=30, color="coral", edgecolor="black", alpha=0.85)
    axes[1].set_xlabel("spacing_x (mm)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Column spacing (PixelSpacing[1])")
    fig.suptitle("In-plane spacing (mm) — marginal histograms")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_2b_in_plane_spacing_marginal.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        df["n_slices"],
        bins=min(50, max(10, df["n_slices"].nunique())),
        color="mediumpurple",
        edgecolor="black",
        alpha=0.85,
    )
    ax.set_xlabel("Number of slices (per study)")
    ax.set_ylabel("Number of studies")
    ax.set_title("Distribution of slice count (z)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_3_num_slices.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    zs = df["z_spacing_mm"].dropna()
    if len(zs):
        axes[0].hist(zs, bins=min(40, max(10, int(zs.nunique()))), color="teal", edgecolor="black", alpha=0.85)
    axes[0].set_xlabel("mm (SpacingBetweenSlices or mean |Δz| from IPP)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Z-spacing between consecutive slices")

    st = df["slice_thickness_tag_mm"].dropna()
    if len(st):
        axes[1].hist(st, bins=min(30, max(8, int(st.nunique()))), color="darkorange", edgecolor="black", alpha=0.85)
    axes[1].set_xlabel("mm (DICOM SliceThickness)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("SliceThickness tag (when present)")
    fig.suptitle("Slice spacing and nominal thickness")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_4_slice_thickness_and_z_spacing.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="DICOM training set distribution histograms")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--max-studies", type=int, default=None, help="Limit studies (debug)")
    parser.add_argument(
        "--from-csv",
        type=str,
        default=None,
        help="Skip DICOM scan; load this CSV and only regenerate figures",
    )
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "dicom_distribution_per_study.csv")

    if args.from_csv:
        df = pd.read_csv(args.from_csv)
        print(f"Loaded {len(df)} rows from {args.from_csv}")
    else:
        training_path = TRAINING_PATH
        print(f"Training path: {training_path}")
        if not os.path.isdir(training_path):
            raise FileNotFoundError(f"Training path not found: {training_path}")

        df = build_dataframe(training_path, workers=args.workers, max_studies=args.max_studies)
        df.to_csv(csv_path, index=False)
        print(f"Wrote {len(df)} studies to {csv_path}")
    plot_histograms(df, out_dir)
    print(f"Figures saved under {out_dir}")

    print("\n--- Summary ---")
    dim_counts = (df["rows"].astype(str) + "x" + df["cols"].astype(str)).value_counts()
    print("In-plane dimensions (counts):")
    print(dim_counts.head(20).to_string())
    print("\nZ-spacing (mm):")
    print(df["z_spacing_mm"].describe().to_string())


if __name__ == "__main__":
    main()
