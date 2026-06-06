"""
Fracture vs non-fracture statistics from ground-truth CSV (vertebra-level labels).

Expects columns: StudyInstanceUID, vertebrae, broken (0 = not fractured, 1 = fractured).
Default CSV: 3D_Classification_nii/ground_truth.csv in the project repo.

Outputs: console summary, CSV tables, and figures under ./output/
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_paths import resolve_data_path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GROUND_TRUTH_CSV = _PROJECT_ROOT / "3D_Classification_nii" / "ground_truth.csv"

DATA_PATH = resolve_data_path()
SEGMENTATION_PATH = DATA_PATH / "segmentations"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def _study_ids_with_segmentation() -> set[str] | None:
    if not SEGMENTATION_PATH.is_dir():
        return None
    out = set()
    for f in os.listdir(SEGMENTATION_PATH):
        if f.endswith(".nii.gz"):
            out.add(f.replace(".nii.gz", ""))
        elif f.endswith(".nii") and not f.endswith(".nii.gz"):
            out.add(f.replace(".nii", ""))
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not GROUND_TRUTH_CSV.is_file():
        print(f"Ground truth CSV not found: {GROUND_TRUTH_CSV}")
        return

    df = pd.read_csv(GROUND_TRUTH_CSV)
    required = {"StudyInstanceUID", "vertebrae", "broken"}
    if not required.issubset(df.columns):
        print(f"CSV must contain columns {required}, got {list(df.columns)}")
        return

    df = df.copy()
    df["broken"] = df["broken"].astype(int)

    seg_studies = _study_ids_with_segmentation()
    df_seg = df[df["StudyInstanceUID"].astype(str).isin(seg_studies)] if seg_studies else df

    def summarize(frame: pd.DataFrame, tag: str):
        n = len(frame)
        n_pos = int((frame["broken"] == 1).sum())
        n_neg = int((frame["broken"] == 0).sum())
        print(f"\n=== {tag} ===")
        print(f"Rows (study × vertebra): {n:,}")
        print(f"  Fractured (broken=1):     {n_pos:>8,}  ({100.0 * n_pos / n:.4f}%)" if n else "")
        print(f"  Not fractured (broken=0): {n_neg:>8,}  ({100.0 * n_neg / n:.4f}%)" if n else "")

        studies_any = frame.groupby("StudyInstanceUID")["broken"].max()
        n_st = len(studies_any)
        n_st_frac = int((studies_any == 1).sum())
        n_st_clean = n_st - n_st_frac
        print(f"\nStudies (unique StudyInstanceUID): {n_st:,}")
        if n_st:
            print(f"  With ≥1 fractured vertebra: {n_st_frac:>8,}  ({100.0 * n_st_frac / n_st:.4f}%)")
            print(f"  With no fractured vertebra: {n_st_clean:>8,}  ({100.0 * n_st_clean / n_st:.4f}%)")

        by_v = frame.groupby("vertebrae")["broken"].agg(["sum", "count"])
        by_v["frac_rate_%"] = 100.0 * by_v["sum"] / by_v["count"]
        by_v = by_v.rename(columns={"sum": "n_fractured", "count": "n_total"})
        print("\nPer-vertebra (row counts):")
        print(by_v.to_string())

        return {
            "tag": tag,
            "n_rows": n,
            "n_fractured_rows": n_pos,
            "n_not_fractured_rows": n_neg,
            "pct_fractured_rows": 100.0 * n_pos / n if n else 0.0,
            "n_studies": n_st,
            "n_studies_with_any_fracture": n_st_frac,
            "n_studies_without_fracture": n_st_clean,
            "by_vertebra": by_v,
        }

    r_all = summarize(df, "All rows in ground_truth.csv")
    if seg_studies is not None and len(df_seg) > 0:
        summarize(df_seg, "Rows for studies that have a file in segmentations/")

    # Save tables
    summary_rows = [
        {
            "subset": r_all["tag"],
            "n_study_vertebra_rows": r_all["n_rows"],
            "n_fractured": r_all["n_fractured_rows"],
            "n_not_fractured": r_all["n_not_fractured_rows"],
            "pct_fractured": round(r_all["pct_fractured_rows"], 6),
            "n_unique_studies": r_all["n_studies"],
            "n_studies_any_fracture": r_all["n_studies_with_any_fracture"],
            "n_studies_no_fracture": r_all["n_studies_without_fracture"],
        }
    ]
    if seg_studies is not None and len(df_seg) > 0:
        studies_any = df_seg.groupby("StudyInstanceUID")["broken"].max()
        n_st = len(studies_any)
        n_pos = int((df_seg["broken"] == 1).sum())
        n_neg = int((df_seg["broken"] == 0).sum())
        n_st_frac = int((studies_any == 1).sum())
        summary_rows.append(
            {
                "subset": "intersection_with_segmentation_folder",
                "n_study_vertebra_rows": len(df_seg),
                "n_fractured": n_pos,
                "n_not_fractured": n_neg,
                "pct_fractured": round(100.0 * n_pos / len(df_seg), 6) if len(df_seg) else 0.0,
                "n_unique_studies": n_st,
                "n_studies_any_fracture": n_st_frac,
                "n_studies_no_fracture": n_st - n_st_frac,
            }
        )
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "fracture_summary.csv", index=False)
    r_all["by_vertebra"].to_csv(OUTPUT_DIR / "fracture_counts_by_vertebra.csv")

    # Plots (all CSV rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels_pie = ["Not fractured", "Fractured"]
    sizes = [r_all["n_not_fractured_rows"], r_all["n_fractured_rows"]]
    colors = ["#6A994E", "#C73E1D"]
    axes[0].pie(
        sizes,
        labels=[f"{labels_pie[i]}\n{sizes[i]:,}\n({100*sizes[i]/r_all['n_rows']:.2f}%)" for i in range(2)],
        colors=colors,
        startangle=90,
    )
    axes[0].set_title("Study×vertebra rows: fracture label")

    bv = r_all["by_vertebra"].sort_index()
    x = np.arange(len(bv))
    axes[1].bar(x, bv["frac_rate_%"].values, color="#2E86AB", edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bv.index.tolist(), rotation=0)
    axes[1].set_ylabel("Fracture rate (%)")
    axes[1].set_title("Share of fractured labels per vertebra")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fracture_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nSaved tables and figures under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
