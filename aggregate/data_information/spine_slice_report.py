"""
Build a short report on spine stack extent vs equivalent slice count at 0.6–0.7 mm spacing.

Uses `output/dicom_distribution_per_study.csv` from analyze_distribution.py.

Formula note
------------
The expression (z_spacing * n_slices) / n_slices simplifies to z_spacing alone.
Here we interpret "total study (z spacing * number of slice)" as the **total
extent along the stack in mm**: spine_length_mm = z_spacing_mm * n_slices.
If you instead need center-to-center span, consider z_spacing * (n_slices - 1).
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

_Z_LO = 0.6
_Z_HI = 0.7


def main() -> None:
    parser = argparse.ArgumentParser(description="Spine length / equivalent slice report")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to dicom_distribution_per_study.csv (default: beside this script)",
    )
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv or os.path.join(here, "output", "dicom_distribution_per_study.csv")
    out_dir = os.path.join(here, "output")
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "spine_slice_report.md")
    per_study_csv = os.path.join(out_dir, "spine_slice_per_study.csv")

    df = pd.read_csv(csv_path)
    need = {"z_spacing_mm", "n_slices"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    work = df[["study_id", "n_slices", "z_spacing_mm"]].copy()
    work = work.dropna(subset=["z_spacing_mm", "n_slices"])
    work["n_slices"] = work["n_slices"].astype(int)
    work = work[work["n_slices"] > 0]
    work["z_spacing_mm"] = work["z_spacing_mm"].astype(float)

    # Per study: total stack extent (mm). (z*n)/n would be just z_spacing — see docstring.
    work["spine_length_mm"] = work["z_spacing_mm"] * work["n_slices"]
    work["equiv_slices_at_0.7mm"] = work["spine_length_mm"] / _Z_HI
    work["equiv_slices_at_0.6mm"] = work["spine_length_mm"] / _Z_LO

    mean_spine = float(work["spine_length_mm"].mean())
    std_spine = float(work["spine_length_mm"].std(ddof=1)) if len(work) > 1 else float("nan")
    mean_n = float(work["n_slices"].mean())

    # Step 2: range of "average number of slice" from mean spine / z in [0.6, 0.7]
    n_at_hi_z = mean_spine / _Z_HI  # smaller equivalent count
    n_at_lo_z = mean_spine / _Z_LO  # larger equivalent count
    low, high = min(n_at_hi_z, n_at_lo_z), max(n_at_hi_z, n_at_lo_z)

    work.to_csv(per_study_csv, index=False)

    lines = [
        "# Spine stack extent and equivalent slice count (0.6–0.7 mm)",
        "",
        f"Source: `{os.path.relpath(csv_path, start=here)}`",
        f"Studies used (valid `z_spacing_mm` and `n_slices` > 0): **{len(work)}**",
        "",
        "## Definitions",
        "",
        "- **Per-study spine length (mm)** is taken as `z_spacing_mm * n_slices` "
        "(total extent along the stack if spacing is uniform).",
        "- Your written step `(z_spacing * n_slices) / n_slices` equals **`z_spacing_mm` only**; "
        "this report uses the **product** `z_spacing * n_slices` as stack length.",
        "- **Equivalent slice count** at a reference through-plane spacing `z_ref` mm: "
        "`spine_length_mm / z_ref`.",
        "",
        "## Step 1 — Spine length per study (summary)",
        "",
        "| Metric | mm |",
        "|--------|-----|",
        f"| Mean `spine_length_mm` | {mean_spine:.4f} |",
        f"| Std (sample) | {std_spine:.4f} |",
        f"| Min | {work['spine_length_mm'].min():.4f} |",
        f"| Max | {work['spine_length_mm'].max():.4f} |",
        "",
        f"Mean observed **n_slices** (actual): **{mean_n:.2f}**",
        "",
        "## Step 2 — Range of average equivalent slice count",
        "",
        f"Using **mean** spine length **{mean_spine:.4f} mm** and reference spacing "
        f"**{_Z_LO} mm** to **{_Z_HI} mm**:",
        "",
        f"- `mean_spine / {_Z_HI}` = **{n_at_hi_z:.2f}** slices (at {_Z_HI} mm spacing)",
        f"- `mean_spine / {_Z_LO}` = **{n_at_lo_z:.2f}** slices (at {_Z_LO} mm spacing)",
        "",
        f"**Reported range (equivalent average slice count): [{low:.2f}, {high:.2f}]**",
        "",
        "## Per-study output",
        "",
        f"Full table with `equiv_slices_at_0.7mm` and `equiv_slices_at_0.6mm`: `{os.path.basename(per_study_csv)}`.",
        "",
    ]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {md_path}")
    print(f"Wrote {per_study_csv}")
    print(f"Mean spine_length_mm: {mean_spine:.4f}")
    print(f"Equivalent average slice range [{low:.2f}, {high:.2f}] at z in [{_Z_LO}, {_Z_HI}] mm")


if __name__ == "__main__":
    main()
