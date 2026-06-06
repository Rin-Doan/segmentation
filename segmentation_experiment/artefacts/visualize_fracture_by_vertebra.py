"""
Fracture distribution across vertebrae (C1–C7) from ground_truth.csv.

Stacked bar chart: counts of fractured vs not-fractured labels per level.
Uses the same CSV path and styling conventions as analyze_fracture_vs_normal.py.

Output: ./output/fracture_distribution_by_vertebra.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GROUND_TRUTH_CSV = _PROJECT_ROOT / "3D_Classification_nii" / "ground_truth.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Cervical order; any other labels in CSV are appended after C7, sorted
CERVICAL_ORDER = [f"C{i}" for i in range(1, 8)]


def _vertebra_sort_key(v: str) -> tuple[int, str]:
    v = str(v).strip()
    if v in CERVICAL_ORDER:
        return (0, CERVICAL_ORDER.index(v))
    return (1, v)


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

    g = df.groupby("vertebrae", as_index=False).agg(
        n_fractured=("broken", "sum"),
        n_total=("broken", "count"),
    )
    g["n_normal"] = g["n_total"] - g["n_fractured"]
    g["pct_fractured"] = 100.0 * g["n_fractured"] / g["n_total"]

    levels = sorted(g["vertebrae"].astype(str).unique(), key=_vertebra_sort_key)
    g = g.set_index("vertebrae").reindex(levels).fillna(0).astype(
        {"n_fractured": int, "n_total": int, "n_normal": int, "pct_fractured": float}
    )

    x = np.arange(len(g))
    n_norm = g["n_normal"].values
    n_frac = g["n_fractured"].values

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x,
        n_norm,
        label="Not fractured",
        color="#6A994E",
        edgecolor="black",
        linewidth=0.6,
    )
    ax.bar(
        x,
        n_frac,
        bottom=n_norm,
        label="Fractured",
        color="#C73E1D",
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(g.index.tolist())
    ax.set_xlabel("Vertebra")
    ax.set_ylabel("Count (study × vertebra rows)")
    ax.set_title("Fracture distribution across vertebrae")
    ax.legend(loc="upper right", framealpha=0.95)

    for i in range(len(g)):
        tot = int(g["n_total"].iloc[i])
        if tot == 0:
            continue
        pct = g["pct_fractured"].iloc[i]
        ax.text(
            i,
            tot + max(1, int(0.02 * g["n_total"].sum())),
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )

    plt.tight_layout()
    out = OUTPUT_DIR / "fracture_distribution_by_vertebra.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()

    g.reset_index().to_csv(OUTPUT_DIR / "fracture_distribution_by_vertebra.csv", index=False)

    print(g.to_string())
    print(f"\nSaved: {out}")
    print(f"Saved: {OUTPUT_DIR / 'fracture_distribution_by_vertebra.csv'}")


if __name__ == "__main__":
    main()
