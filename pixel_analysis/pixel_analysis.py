"""
Pixel (voxel) count analysis per vertebrae label across all segmentation files.

For each vertebrae label: reports max, min, and mean voxel count,
and plots distribution histograms.

Data source: /vast/s222440401/segmentations  (NIfTI files produced by
aggregate_data/aggregate_work_50/aggregate_data.py)
"""

import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import nibabel as nib
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = '/vast/s222440401'
SEGMENTATION_PATH = DATA_PATH + '/segmentations'
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Vertebrae label map (VerSe / TotalSegmentator convention, labels 1-25)
# Labels not present in the data are simply not printed.
# ---------------------------------------------------------------------------
VERTEBRAE_NAMES = {
    1:  'C1',  2:  'C2',  3:  'C3',  4:  'C4',  5:  'C5',  6:  'C6',  7:  'C7',
    8:  'T1',  9:  'T2', 10:  'T3', 11:  'T4', 12:  'T5', 13:  'T6', 14:  'T7',
    15: 'T8', 16:  'T9', 17: 'T10', 18: 'T11', 19: 'T12',
    20: 'L1', 21:  'L2', 22:  'L3', 23:  'L4', 24:  'L5',
    25: 'Sacrum',
}


def collect_pixel_counts(seg_dir: str) -> dict[int, list[int]]:
    """
    Iterate over every NIfTI segmentation and count voxels per label.

    Returns
    -------
    counts : dict mapping label_id -> list of per-study voxel counts
             (only includes studies where that label is present)
    """
    files = sorted(
        f for f in os.listdir(seg_dir)
        if f.endswith(('.nii', '.nii.gz'))
    )
    if not files:
        raise FileNotFoundError(f"No NIfTI files found in {seg_dir}")

    counts: dict[int, list[int]] = defaultdict(list)

    print(f"Found {len(files)} segmentation files. Computing voxel counts …")
    for fname in tqdm(files, unit='study'):
        seg = nib.load(os.path.join(seg_dir, fname)).get_fdata().astype(np.int32)
        labels, cnts = np.unique(seg, return_counts=True)
        for lbl, cnt in zip(labels, cnts):
            if lbl == 0:
                continue          # skip background
            counts[int(lbl)].append(int(cnt))

    return counts


def print_summary(counts: dict[int, list[int]]) -> None:
    """Print a formatted table: label | name | n_studies | min | mean | max."""
    header = f"{'Label':>6}  {'Name':<10}  {'Studies':>7}  {'Min':>10}  {'Mean':>10}  {'Max':>10}"
    print("\n" + "=" * len(header))
    print("Voxel-count statistics per vertebrae (across all studies)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for lbl in sorted(counts):
        arr = np.array(counts[lbl])
        name = VERTEBRAE_NAMES.get(lbl, f'Label {lbl}')
        print(
            f"{lbl:>6}  {name:<10}  {len(arr):>7}  "
            f"{arr.min():>10,}  {arr.mean():>10,.1f}  {arr.max():>10,}"
        )
    print("=" * len(header))


def save_stats_csv(counts: dict[int, list[int]], out_path: str) -> None:
    """Save per-label summary stats as CSV."""
    import csv
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'name', 'n_studies', 'min', 'mean', 'max', 'std'])
        for lbl in sorted(counts):
            arr = np.array(counts[lbl])
            writer.writerow([
                lbl,
                VERTEBRAE_NAMES.get(lbl, f'Label {lbl}'),
                len(arr),
                int(arr.min()),
                round(float(arr.mean()), 1),
                int(arr.max()),
                round(float(arr.std()), 1),
            ])
    print(f"\nStats saved → {out_path}")


def plot_distributions(counts: dict[int, list[int]], out_path: str) -> None:
    """
    Plot per-vertebrae voxel-count distributions as a grid of histograms.
    Each subplot = one label, x-axis = voxel count, with vertical lines for
    min / mean / max.
    """
    labels = sorted(counts)
    n = len(labels)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(ncols * 4.5, nrows * 3.2))
    fig.suptitle(
        'Voxel-count distribution per vertebrae\n'
        '(across all studies; dashed = min/max, solid = mean)',
        fontsize=13, y=1.01,
    )

    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.55, wspace=0.35)

    for idx, lbl in enumerate(labels):
        arr = np.array(counts[lbl])
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])

        ax.hist(arr, bins=min(20, len(arr)), color='steelblue', edgecolor='white',
                linewidth=0.5, alpha=0.85)
        ax.axvline(arr.mean(), color='crimson', linewidth=1.6, label=f'mean {arr.mean():,.0f}')
        ax.axvline(arr.min(),  color='darkorange', linestyle='--', linewidth=1.2,
                   label=f'min {arr.min():,}')
        ax.axvline(arr.max(),  color='green',      linestyle='--', linewidth=1.2,
                   label=f'max {arr.max():,}')

        name = VERTEBRAE_NAMES.get(lbl, f'Label {lbl}')
        ax.set_title(f'{name}  (n={len(arr)})', fontsize=9, fontweight='bold')
        ax.set_xlabel('Voxel count', fontsize=7.5)
        ax.set_ylabel('Studies', fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e3:.0f}k'))
        ax.legend(fontsize=6, framealpha=0.7)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        fig.add_subplot(gs[idx // ncols, idx % ncols]).set_visible(False)

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Distribution plot saved → {out_path}")


def plot_summary_bar(counts: dict[int, list[int]], out_path: str) -> None:
    """
    Single bar-chart overview: mean voxel count per vertebrae with error bars
    (±1 std) and min/max markers.
    """
    labels = sorted(counts)
    names  = [VERTEBRAE_NAMES.get(l, f'L{l}') for l in labels]
    means  = [np.mean(counts[l]) for l in labels]
    stds   = [np.std(counts[l])  for l in labels]
    mins_  = [np.min(counts[l])  for l in labels]
    maxs_  = [np.max(counts[l])  for l in labels]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9), 5))

    bars = ax.bar(x, means, color='steelblue', alpha=0.8, label='Mean')
    ax.errorbar(x, means, yerr=stds, fmt='none', color='black',
                capsize=3, linewidth=1.2, label='±1 std')
    ax.scatter(x, mins_, marker='v', color='darkorange', zorder=5,
               s=40, label='Min')
    ax.scatter(x, maxs_, marker='^', color='green', zorder=5,
               s=40, label='Max')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Voxel count', fontsize=10)
    ax.set_title('Mean voxel count per vertebrae (±1 std, with min/max)', fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1e3:.0f}k'))
    ax.legend(fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Summary bar chart saved → {out_path}")


def main():
    counts = collect_pixel_counts(SEGMENTATION_PATH)
    print_summary(counts)
    save_stats_csv(counts, os.path.join(OUTPUT_DIR, 'pixel_stats.csv'))
    plot_distributions(counts, os.path.join(OUTPUT_DIR, 'pixel_distribution_per_vertebrae.png'))
    plot_summary_bar(counts, os.path.join(OUTPUT_DIR, 'pixel_summary_bar.png'))
    print("\nDone.")


if __name__ == '__main__':
    main()
