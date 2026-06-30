"""
Inter-vertebrae distance analysis.

For each segmentation study:
  1. Compute the 3-D centroid (in physical mm space) of every vertebrae label.
  2. Measure the Euclidean distance between every pair of consecutive labels
     (1→2, 2→3, …) that are both present in that study.

Across all studies, reports max / min / mean per neighbour pair and saves
a distribution plot.

Data source: /vast/s222440401/segmentations  (raw NIfTI files, before resampling)
"""

import os
from collections import defaultdict

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = '/vast/s222440401'
SEGMENTATION_PATH = DATA_PATH + '/segmentations'
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Vertebrae label map
# ---------------------------------------------------------------------------
VERTEBRAE_NAMES = {
    1:  'C1',  2:  'C2',  3:  'C3',  4:  'C4',  5:  'C5',  6:  'C6',  7:  'C7',
    8:  'T1',  9:  'T2', 10:  'T3', 11:  'T4', 12:  'T5', 13:  'T6', 14:  'T7',
    15: 'T8', 16:  'T9', 17: 'T10', 18: 'T11', 19: 'T12',
    20: 'L1', 21:  'L2', 22:  'L3', 23:  'L4', 24:  'L5',
    25: 'Sacrum',
}


def voxel_to_world(centroid_ijk: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Map a voxel centroid (i, j, k) to world-space mm coordinates via the
    NIfTI affine.  Returns a (3,) array in mm.
    """
    ijk_h = np.array([centroid_ijk[0], centroid_ijk[1], centroid_ijk[2], 1.0])
    return (affine @ ijk_h)[:3]


def collect_distances(seg_dir: str) -> dict[tuple[int, int], list[float]]:
    """
    Iterate over every NIfTI segmentation.

    For each study:
      - Compute the centroid of each present label in world-space mm.
      - Record the Euclidean distance between every consecutive label pair
        (label L → label L+1) that are both present.

    Returns
    -------
    distances : dict mapping (label_a, label_b) -> list of distances in mm
    """
    files = sorted(
        f for f in os.listdir(seg_dir)
        if f.endswith(('.nii', '.nii.gz'))
    )
    if not files:
        raise FileNotFoundError(f"No NIfTI files found in {seg_dir}")

    distances: dict[tuple[int, int], list[float]] = defaultdict(list)

    print(f"Found {len(files)} segmentation files. Computing inter-vertebrae distances …")
    for fname in tqdm(files, unit='study'):
        nii = nib.load(os.path.join(seg_dir, fname))
        seg = nii.get_fdata().astype(np.int32)
        affine = nii.affine                    # voxel → world (mm)

        present_labels = sorted(int(l) for l in np.unique(seg) if l > 0)

        # Compute centroids for ALL labels in a single vectorised pass.
        # np.bincount accumulates i/j/k sums per label without repeated
        # full-volume scans.
        shape = seg.shape
        flat = seg.ravel()
        non_bg = flat > 0
        flat_lbl = flat[non_bg].astype(np.int32)
        flat_idx = np.where(non_bg)[0]

        ci = (flat_idx // (shape[1] * shape[2])).astype(np.float64)
        cj = ((flat_idx % (shape[1] * shape[2])) // shape[2]).astype(np.float64)
        ck = (flat_idx % shape[2]).astype(np.float64)

        max_lbl = int(flat_lbl.max()) + 1
        counts = np.bincount(flat_lbl, minlength=max_lbl)
        sum_i  = np.bincount(flat_lbl, weights=ci, minlength=max_lbl)
        sum_j  = np.bincount(flat_lbl, weights=cj, minlength=max_lbl)
        sum_k  = np.bincount(flat_lbl, weights=ck, minlength=max_lbl)

        centroids: dict[int, np.ndarray] = {}
        for lbl in present_labels:
            if counts[lbl] > 0:
                cijk = np.array([sum_i[lbl], sum_j[lbl], sum_k[lbl]]) / counts[lbl]
                centroids[lbl] = voxel_to_world(cijk, affine)

        # Record distances between consecutive label pairs
        for lbl in present_labels:
            next_lbl = lbl + 1
            if next_lbl in centroids:
                dist = float(np.linalg.norm(centroids[lbl] - centroids[next_lbl]))
                distances[(lbl, next_lbl)].append(dist)

    return distances


def pair_label(pair: tuple[int, int]) -> str:
    a, b = pair
    return f"{VERTEBRAE_NAMES.get(a, str(a))}→{VERTEBRAE_NAMES.get(b, str(b))}"


def print_summary(distances: dict[tuple[int, int], list[float]]) -> None:
    header = (
        f"{'Pair':<12}  {'Studies':>7}  "
        f"{'Min (mm)':>9}  {'Mean (mm)':>10}  {'Max (mm)':>9}"
    )
    print("\n" + "=" * len(header))
    print("Inter-vertebrae distance statistics (Euclidean, world-space mm)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for pair in sorted(distances):
        arr = np.array(distances[pair])
        print(
            f"{pair_label(pair):<12}  {len(arr):>7}  "
            f"{arr.min():>9.1f}  {arr.mean():>10.1f}  {arr.max():>9.1f}"
        )
    print("=" * len(header))


def save_stats_csv(distances: dict[tuple[int, int], list[float]], out_path: str) -> None:
    import csv
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'pair', 'label_a', 'label_b', 'name_a', 'name_b',
            'n_studies', 'min_mm', 'mean_mm', 'max_mm', 'std_mm',
        ])
        for pair in sorted(distances):
            a, b = pair
            arr = np.array(distances[pair])
            writer.writerow([
                pair_label(pair), a, b,
                VERTEBRAE_NAMES.get(a, str(a)),
                VERTEBRAE_NAMES.get(b, str(b)),
                len(arr),
                round(float(arr.min()), 2),
                round(float(arr.mean()), 2),
                round(float(arr.max()), 2),
                round(float(arr.std()), 2),
            ])
    print(f"\nStats saved → {out_path}")


def plot_distributions(
    distances: dict[tuple[int, int], list[float]], out_path: str
) -> None:
    """Grid of per-pair histograms with min / mean / max lines."""
    pairs = sorted(distances)
    n = len(pairs)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(ncols * 4.5, nrows * 3.2))
    fig.suptitle(
        'Inter-vertebrae distance distribution (mm)\n'
        '(dashed = min/max, solid = mean)',
        fontsize=13, y=1.01,
    )
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.6, wspace=0.38)

    for idx, pair in enumerate(pairs):
        arr = np.array(distances[pair])
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])

        ax.hist(arr, bins=min(20, max(5, len(arr))),
                color='teal', edgecolor='white', linewidth=0.5, alpha=0.85)
        ax.axvline(arr.mean(), color='crimson',    linewidth=1.6,
                   label=f'mean {arr.mean():.1f}')
        ax.axvline(arr.min(),  color='darkorange', linewidth=1.2, linestyle='--',
                   label=f'min {arr.min():.1f}')
        ax.axvline(arr.max(),  color='green',      linewidth=1.2, linestyle='--',
                   label=f'max {arr.max():.1f}')

        ax.set_title(f'{pair_label(pair)}  (n={len(arr)})', fontsize=9, fontweight='bold')
        ax.set_xlabel('Distance (mm)', fontsize=7.5)
        ax.set_ylabel('Studies', fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, framealpha=0.7)

    for idx in range(n, nrows * ncols):
        fig.add_subplot(gs[idx // ncols, idx % ncols]).set_visible(False)

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Distribution grid saved → {out_path}")


def plot_summary_bar(
    distances: dict[tuple[int, int], list[float]], out_path: str
) -> None:
    """
    Single bar chart: mean distance per consecutive pair with ±1 std error
    bars and min/max scatter markers.
    """
    pairs  = sorted(distances)
    labels = [pair_label(p) for p in pairs]
    means  = [np.mean(distances[p]) for p in pairs]
    stds   = [np.std(distances[p])  for p in pairs]
    mins_  = [np.min(distances[p])  for p in pairs]
    maxs_  = [np.max(distances[p])  for p in pairs]

    x = np.arange(len(pairs))
    fig, ax = plt.subplots(figsize=(max(10, len(pairs) * 1.1), 5))

    ax.bar(x, means, color='teal', alpha=0.8, label='Mean')
    ax.errorbar(x, means, yerr=stds, fmt='none', color='black',
                capsize=4, linewidth=1.2, label='±1 std')
    ax.scatter(x, mins_, marker='v', color='darkorange', zorder=5,
               s=50, label='Min')
    ax.scatter(x, maxs_, marker='^', color='green', zorder=5,
               s=50, label='Max')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Distance (mm)', fontsize=10)
    ax.set_title(
        'Mean inter-vertebrae distance per consecutive pair (±1 std, with min/max)',
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Summary bar chart saved  → {out_path}")


def main():
    distances = collect_distances(SEGMENTATION_PATH)
    print_summary(distances)
    save_stats_csv(
        distances,
        os.path.join(OUTPUT_DIR, 'distance_stats.csv'),
    )
    plot_distributions(
        distances,
        os.path.join(OUTPUT_DIR, 'distance_distribution_per_pair.png'),
    )
    plot_summary_bar(
        distances,
        os.path.join(OUTPUT_DIR, 'distance_summary_bar.png'),
    )
    print("\nDone.")


if __name__ == '__main__':
    main()
