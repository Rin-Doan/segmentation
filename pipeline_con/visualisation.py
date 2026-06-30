"""
Visualise pipeline output for all studies in ML_DATA_PATH.

For each study, generates 256 PNG files (one per axial slice), each showing
four panels side-by-side:

  Col 1 – Pred Axial   : coloured ML prediction from pipeline.py at z
  Col 2 – GT Axial     : coloured ground-truth segmentation at z (aligned per evaluation.py)
  Col 3 – Pred Sagittal: middle x-slice of ML prediction with red line at current z
  Col 4 – GT Sagittal  : middle x-slice of GT volume with red line at current z

A colour legend is saved to the output root as legend.png.
Studies already fully rendered are skipped automatically.

Output: /vast/s222440401/pipeline/visualisations/{study_id}/slice_{z:03d}.png

Run from the pipeline/ directory:
    uv run visualisation.py
"""

import csv
import os
import warnings

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from aggregate.aggregate_data import (
    TARGET_SPACING,
    load_nifti_volume,
    resample_to_standard_spacing_3d,
)
from aggregate.data_process import front_crop_or_pad_to_size

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH = "/vast/s222440401/pipeline"
ML_DATA_PATH = DATA_PATH + "/ml_data"
SEGMENTATION_PATH = "/vast/s222440401/segmentations"
FIRST_SLICE_CSV = "./first_slices.csv"
VIS_PATH = DATA_PATH + "/visualisations"

# ── Pipeline constants (must match pipeline.py / evaluation.py) ─────────────
TARGET_SHAPE = (256, 256, 256)
SAG_X = 128  # column index for sagittal cut (middle of 256-wide volume)

# ── Segmentation colour table: class index → RGB uint8 ─────────────────────
#   0 = background, 1–7 = C1–C7, 8 = other (T1 and below)
SEG_COLORS = np.array(
    [
        [0,   0,   0  ],  # 0  background
        [220, 50,  50 ],  # 1  C1
        [240, 140, 40 ],  # 2  C2
        [230, 210, 50 ],  # 3  C3
        [60,  180, 60 ],  # 4  C4
        [50,  190, 220],  # 5  C5
        [60,  100, 220],  # 6  C6
        [170, 60,  220],  # 7  C7
        [160, 160, 160],  # 8  other
    ],
    dtype=np.uint8,
)

CLASS_NAMES = ["background", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "other"]

# ── Canvas layout ────────────────────────────────────────────────────────────
PANEL_W, PANEL_H = 256, 256
HEADER_H = 22          # height of the label strip at the top
GAP = 2                # pixel gap between panels (light-grey fill)
CANVAS_W = PANEL_W * 4 + GAP * 3
CANVAS_H = HEADER_H + PANEL_H

PANEL_LABELS = ["Pred Axial", "GT Axial", "Pred Sagittal", "GT Sagittal"]
LINE_COLOR = np.array([255, 60, 60], dtype=np.uint8)   # red indicator line
_BLANK = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
_GAP_COL = np.full((PANEL_H, GAP, 3), 180, dtype=np.uint8)


# ── I/O helpers ──────────────────────────────────────────────────────────────

def get_study_id(filename: str) -> str:
    return filename.replace(".nii.gz", "").replace(".nii", "")


def find_file(directory: str, study_id: str) -> str | None:
    for ext in (".nii", ".nii.gz"):
        p = os.path.join(directory, f"{study_id}{ext}")
        if os.path.exists(p):
            return p
    return None


def load_first_slices(csv_path: str = FIRST_SLICE_CSV) -> dict[str, int]:
    offsets: dict[str, int] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            offsets[row["study_id"]] = int(row["first_slice"])
    return offsets


def discover_studies() -> list[str]:
    return sorted(
        get_study_id(f)
        for f in os.listdir(ML_DATA_PATH)
        if f.endswith((".nii", ".nii.gz"))
    )


def load_prediction(study_id: str) -> np.ndarray | None:
    """Load the multiclass ML prediction (256, 256, 256) written by pipeline.py."""
    path = find_file(ML_DATA_PATH, study_id)
    if path is None:
        return None
    return nib.load(path).get_fdata().astype(np.int64)


def load_ground_truth(study_id: str, first_slice: int) -> np.ndarray | None:
    """Load GT segmentation aligned to the prediction grid (mirrors evaluation.py)."""
    seg_path = find_file(SEGMENTATION_PATH, study_id)
    if seg_path is None:
        return None
    seg, spacing = load_nifti_volume(seg_path)
    seg = resample_to_standard_spacing_3d(seg, spacing, TARGET_SPACING, order=0)
    if 0 < first_slice < seg.shape[0]:
        seg = seg[first_slice:]
    seg = front_crop_or_pad_to_size(seg, TARGET_SHAPE)
    return np.where(seg > 7, 8, seg).astype(np.int64)


# ── Pixel rendering helpers ───────────────────────────────────────────────────

def seg_to_rgb(seg_2d: np.ndarray) -> np.ndarray:
    """(H, W) int [0–8] → (H, W, 3) uint8 using SEG_COLORS table."""
    return SEG_COLORS[np.clip(seg_2d.astype(np.int64), 0, 8)]


def _build_header() -> np.ndarray:
    """(HEADER_H, CANVAS_W, 3) strip with centred panel labels."""
    img = Image.new("RGB", (CANVAS_W, HEADER_H), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    for i, label in enumerate(PANEL_LABELS):
        x = i * (PANEL_W + GAP) + PANEL_W // 2
        try:
            draw.text((x, HEADER_H // 2), label, fill=(220, 220, 220), anchor="mm")
        except TypeError:
            # anchor kwarg not available in older PIL
            draw.text((x - len(label) * 3, 4), label, fill=(220, 220, 220))
    return np.array(img)


_HEADER = _build_header()


def build_legend() -> Image.Image:
    """Small image showing the class colour → name mapping."""
    swatch, pad = 18, 8
    rows = len(CLASS_NAMES)
    w, h = 220, pad + rows * (swatch + 4) + pad
    img = Image.new("RGB", (w, h), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    draw.text((pad, pad // 2), "Segmentation labels", fill=(220, 220, 220))
    for i, name in enumerate(CLASS_NAMES):
        y = pad + 14 + i * (swatch + 4)
        col = tuple(int(c) for c in SEG_COLORS[i])
        draw.rectangle([pad, y, pad + swatch, y + swatch], fill=col, outline=(200, 200, 200))
        draw.text((pad + swatch + 6, y + 2), f"{i} – {name}", fill=(220, 220, 220))
    return img


# ── Per-study renderer ────────────────────────────────────────────────────────

def render_study(
    study_id: str,
    pred_vol: np.ndarray | None,
    gt_vol: np.ndarray | None,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Sagittal slices are constant across z; only the indicator line moves.
    sag_pred_base = seg_to_rgb(pred_vol[:, :, SAG_X]) if pred_vol is not None else None
    sag_gt_base   = seg_to_rgb(gt_vol[:, :, SAG_X])   if gt_vol is not None   else None

    for z in tqdm(range(TARGET_SHAPE[0]), desc=f"  {study_id}", leave=False):
        # Panel 1 – axial ML prediction
        p1 = seg_to_rgb(pred_vol[z, :, :]) if pred_vol is not None else _BLANK.copy()

        # Panel 2 – axial GT
        p2 = seg_to_rgb(gt_vol[z, :, :]) if gt_vol is not None else _BLANK.copy()

        # Panel 3 – sagittal prediction with indicator line at row z
        if sag_pred_base is not None:
            p3 = sag_pred_base.copy()
            p3[z, :] = LINE_COLOR
        else:
            p3 = _BLANK.copy()

        # Panel 4 – sagittal GT with indicator line at row z
        if sag_gt_base is not None:
            p4 = sag_gt_base.copy()
            p4[z, :] = LINE_COLOR
        else:
            p4 = _BLANK.copy()

        row = np.concatenate([p1, _GAP_COL, p2, _GAP_COL, p3, _GAP_COL, p4], axis=1)
        canvas = np.concatenate([_HEADER, row], axis=0)

        Image.fromarray(canvas).save(
            os.path.join(out_dir, f"slice_{z:03d}.png"),
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Pipeline Visualisation")
    print("=" * 60)
    print(f"ML predictions: {ML_DATA_PATH}")
    print(f"GT segs:        {SEGMENTATION_PATH}")
    print(f"First slices:   {FIRST_SLICE_CSV}")
    print(f"Output:         {VIS_PATH}")
    print()

    if not os.path.isfile(FIRST_SLICE_CSV):
        raise FileNotFoundError(
            f"First-slice CSV not found: {FIRST_SLICE_CSV}  (run pipeline.py first)"
        )

    first_slices = load_first_slices()
    studies = discover_studies()
    print(f"Studies to visualise: {len(studies)}\n")

    os.makedirs(VIS_PATH, exist_ok=True)
    legend_path = os.path.join(VIS_PATH, "legend.png")
    build_legend().save(legend_path)
    print(f"Colour legend saved → {legend_path}\n")

    for study_id in tqdm(studies, desc="Studies"):
        out_dir = os.path.join(VIS_PATH, study_id)
        # Skip if already fully rendered.
        if os.path.isdir(out_dir) and len(os.listdir(out_dir)) >= TARGET_SHAPE[0]:
            tqdm.write(f"  Skipping {study_id} (already rendered)")
            continue

        first_slice = first_slices.get(study_id, 0)
        pred_vol = load_prediction(study_id)
        gt_vol = load_ground_truth(study_id, first_slice)

        if pred_vol is None:
            tqdm.write(f"  Warning: no ML prediction found for {study_id}")
        if gt_vol is None:
            tqdm.write(f"  Warning: no GT segmentation found for {study_id}")

        render_study(study_id, pred_vol, gt_vol, out_dir)

    print(f"\nDone. All visualisations saved under {VIS_PATH}/")


if __name__ == "__main__":
    main()
