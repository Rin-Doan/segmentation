"""
For each study and each vertebra level (C1–T1):
  - Isolate that vertebra (zero all other labels)
  - Find the suitable projection plane via PCA:
      * Collect all voxel coords of the label  →  (N, 3) array
      * SVD on centred coords → 3 principal components
      * PC3 (smallest variance) = canal axis = normal to the suitable plane
      * PC1, PC2 span the plane in which the vertebra is most spread out
  - Project every voxel onto the PC1-PC2 plane  →  2D scatter
  - Rasterise to a pixel image (one pixel per voxel footprint)
  - Save to  projections/<study_id>/<label_name>.png
"""

import os
import numpy as np
import nibabel as nib
from PIL import Image

SEG_DIR  = "/vast/s222440401/triage_database/ml_data/ml_predictions"
OUT_DIR  = "/vast/s222440401/triage_database/projections"
CROP_PAD = 10          # pixel padding around the rasterised image

LABEL_NAMES = {
    1: "C1", 2: "C2", 3: "C3", 4: "C4",
    5: "C5", 6: "C6", 7: "C7", 8: "T1",
}

LABEL_RGB = {
    1: (230,  38,  26),
    2: (242, 115,  13),
    3: (242, 204,  26),
    4: ( 51, 199,  77),
    5: ( 26, 179, 166),
    6: ( 38, 102, 217),
    7: (140,  38, 204),
    8: (166, 166, 166),
}
BG_RGB = (13, 13, 26)


# ── PCA projection ─────────────────────────────────────────────────────────────

# Voxel coordinate system is (z, y, x):
#   z increases inferior  →  C1 at small z, T1 at large z
#   y  =  anterior–posterior
#   x  =  left–right
#
# Anatomical anchors used to fix PC sign ambiguity:
#   PC1 (horizontal axis of image) →  patient's RIGHT   (+x = [0,0,1])
#   PC2 (vertical axis of image)   →  patient's ANTERIOR (+y = [0,1,0])
#   PC3 (plane normal / canal axis) →  INFERIOR          (+z = [1,0,0])
#
# After sign-fixing:
#   moving RIGHT in the output image  = moving toward patient's right
#   moving DOWN  in the output image  = moving toward patient's anterior
# This orientation is identical for every vertebra and every study.

_WORLD_RIGHT    = np.array([0., 0., 1.])   # +x in (z,y,x)
_WORLD_ANTERIOR = np.array([0., 1., 0.])   # +y in (z,y,x)
_WORLD_INFERIOR = np.array([1., 0., 0.])   # +z in (z,y,x)


def _fix_sign(vec, reference):
    """Return vec or -vec so that dot(result, reference) > 0."""
    return vec if np.dot(vec, reference) >= 0 else -vec


def pca_project(volume, label):
    """
    Suitable-plane projection of `label` using PCA.

    1. Collect all voxel coords  →  (N, 3) float
    2. Centre, then SVD
    3. Fix each PC sign using anatomical anchors (see module header)
    4. Project onto (PC1, PC2)  →  (N, 2) with consistent orientation
    5. Rasterise to 2D bool image

    Returns
    -------
    image : 2D bool array (rows = AP axis, cols = LR axis)
    pc3   : (3,) — canal axis (plane normal), points inferior
    S     : (3,) — singular values (spread per axis)
    """
    coords = np.argwhere(volume == label).astype(np.float64)   # (N, 3)
    if len(coords) < 3:
        return None, None, None

    centred = coords - coords.mean(axis=0)
    _, S, Vt = np.linalg.svd(centred, full_matrices=False)

    # ── fix sign of every PC to match anatomical anchors ──────────────────────
    pc1 = _fix_sign(Vt[0], _WORLD_RIGHT)     # horizontal: left → right
    pc2 = _fix_sign(Vt[1], _WORLD_ANTERIOR)  # vertical:   posterior → anterior
    pc3 = _fix_sign(Vt[2], _WORLD_INFERIOR)  # normal:     superior → inferior

    # ── project onto the suitable plane ───────────────────────────────────────
    col_coords = centred @ pc1    # (N,)  positive = right
    row_coords = centred @ pc2    # (N,)  positive = anterior  →  bottom of image

    # Shift to non-negative pixel indices
    col_px = np.round(col_coords - col_coords.min()).astype(int)
    row_px = np.round(row_coords - row_coords.min()).astype(int)

    H = row_px.max() + 1
    W = col_px.max() + 1
    img = np.zeros((H, W), dtype=bool)
    img[row_px, col_px] = True

    return img, pc3, S


def add_padding(img_bool, pad=CROP_PAD):
    return np.pad(img_bool, pad, mode="constant", constant_values=False)


# ── saving ─────────────────────────────────────────────────────────────────────

def save_png(img_bool, label, out_path):
    fg  = np.array(LABEL_RGB.get(label, (200, 200, 200)), dtype=np.uint8)
    H, W = img_bool.shape
    rgb = np.full((H, W, 3), BG_RGB, dtype=np.uint8)
    rgb[img_bool] = fg
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(out_path)


# ── per-study processing ───────────────────────────────────────────────────────

def process_study(nii_path):
    study_id = os.path.splitext(os.path.basename(nii_path))[0]

    img    = nib.load(nii_path)
    volume = img.get_fdata().astype(np.int16)   # (z, y, x)

    present = [l for l in range(1, 9) if np.any(volume == l)]
    if not present:
        return

    saved = []
    for label in present:
        label_name = LABEL_NAMES[label]

        proj_img, pc3, S = pca_project(volume, label)
        if proj_img is None:
            continue

        proj_img = add_padding(proj_img)

        out_path = os.path.join(OUT_DIR, study_id, f"{label_name}.png")
        save_png(proj_img, label, out_path)
        saved.append(label_name)

    print(f"{study_id[-12:]}  →  {'  '.join(saved)}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    seg_files = sorted(
        os.path.join(SEG_DIR, f)
        for f in os.listdir(SEG_DIR)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    )
    print(f"Found {len(seg_files)} studies\n")
    for nii_path in seg_files:
        process_study(nii_path)
    print(f"\nDone. PCA projections saved to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
