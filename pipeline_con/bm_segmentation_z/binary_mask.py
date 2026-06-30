"""
Create binary masks from raw NIfTI segmentations.

Loads segmentations with the same coordinate correction used in
aggregate_work_50/aggregate_data.py, then binarizes labels (any label > 0 -> 1).
"""

import os
import warnings

import nibabel as nib
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

DATA_PATH = "../../../../../vast/s222440401"
SEGMENTATION_PATH = os.path.join(DATA_PATH, "segmentations")
OUTPUT_PATH = os.path.join(DATA_PATH, "bm_data/binary_mask")


def get_spacing_from_nifti_3d(nii):
    """Extract (z, y, x) spacing from a NIfTI header."""
    try:
        pixdim = nii.header.get_zooms()
        return (float(pixdim[2]), float(pixdim[1]), float(pixdim[0]))
    except Exception as e:
        print(f"Warning: Could not extract NIfTI spacing: {e}")
        return (1.0, 1.0, 1.0)


def load_nifti_volume(seg_path):
    """Load a segmentation volume with project-standard coordinate correction."""
    nii = nib.load(seg_path)
    seg_volume = nii.get_fdata()
    seg_volume = seg_volume[:, ::-1, ::-1].transpose(2, 1, 0)
    seg_volume = seg_volume.astype(np.int32)
    spacing = get_spacing_from_nifti_3d(nii)
    return seg_volume, spacing


def create_binary_mask(seg_volume):
    """Convert multi-label segmentation to binary mask (any vertebra label > 0 -> 1)."""
    return (seg_volume > 0).astype(np.uint8)


def build_affine(spacing):
    """Build a diagonal affine for corrected (z, y, x) volumes."""
    affine = np.eye(4)
    affine[0, 0] = spacing[2]  # x
    affine[1, 1] = spacing[1]  # y
    affine[2, 2] = spacing[0]  # z
    return affine


def process_segmentations():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    segmentation_files = sorted(
        f for f in os.listdir(SEGMENTATION_PATH) if f.endswith((".nii", ".nii.gz"))
    )

    print("=" * 60)
    print("Binary Mask Generation")
    print("=" * 60)
    print(f"Input:  {os.path.abspath(SEGMENTATION_PATH)}")
    print(f"Output: {os.path.abspath(OUTPUT_PATH)}")
    print(f"Found {len(segmentation_files)} segmentation files")

    saved = 0
    skipped = 0

    for filename in tqdm(segmentation_files, desc="Creating binary masks"):
        study_id = filename.replace(".nii.gz", "").replace(".nii", "")
        seg_path = os.path.join(SEGMENTATION_PATH, filename)
        output_path = os.path.join(OUTPUT_PATH, f"{study_id}.nii")

        try:
            seg_volume, spacing = load_nifti_volume(seg_path)
            binary_mask = create_binary_mask(seg_volume)
            affine = build_affine(spacing)
            nib.save(nib.Nifti1Image(binary_mask, affine), output_path)
            saved += 1
        except Exception as e:
            skipped += 1
            print(f"Failed to process {study_id}: {e}")

    print("\n" + "=" * 60)
    print("Binary mask generation complete")
    print(f"Saved:   {saved}")
    print(f"Skipped: {skipped}")
    print(f"Format:  .nii")
    print("=" * 60)


if __name__ == "__main__":
    process_segmentations()
