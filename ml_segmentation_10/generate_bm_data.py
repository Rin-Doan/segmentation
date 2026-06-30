"""
Generate bone-marrow training volumes from aggregated images and segmentations.

Pipeline
--------
1. Load resampled images from agg_data/images_nii (spacing already 0.5, 0.5, 0.5 mm).
2. Load segmentations from agg_data/segmentations_nii and binarize (label > 0 -> 1).
3. Mask images: image * binary_mask (background set to 0).
4. Save masked images to bm_data/bm_images_nii.
5. Save segmentations to bm_data/bm_segmentations_nii.
6. Save binary masks to bm_data/bm_masks_nii.

First-slice cropping is handled at training time by aggregate_data/data_process.py
via first_slices.csv.
"""

import os
import warnings

import nibabel as nib
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

DATA_PATH = "/vast/s222440401"
IMAGES_PATH = os.path.join(DATA_PATH, "agg_data/images_nii")
SEGMENTATIONS_PATH = os.path.join(DATA_PATH, "agg_data/segmentations_nii")
OUTPUT_IMAGES_PATH = os.path.join(DATA_PATH, "bm_data/bm_images_nii")
OUTPUT_SEG_PATH = os.path.join(DATA_PATH, "bm_data/bm_segmentations_nii")
OUTPUT_MASK_PATH = os.path.join(DATA_PATH, "bm_data/bm_masks_nii")

TARGET_SPACING = (0.5, 0.5, 0.5)  # (z, y, x) in mm — matches aggregate_data.py output


def build_affine(spacing):
    """Build a diagonal affine for (z, y, x) volumes."""
    affine = np.eye(4)
    affine[0, 0] = spacing[2]  # x
    affine[1, 1] = spacing[1]  # y
    affine[2, 2] = spacing[0]  # z
    return affine


def load_binary_mask_from_segmentation(seg_path, target_shape):
    """Load aggregated segmentation and binarize (any label > 0 -> 1)."""
    seg = nib.load(seg_path).get_fdata()
    if seg.shape != target_shape:
        raise ValueError(
            f"Segmentation shape {seg.shape} does not match image shape {target_shape}"
        )
    return (seg > 0).astype(np.uint8)


def get_overlapping_studies():
    """Return study IDs present in both images and segmentations."""
    image_files = {
        f.replace(".nii.gz", "").replace(".nii", "")
        for f in os.listdir(IMAGES_PATH)
        if f.endswith((".nii", ".nii.gz"))
    }
    seg_files = {
        f.replace(".nii.gz", "").replace(".nii", "")
        for f in os.listdir(SEGMENTATIONS_PATH)
        if f.endswith((".nii", ".nii.gz"))
    }
    return sorted(image_files & seg_files)


def generate_bm_data():
    os.makedirs(OUTPUT_IMAGES_PATH, exist_ok=True)
    os.makedirs(OUTPUT_SEG_PATH, exist_ok=True)
    os.makedirs(OUTPUT_MASK_PATH, exist_ok=True)

    study_ids = get_overlapping_studies()
    saved = 0
    skipped = 0
    affine = build_affine(TARGET_SPACING)

    print("=" * 60)
    print("BM Data Generation")
    print("=" * 60)
    print(f"Images:         {os.path.abspath(IMAGES_PATH)}")
    print(f"Segmentations:  {os.path.abspath(SEGMENTATIONS_PATH)}")
    print(f"Output images:  {os.path.abspath(OUTPUT_IMAGES_PATH)}")
    print(f"Output segs:    {os.path.abspath(OUTPUT_SEG_PATH)}")
    print(f"Output masks:   {os.path.abspath(OUTPUT_MASK_PATH)}")
    print(f"Studies:        {len(study_ids)}")
    print(f"Target spacing: {TARGET_SPACING} mm (z, y, x)")
    print("=" * 60)

    for study_id in tqdm(study_ids, desc="Generating BM data"):
        image_path = os.path.join(IMAGES_PATH, f"{study_id}.nii")
        seg_path = os.path.join(SEGMENTATIONS_PATH, f"{study_id}.nii")

        try:
            image = nib.load(image_path).get_fdata().astype(np.float32)
            binary_mask = load_binary_mask_from_segmentation(seg_path, image.shape)

            if not np.any(binary_mask):
                skipped += 1
                print(f"Skipping {study_id}: no vertebra pixels in segmentation")
                continue

            masked_image = (image * binary_mask).astype(np.float32)
            seg = nib.load(seg_path).get_fdata()

            nib.save(
                nib.Nifti1Image(masked_image, affine),
                os.path.join(OUTPUT_IMAGES_PATH, f"{study_id}.nii"),
            )
            nib.save(
                nib.Nifti1Image(seg.astype(np.float32), affine),
                os.path.join(OUTPUT_SEG_PATH, f"{study_id}.nii"),
            )
            nib.save(
                nib.Nifti1Image(binary_mask, affine),
                os.path.join(OUTPUT_MASK_PATH, f"{study_id}.nii"),
            )
            saved += 1
        except Exception as e:
            skipped += 1
            print(f"Failed to process {study_id}: {e}")

    print("\n" + "=" * 60)
    print("BM data generation complete")
    print(f"Saved:   {saved}")
    print(f"Skipped: {skipped}")
    print("=" * 60)


if __name__ == "__main__":
    generate_bm_data()
