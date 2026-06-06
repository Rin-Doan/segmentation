"""
Generate bone-marrow training volumes from aggregated images and binary masks.

Pipeline
--------
1. Load resampled images from agg_data_50/images_nii.
2. Load binary masks from bm_data/binary_mask and resample them to the image
   grid (same target spacing as aggregate_data.py).
3. Mask images: keep pixel values where binary mask == 1, set background to 0.
4. Find contiguous vertebra sequences along z, take the first slice of the
   longest sequence, and crop 128 slices from there.
5. Save masked, cropped images to bm_data/bm_images_nii.
6. Record each study's first vertebra slice in a CSV file.
7. Crop the matching 128 slices from agg_data_50/segmentations_nii and save
   to bm_data/bm_segmentations_nii.
"""

import os
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from tqdm import tqdm

warnings.filterwarnings("ignore")

DATA_PATH = "../../../../../vast/s222440401"
IMAGES_PATH = os.path.join(DATA_PATH, "agg_data_50/images_nii")
SEGMENTATIONS_PATH = os.path.join(DATA_PATH, "agg_data_50/segmentations_nii")
BINARY_MASK_PATH = os.path.join(DATA_PATH, "bm_data/binary_mask")
OUTPUT_IMAGES_PATH = os.path.join(DATA_PATH, "bm_data/bm_images_nii")
OUTPUT_SEG_PATH = os.path.join(DATA_PATH, "bm_data/bm_segmentations_nii")
FIRST_SLICE_CSV = os.path.join(DATA_PATH, "bm_data/first_slice.csv")

TARGET_SPACING = (0.7, 0.5, 0.5)  # (z, y, x) in mm
NUM_SLICES = 128


def get_spacing_from_nifti_3d(nii):
    """Extract (z, y, x) spacing from a NIfTI header."""
    pixdim = nii.header.get_zooms()
    return (float(pixdim[2]), float(pixdim[1]), float(pixdim[0]))


def resample_to_target_spacing(volume, original_spacing, target_spacing=TARGET_SPACING, order=0):
    """Resample a 3D volume to the target physical spacing."""
    zoom_factors = tuple(
        original_spacing[i] / target_spacing[i] for i in range(3)
    )
    if all(abs(z - 1.0) < 0.01 for z in zoom_factors):
        return volume
    return zoom(volume, zoom_factors, order=order)


def build_affine(spacing, z_offset=0):
    """Build a diagonal affine for (z, y, x) volumes."""
    affine = np.eye(4)
    affine[0, 0] = spacing[2]  # x
    affine[1, 1] = spacing[1]  # y
    affine[2, 2] = spacing[0]  # z
    affine[2, 3] = z_offset * spacing[0]
    return affine


def load_binary_mask_resampled(mask_path, target_shape):
    """Load a binary mask and resample it to match the aggregated image grid."""
    nii = nib.load(mask_path)
    mask = nii.get_fdata()
    spacing = get_spacing_from_nifti_3d(nii)
    mask = resample_to_target_spacing(mask, spacing, order=0)

    if mask.shape != target_shape:
        raise ValueError(
            f"Resampled mask shape {mask.shape} does not match image shape {target_shape}"
        )
    return (mask > 0).astype(np.uint8)


def find_vertebra_sequences(mask):
    """
    Return contiguous z-index runs where the slice contains vertebra pixels.

    Each run is (start, end_inclusive, length).
    """
    present = np.any(mask > 0, axis=(1, 2))
    indices = np.where(present)[0]
    if indices.size == 0:
        return []

    sequences = []
    start = int(indices[0])
    prev = int(indices[0])

    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        sequences.append((start, prev, prev - start + 1))
        start = idx
        prev = idx

    sequences.append((start, prev, prev - start + 1))
    return sequences


def find_first_slice_of_longest_sequence(mask):
    """
    Return the first slice of the longest contiguous vertebra sequence.

    If multiple sequences tie for longest length, return the earliest one.
    """
    sequences = find_vertebra_sequences(mask)
    if not sequences:
        return None, None

    longest = max(sequences, key=lambda seq: (seq[2], -seq[0]))
    return int(longest[0]), int(longest[2])


def crop_volume(volume, first_slice, num_slices=NUM_SLICES):
    """Crop [first_slice : first_slice + num_slices], zero-padding if needed."""
    end = first_slice + num_slices
    cropped = volume[first_slice:end].copy()

    if cropped.shape[0] < num_slices:
        pad_shape = (num_slices - cropped.shape[0],) + cropped.shape[1:]
        cropped = np.concatenate(
            [cropped, np.zeros(pad_shape, dtype=cropped.dtype)],
            axis=0,
        )
    return cropped


def get_overlapping_studies():
    """Return study IDs present in images, binary masks, and segmentations."""
    image_files = {
        f.replace(".nii.gz", "").replace(".nii", "")
        for f in os.listdir(IMAGES_PATH)
        if f.endswith((".nii", ".nii.gz"))
    }
    mask_files = {
        f.replace(".nii.gz", "").replace(".nii", "")
        for f in os.listdir(BINARY_MASK_PATH)
        if f.endswith((".nii", ".nii.gz"))
    }
    seg_files = {
        f.replace(".nii.gz", "").replace(".nii", "")
        for f in os.listdir(SEGMENTATIONS_PATH)
        if f.endswith((".nii", ".nii.gz"))
    }
    return sorted(image_files & mask_files & seg_files)


def generate_bm_data():
    os.makedirs(OUTPUT_IMAGES_PATH, exist_ok=True)
    os.makedirs(OUTPUT_SEG_PATH, exist_ok=True)

    study_ids = get_overlapping_studies()
    records = []
    saved = 0
    skipped = 0

    print("=" * 60)
    print("BM Data Generation")
    print("=" * 60)
    print(f"Images:         {os.path.abspath(IMAGES_PATH)}")
    print(f"Binary masks:   {os.path.abspath(BINARY_MASK_PATH)}")
    print(f"Segmentations:  {os.path.abspath(SEGMENTATIONS_PATH)}")
    print(f"Output images:  {os.path.abspath(OUTPUT_IMAGES_PATH)}")
    print(f"Output segs:    {os.path.abspath(OUTPUT_SEG_PATH)}")
    print(f"First-slice CSV:{os.path.abspath(FIRST_SLICE_CSV)}")
    print(f"Studies:        {len(study_ids)}")
    print(f"Target slices:  {NUM_SLICES}")
    print("=" * 60)

    for study_id in tqdm(study_ids, desc="Generating BM data"):
        image_path = os.path.join(IMAGES_PATH, f"{study_id}.nii")
        mask_path = os.path.join(BINARY_MASK_PATH, f"{study_id}.nii")
        seg_path = os.path.join(SEGMENTATIONS_PATH, f"{study_id}.nii")

        try:
            image_nii = nib.load(image_path)
            image = image_nii.get_fdata().astype(np.float32)
            binary_mask = load_binary_mask_resampled(mask_path, image.shape)

            first_slice, sequence_length = find_first_slice_of_longest_sequence(
                binary_mask
            )
            if first_slice is None:
                skipped += 1
                print(f"Skipping {study_id}: no vertebra pixels in binary mask")
                continue

            masked_image = np.where(binary_mask > 0, image, 0.0).astype(np.float32)
            cropped_image = crop_volume(masked_image, first_slice)
            affine = build_affine(TARGET_SPACING, z_offset=first_slice)

            img_out = os.path.join(OUTPUT_IMAGES_PATH, f"{study_id}.nii")
            nib.save(nib.Nifti1Image(cropped_image, affine), img_out)

            seg = nib.load(seg_path).get_fdata()
            cropped_seg = crop_volume(seg, first_slice)
            seg_out = os.path.join(OUTPUT_SEG_PATH, f"{study_id}.nii")
            nib.save(
                nib.Nifti1Image(cropped_seg.astype(np.float32), affine),
                seg_out,
            )

            available_slices = image.shape[0] - first_slice
            records.append(
                {
                    "study": study_id,
                    "first_slice": first_slice,
                    "longest_sequence_length": sequence_length,
                    "available_slices": available_slices,
                    "padded_slices": max(0, NUM_SLICES - available_slices),
                }
            )
            saved += 1
        except Exception as e:
            skipped += 1
            print(f"Failed to process {study_id}: {e}")

    if records:
        pd.DataFrame(records).to_csv(FIRST_SLICE_CSV, index=False)

    print("\n" + "=" * 60)
    print("BM data generation complete")
    print(f"Saved:   {saved}")
    print(f"Skipped: {skipped}")
    print(f"CSV:     {os.path.abspath(FIRST_SLICE_CSV)}")
    print("=" * 60)


if __name__ == "__main__":
    generate_bm_data()
