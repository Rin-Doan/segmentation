"""
Run the trained nnU-Net 3D segmentation model over every image volume in
``agg_data_1/images_nii`` and save the predicted segmentation masks as NIfTI
files in ``segmented_data``.

Preprocessing and postprocessing mirror the aggregate training/evaluation
pipeline (see ../aggregate_exp/data_process.py and ../aggregate_exp/evaluation.py)
so the predictions align with the original input volume orientation and shape.
"""

from __future__ import annotations

import os
import sys

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

# Reuse the exact model builder and preprocessing helpers used for training.
AGGREGATE_EXP_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "aggregate_exp")
)
if AGGREGATE_EXP_DIR not in sys.path:
    sys.path.insert(0, AGGREGATE_EXP_DIR)

from data_process import crop_or_pad_to_size  # noqa: E402
from segmentation_models import build_segmentation_model  # noqa: E402


# Paths and parameters aligned with aggregate_exp/evaluation.py.
DATA_PATH = "../../../../../vast/s222440401/agg_data_1"
OUTPUT_PATH = "../../../../../vast/s222440401/segmented_data"
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "best_nnunet_agg_data_1_3d_seg.pth"
)
TARGET_SHAPE: tuple[int, int, int] = (128, 256, 256)  # (D, H, W)
NUM_CLASSES = 9
ARCH = "nnunet"


def reverse_crop_or_pad(volume: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Invert ``crop_or_pad_to_size`` so the prediction has ``original_shape``.

    Regions that were cropped away during preprocessing are filled with zeros
    (the background class) since that information is not recoverable.
    """
    current_shape = np.array(volume.shape)
    target_shape = np.array(original_shape)
    output = np.zeros(target_shape, dtype=volume.dtype)

    src_slices: list[slice] = []
    dst_slices: list[slice] = []
    for dim in range(3):
        if current_shape[dim] >= target_shape[dim]:
            start = (current_shape[dim] - target_shape[dim]) // 2
            src_slices.append(slice(start, start + target_shape[dim]))
            dst_slices.append(slice(None))
        else:
            start = (target_shape[dim] - current_shape[dim]) // 2
            src_slices.append(slice(None))
            dst_slices.append(slice(start, start + current_shape[dim]))

    output[dst_slices[0], dst_slices[1], dst_slices[2]] = volume[
        src_slices[0], src_slices[1], src_slices[2]
    ]
    return output


def preprocess_image(image_volume: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Apply the same orientation + intensity preprocessing as training.

    Returns the padded/cropped tensor-ready volume and the internal (D, H, W)
    shape prior to cropping/padding, which is needed to revert the shape.
    """
    # Match Medical3DSegmentationDataset: reorient (X, Y, Z) -> (D, H, W).
    image_volume = image_volume[:, ::-1, ::-1].transpose(2, 1, 0)
    internal_shape = image_volume.shape  # (D, H, W) before crop/pad

    # HU windowing + scale to [0, 1].
    image_volume = np.clip(image_volume, -200, 1800)
    image_volume = (image_volume - (-200)) / (1800 - (-200))

    image_volume = crop_or_pad_to_size(image_volume, TARGET_SHAPE)
    image_volume = np.ascontiguousarray(image_volume, dtype=np.float32)
    return image_volume, internal_shape


def postprocess_prediction(
    pred: np.ndarray, internal_shape: tuple[int, int, int]
) -> np.ndarray:
    """Revert crop/pad and orientation swap so the mask matches the input NIfTI."""
    pred = reverse_crop_or_pad(pred, internal_shape)
    # Inverse of ``[:, ::-1, ::-1].transpose(2, 1, 0)``.
    pred = pred.transpose(2, 1, 0)[:, ::-1, ::-1]
    return np.ascontiguousarray(pred)


def load_model(device: torch.device) -> torch.nn.Module:
    print(f"Loading {ARCH} from {MODEL_PATH}")
    model = build_segmentation_model(
        ARCH, spatial_dims=3, in_channels=1, out_channels=NUM_CLASSES
    )
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded model with {n_params:,} parameters")
    return model


def main() -> None:
    images_dir = os.path.join(DATA_PATH, "images_nii")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = load_model(device)

    image_files = sorted(
        f for f in os.listdir(images_dir) if f.endswith((".nii", ".nii.gz"))
    )
    print(f"Found {len(image_files)} image volumes in {images_dir}")
    print(f"Writing predictions to {OUTPUT_PATH}")

    with torch.no_grad():
        for fname in tqdm(image_files, desc="Segmenting volumes"):
            study_id = fname.replace(".nii.gz", "").replace(".nii", "")
            img_path = os.path.join(images_dir, fname)

            nii = nib.load(img_path)
            raw_volume = np.asarray(nii.get_fdata(), dtype=np.float32)

            preprocessed, internal_shape = preprocess_image(raw_volume)

            image_tensor = (
                torch.from_numpy(preprocessed).unsqueeze(0).unsqueeze(0).to(device)
            )
            logits = model(image_tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int16)

            pred = postprocess_prediction(pred, internal_shape)

            out_nii = nib.Nifti1Image(pred.astype(np.int16), nii.affine, nii.header)
            out_nii.set_data_dtype(np.int16)
            out_path = os.path.join(OUTPUT_PATH, f"{study_id}.nii")
            nib.save(out_nii, out_path)

    print(f"\nDone. Saved {len(image_files)} segmentation volumes to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
