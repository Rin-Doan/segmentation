"""
Script to generate segmented data using the best trained 3D U-Net model
Processes all studies in training_images and saves predictions to segmented_data
"""

import os
import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets import UNet
import nibabel as nib
import pydicom
from scipy.ndimage import zoom
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing functions from data_process
from data_process import (
    get_spacing_from_dicom_series,
    resample_to_standard_spacing_3d,
    crop_or_pad_to_size
)

# Configuration
MODEL_PATH = 'best_unet3d_model.pth'
TRAINING_IMAGES_PATH = '../../../../../vast/s222440401/training_images'
SEGMENTED_DATA_PATH = '../../../../../vast/s222440401/segmented_data'
NUM_CLASSES = 9  # Background + C1-C7 + other vertebrae
TARGET_SPACING = (2.0, 1.0, 1.0)  # (z, y, x) spacing in mm
TARGET_SHAPE = (64, 256, 256)  # (D, H, W) output shape - D is ignored, preserves all slices
TARGET_HW = (256, 256)  # Target height and width (slices preserved)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")


def load_model(model_path=MODEL_PATH):
    """Load trained MONAI 3D U-Net model"""
    print("Loading 3D U-Net model...")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=NUM_CLASSES,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm='batch',
        dropout=0.1,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✓ MONAI 3D U-Net loaded from {model_path}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    return model


def load_dicom_volume(study_dir):
    """Load entire DICOM series as 3D volume"""
    # Find all DICOM files
    dicom_files = []
    for root, _, files in os.walk(study_dir):
        for file in files:
            if file.lower().endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
    
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {study_dir}")
    
    # Sort by InstanceNumber
    try:
        dicom_meta = []
        for dcm_path in dicom_files:
            ds = pydicom.dcmread(dcm_path, stop_before_pixels=True, force=True)
            instance_num = getattr(ds, 'InstanceNumber', 0)
            dicom_meta.append((instance_num, dcm_path))
        dicom_meta.sort(key=lambda x: x[0])
        sorted_dicom_paths = [path for _, path in dicom_meta]
    except Exception as e:
        print(f"Warning: Could not sort DICOM files: {e}")
        sorted_dicom_paths = sorted(dicom_files)
    
    # Load all slices
    slices = []
    for dcm_path in sorted_dicom_paths:
        try:
            ds = pydicom.dcmread(dcm_path)
            slice_data = ds.pixel_array.astype(np.float32)
            slices.append(slice_data)
        except Exception as e:
            print(f"Error loading {dcm_path}: {e}")
            continue
    
    if not slices:
        raise ValueError(f"No valid DICOM slices found in {study_dir}")
    
    # Stack into 3D volume (D, H, W)
    volume = np.stack(slices, axis=0)
    
    # Get spacing information
    spacing = get_spacing_from_dicom_series(sorted_dicom_paths)
    
    return volume, spacing


def preprocess_volume(image_volume, dicom_spacing, preserve_slices=True):
    """
    Preprocess 3D volume for inference
    
    Args:
        image_volume: 3D numpy array (D, H, W)
        dicom_spacing: (z, y, x) spacing in mm (not used, kept for compatibility)
        preserve_slices: If True, preserve all slices and original dimensions
    
    Returns:
        Preprocessed volume tensor: (1, 1, D, H, W)
    """
    # HU windowing and normalization
    image_volume = np.clip(image_volume, -200, 1800)
    image_volume = (image_volume - (-200)) / (1800 - (-200))  # Scale to [0, 1]
    
    # No spatial standardization - preserve original dimensions
    # Ensure contiguous array
    image_volume = np.ascontiguousarray(image_volume)
    
    # Convert to tensor: (1, 1, D, H, W)
    image_tensor = torch.FloatTensor(image_volume).unsqueeze(0).unsqueeze(0)
    
    return image_tensor


def predict_volume(model, image_tensor, preserve_slices=True):
    """
    Run inference on preprocessed volume
    
    Args:
        model: Trained 3D U-Net model
        image_tensor: Preprocessed image tensor (1, 1, D, H, W)
        preserve_slices: If True, handle variable depth with sliding window
    
    Returns:
        Prediction array: (D, H, W) with class labels
    """
    model.eval()
    _, _, D, H, W = image_tensor.shape
    
    if preserve_slices and D != TARGET_SHAPE[0]:
        # Handle variable depth volumes
        if D > TARGET_SHAPE[0]:
            # Use sliding window approach for volumes larger than 64 slices
            predictions = []
            window_size = TARGET_SHAPE[0]
            overlap = 16  # Overlap between windows to avoid boundary artifacts
            step_size = window_size - overlap
            
            for start_idx in range(0, D, step_size):
                end_idx = min(start_idx + window_size, D)
                
                # Extract window
                window = image_tensor[:, :, start_idx:end_idx, :, :]
                
                # Pad if necessary (at the end)
                if window.shape[2] < window_size:
                    padding = torch.zeros(
                        (1, 1, window_size - window.shape[2], H, W),
                        dtype=window.dtype,
                        device=window.device
                    )
                    window = torch.cat([window, padding], dim=2)
                
                # Predict on window
                with torch.no_grad():
                    window = window.to(device)
                    outputs = model(window)  # (1, C, D, H, W)
                    preds = torch.argmax(outputs, dim=1)  # (1, D, H, W)
                    preds = preds.squeeze(0).cpu().numpy()  # (D, H, W)
                
                # Handle overlap: use center region, blend at boundaries
                actual_slices = end_idx - start_idx
                if start_idx == 0:
                    # First window: use all slices
                    predictions.append(preds[:actual_slices])
                elif end_idx >= D:
                    # Last window: use remaining slices
                    predictions.append(preds[:actual_slices])
                else:
                    # Middle window: skip overlap region, use center
                    skip = overlap // 2
                    predictions.append(preds[skip:actual_slices - skip])
            
            # Concatenate all predictions
            final_pred = np.concatenate(predictions, axis=0)
            # Ensure we have exactly D slices
            if final_pred.shape[0] > D:
                final_pred = final_pred[:D]
            elif final_pred.shape[0] < D:
                # Pad if somehow we got fewer slices
                padding = np.zeros((D - final_pred.shape[0], H, W), dtype=final_pred.dtype)
                final_pred = np.concatenate([final_pred, padding], axis=0)
            return final_pred
        else:
            # D < 64: Pad to 64, predict, then crop back
            padding_size = TARGET_SHAPE[0] - D
            padding = torch.zeros(
                (1, 1, padding_size, H, W),
                dtype=image_tensor.dtype,
                device=image_tensor.device
            )
            padded_tensor = torch.cat([image_tensor, padding], dim=2)
            
            with torch.no_grad():
                padded_tensor = padded_tensor.to(device)
                outputs = model(padded_tensor)  # (1, C, D, H, W)
                preds = torch.argmax(outputs, dim=1)  # (1, D, H, W)
                preds = preds.squeeze(0).cpu().numpy()  # (D, H, W)
            
            # Return only the original D slices
            return preds[:D]
    else:
        # Standard prediction for volumes <= 64 slices
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)  # (1, C, D, H, W)
            preds = torch.argmax(outputs, dim=1)  # (1, D, H, W)
            preds = preds.squeeze(0).cpu().numpy()  # (D, H, W)
        
        return preds


def save_segmentation_as_nifti(prediction, output_path, spacing=TARGET_SPACING):
    """
    Save prediction as NIfTI file
    
    Args:
        prediction: 3D numpy array (D, H, W) with class labels = (z, y, x)
        output_path: Path to save NIfTI file
        spacing: (z, y, x) spacing in mm
    """
    # Convert to int16 for NIfTI
    prediction = prediction.astype(np.int16)
    
    # Reverse the coordinate transformation applied during loading
    # Loading does: nifti[:, ::-1, ::-1].transpose(2, 1, 0)
    # So: (x, y, z) -> (x, y[::-1], z[::-1]) -> (z[::-1], y[::-1], x)
    # To reverse: (z[::-1], y[::-1], x) -> transpose -> (x, y[::-1], z[::-1]) -> reverse y and z -> (x, y, z)
    
    # Step 1: Transpose from (D, H, W) = (z, y, x) to (x, y, z)
    nifti_data = prediction.transpose(2, 1, 0)  # (x, y, z)
    
    # Step 2: Reverse y and z axes to match original NIfTI orientation
    nifti_data = nifti_data[:, ::-1, ::-1]  # (x, y, z) -> (x, y[::-1], z[::-1])
    
    # Create affine matrix with correct spacing
    # NIfTI spacing order is (x, y, z)
    affine = np.eye(4)
    affine[0, 0] = spacing[2]  # x spacing
    affine[1, 1] = spacing[1]  # y spacing
    affine[2, 2] = spacing[0]  # z spacing
    
    # Create NIfTI image
    nii_img = nib.Nifti1Image(nifti_data, affine)
    
    # Save
    nib.save(nii_img, output_path)


def process_study(model, study_id, study_dir, output_dir):
    """
    Process a single study: load, preprocess, predict, and save
    
    Args:
        model: Trained 3D U-Net model
        study_id: Study ID (directory name)
        study_dir: Path to study DICOM directory
        output_dir: Path to save segmented output
    """
    try:
        # Load DICOM volume
        image_volume, dicom_spacing = load_dicom_volume(study_dir)
        
        # Preprocess (preserve all slices for classification)
        image_tensor = preprocess_volume(image_volume, dicom_spacing, preserve_slices=True)
        
        # Predict (with sliding window if needed to preserve all slices)
        prediction = predict_volume(model, image_tensor, preserve_slices=True)
        
        # Save as NIfTI (use original DICOM spacing to preserve original dimensions)
        output_path = os.path.join(output_dir, f"{study_id}.nii.gz")
        save_segmentation_as_nifti(prediction, output_path, spacing=dicom_spacing)
        
        return True, None
        
    except Exception as e:
        return False, str(e)


def main():
    """Main function to process all studies"""
    print("="*80)
    print("3D U-NET SEGMENTATION DATA GENERATION")
    print("="*80)
    print()
    
    # Create output directory
    os.makedirs(SEGMENTED_DATA_PATH, exist_ok=True)
    print(f"Output directory: {SEGMENTED_DATA_PATH}")
    print()
    
    # Load model
    model = load_model(MODEL_PATH)
    
    # Find all studies in training_images
    print(f"Scanning training images directory: {TRAINING_IMAGES_PATH}")
    if not os.path.exists(TRAINING_IMAGES_PATH):
        raise ValueError(f"Training images path does not exist: {TRAINING_IMAGES_PATH}")
    
    study_dirs = [d for d in os.listdir(TRAINING_IMAGES_PATH) 
                  if os.path.isdir(os.path.join(TRAINING_IMAGES_PATH, d))]
    study_dirs = sorted(study_dirs)
    
    print(f"Found {len(study_dirs)} studies to process")
    print()
    
    # Process each study
    successful = 0
    failed = 0
    failed_studies = []
    
    print("Processing studies...")
    print("="*80)
    
    for study_id in tqdm(study_dirs, desc="Processing studies"):
        study_dir = os.path.join(TRAINING_IMAGES_PATH, study_id)
        
        success, error = process_study(model, study_id, study_dir, SEGMENTED_DATA_PATH)
        
        if success:
            successful += 1
        else:
            failed += 1
            failed_studies.append((study_id, error))
            print(f"\n⚠ Failed to process {study_id}: {error}")
    
    print()
    print("="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)
    print(f"Successfully processed: {successful}/{len(study_dirs)}")
    print(f"Failed: {failed}/{len(study_dirs)}")
    
    if failed_studies:
        print("\nFailed studies:")
        for study_id, error in failed_studies:
            print(f"  - {study_id}: {error}")
    
    print(f"\nSegmented data saved to: {SEGMENTED_DATA_PATH}")
    print("="*80)


if __name__ == "__main__":
    main()

