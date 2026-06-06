"""
Test script to generate segmentation for a single study
Verifies that all original slices are preserved
"""

import os
import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets import UNet
import nibabel as nib
import pydicom
from scipy.ndimage import zoom
import sys

# Import functions from generate_segmented_data
from generate_segmented_data import (
    load_dicom_volume,
    preprocess_volume,
    predict_volume,
    save_segmentation_as_nifti,
    get_spacing_from_dicom_series
)

# Configuration
MODEL_PATH = 'best_unet3d_model.pth'
TRAINING_IMAGES_PATH = '../../../../../vast/s222440401/training_images'
SEGMENTED_DATA_PATH = '../../../../../vast/s222440401/segmented_data'
NUM_CLASSES = 9
TARGET_SPACING = (2.0, 1.0, 1.0)  # Not used now, but kept for compatibility
TARGET_SHAPE = (64, 256, 256)  # Not used now, but kept for compatibility

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
    print(f"✓ Model loaded from {model_path}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    return model


def test_single_study(study_id, model):
    """
    Test processing a single study and verify slice preservation
    
    Args:
        study_id: Study ID to process
        model: Loaded model
    """
    print("=" * 60)
    print(f"Testing Study: {study_id}")
    print("=" * 60)
    
    # Step 1: Load original DICOM volume
    print("\n1. Loading original DICOM volume...")
    study_dir = os.path.join(TRAINING_IMAGES_PATH, study_id)
    
    if not os.path.exists(study_dir):
        print(f"❌ Error: Study directory not found: {study_dir}")
        return False
    
    try:
        original_volume, dicom_spacing = load_dicom_volume(study_dir)
        original_shape = original_volume.shape
        original_slices = original_shape[0]
        
        print(f"   ✓ Original volume shape: {original_shape} (D, H, W)")
        print(f"   ✓ Original number of slices: {original_slices}")
        print(f"   ✓ Original spacing: {dicom_spacing} (z, y, x) mm")
    except Exception as e:
        print(f"   ❌ Error loading DICOM: {e}")
        return False
    
    # Step 2: Preprocess volume
    print("\n2. Preprocessing volume...")
    try:
        image_tensor = preprocess_volume(original_volume, dicom_spacing, preserve_slices=True)
        preprocessed_shape = image_tensor.shape[2:]  # Remove batch and channel dims
        preprocessed_slices = preprocessed_shape[0]
        
        print(f"   ✓ Preprocessed volume shape: {preprocessed_shape} (D, H, W)")
        print(f"   ✓ Preprocessed number of slices: {preprocessed_slices}")
        
        if preprocessed_slices != original_slices:
            print(f"   ⚠️  WARNING: Slice count changed! {original_slices} -> {preprocessed_slices}")
            return False
        else:
            print(f"   ✓ Slice count preserved: {original_slices} slices")
    except Exception as e:
        print(f"   ❌ Error preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Run prediction
    print("\n3. Running model prediction...")
    try:
        prediction = predict_volume(model, image_tensor, preserve_slices=True)
        prediction_shape = prediction.shape
        prediction_slices = prediction_shape[0]
        
        print(f"   ✓ Prediction shape: {prediction_shape} (D, H, W)")
        print(f"   ✓ Prediction number of slices: {prediction_slices}")
        
        if prediction_slices != original_slices:
            print(f"   ⚠️  WARNING: Slice count changed during prediction! {original_slices} -> {prediction_slices}")
            return False
        else:
            print(f"   ✓ Slice count preserved: {original_slices} slices")
    except Exception as e:
        print(f"   ❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Save segmentation
    print("\n4. Saving segmentation...")
    output_path = os.path.join(SEGMENTED_DATA_PATH, f"{study_id}.nii.gz")
    os.makedirs(SEGMENTED_DATA_PATH, exist_ok=True)
    
    try:
        save_segmentation_as_nifti(prediction, output_path, spacing=dicom_spacing)
        print(f"   ✓ Saved to: {output_path}")
    except Exception as e:
        print(f"   ❌ Error saving: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Verify saved file
    print("\n5. Verifying saved file...")
    try:
        nii = nib.load(output_path)
        saved_volume = nii.get_fdata()
        saved_corrected = saved_volume[:, ::-1, ::-1].transpose(2, 1, 0)
        saved_shape = saved_corrected.shape
        saved_slices = saved_shape[0]
        
        print(f"   ✓ Saved volume shape (after transformation): {saved_shape} (D, H, W)")
        print(f"   ✓ Saved number of slices: {saved_slices}")
        
        if saved_slices != original_slices:
            print(f"   ⚠️  WARNING: Slice count mismatch in saved file! {original_slices} -> {saved_slices}")
            return False
        else:
            print(f"   ✓ Slice count matches: {original_slices} slices")
    except Exception as e:
        print(f"   ❌ Error verifying saved file: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original slices:     {original_slices}")
    print(f"Preprocessed slices:  {preprocessed_slices}")
    print(f"Prediction slices:    {prediction_slices}")
    print(f"Saved slices:         {saved_slices}")
    
    if all(s == original_slices for s in [preprocessed_slices, prediction_slices, saved_slices]):
        print("\n✅ SUCCESS: All slices preserved throughout the pipeline!")
        return True
    else:
        print("\n❌ FAILED: Slice count changed somewhere in the pipeline")
        return False


def main():
    """Main function"""
    print("=" * 60)
    print("SINGLE STUDY TEST - Slice Preservation Check")
    print("=" * 60)
    print()
    
    # Load model
    try:
        model = load_model()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get a test study ID
    if len(sys.argv) > 1:
        study_id = sys.argv[1]
    else:
        # Get first available study
        if os.path.exists(TRAINING_IMAGES_PATH):
            studies = [d for d in os.listdir(TRAINING_IMAGES_PATH) 
                      if os.path.isdir(os.path.join(TRAINING_IMAGES_PATH, d))]
            if studies:
                study_id = sorted(studies)[0]
                print(f"No study ID provided, using first study: {study_id}\n")
            else:
                print("❌ No studies found in training_images directory")
                return
        else:
            print(f"❌ Training images path does not exist: {TRAINING_IMAGES_PATH}")
            return
    
    # Test the study
    success = test_single_study(study_id, model)
    
    if success:
        print("\n✅ Test passed! All slices preserved.")
    else:
        print("\n❌ Test failed! Check the warnings above.")


if __name__ == '__main__':
    main()

