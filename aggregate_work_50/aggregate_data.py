"""
Aggregate and resample 3D medical imaging data
Processes full volumes and resamples to standard spacing
Return a dataset for training and validation
"""

# 1 import libraries
import os
import numpy as np
import nibabel as nib
import pydicom
from scipy.ndimage import zoom, rotate
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# 2 Data paths and parameters
DATA_PATH = '../../../../../vast/s222440401'
TRAINING_PATH = DATA_PATH + '/training_images'
SEGMENTATION_PATH = DATA_PATH + '/segmentations'
OUTPUT_PATH = DATA_PATH + '/agg_data_50'
TARGET_SPACING = (0.7, 0.5, 0.5)  # (z, y, x) in mm
FULL_DATA=False

# Spatial Standardization Functions for 3D
def get_spacing_from_dicom_series(dicom_files):
    """
    Extract 3D spacing from a series of DICOM files
    Returns (z_spacing, y_spacing, x_spacing)
    """
    try:
        # Get in-plane spacing from first slice
        ds_first = pydicom.dcmread(dicom_files[0], stop_before_pixels=True, force=True)
        if hasattr(ds_first, 'PixelSpacing'):
            pixel_spacing = ds_first.PixelSpacing
            spacing_y = float(pixel_spacing[0])  # Row spacing
            spacing_x = float(pixel_spacing[1])  # Column spacing
        else:
            spacing_y, spacing_x = 1.0, 1.0
        z_positions = []

        for file in dicom_files:
            if not file.endswith(".dcm"):
                continue
            dcm = pydicom.dcmread(file)
            z_positions.append(dcm.ImagePositionPatient[2])

        z_positions = np.array(sorted(z_positions))
        spacings = np.diff(z_positions)
        spacing_z=np.mean(np.abs(spacings))
        return (spacing_z, spacing_y, spacing_x)
    except Exception as e:
        print(f"Warning: Could not extract DICOM spacing: {e}")
        return (1.0, 1.0, 1.0)


def get_spacing_from_nifti_3d(nii):
    """
    Extract 3D spacing from NIfTI file
    Returns (z_spacing, y_spacing, x_spacing) after correction
    """
    try:
        pixdim = nii.header.get_zooms()
        # After transpose (2, 1, 0): (z, y, x) order
        spacing_z = float(pixdim[2])
        spacing_y = float(pixdim[1])
        spacing_x = float(pixdim[0])
        return (spacing_z, spacing_y, spacing_x)
    except Exception as e:
        print(f"Warning: Could not extract NIfTI spacing: {e}")
        return (1.0, 1.0, 1.0)


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
            slope = float(getattr(ds, 'RescaleSlope', 1.0))
            intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
            slice_data = slice_data * slope + intercept
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
    
    return volume, spacing, sorted_dicom_paths


def load_nifti_volume(seg_path):
    """Load NIfTI segmentation volume"""
    nii = nib.load(seg_path)
    seg_volume = nii.get_fdata()
    
    # Apply coordinate correction (same as in data_process.py)
    seg_volume = seg_volume[:, ::-1, ::-1].transpose(2, 1, 0)
    seg_volume = seg_volume.astype(np.int32)
    
    # Get spacing
    spacing = get_spacing_from_nifti_3d(nii)
    
    return seg_volume, spacing

def resample_to_standard_spacing_3d(volume, original_spacing, target_spacing=(1.0, 1.0, 1.0), order=1):
    """
    Resample 3D volume to standard physical spacing
    
    Args:
        volume: 3D numpy array (D, H, W)
        original_spacing: (z_spacing, y_spacing, x_spacing) in mm
        target_spacing: desired spacing in mm
        order: interpolation order (1=linear for images, 0=nearest for labels)
    
    Returns:
        Resampled 3D volume
    """
    zoom_factors = (
        original_spacing[0] / target_spacing[0],  # z dimension
        original_spacing[1] / target_spacing[1],  # y dimension
        original_spacing[2] / target_spacing[2]   # x dimension
    )
    
    # Only resample if spacing is significantly different
    if all(abs(z - 1.0) < 0.01 for z in zoom_factors):
        return volume
    
    resampled = zoom(volume, zoom_factors, order=order)
    return resampled

def process_study(study_id, target_spacing, return_data=False):
    """
    Process a single study: resample the full volume to target spacing

    Args:
        study_id: Study identifier
        target_spacing: Target spacing for resampling
        return_data: If True, return the processed volumes along with statistics

    Returns:
        If return_data=False: Dictionary with processing results and statistics
        If return_data=True: Tuple of (image_volume, seg_volume, result_dict)
    """
    result = {
        'study_id': study_id,
        'status': 'unknown',
        'error': None,
        'original_slices': 0,
        'resampled_slices': 0,
        'original_spacing_dicom': None,
        'original_spacing_nifti': None,
        'resampled_shape': None
    }

    try:
        # Load DICOM volume
        study_dir = os.path.join(TRAINING_PATH, study_id)
        if not os.path.exists(study_dir):
            result['status'] = 'error'
            result['error'] = f"DICOM directory not found: {study_dir}"
            return (None, None, result) if return_data else result

        image_volume, dicom_spacing, dicom_paths = load_dicom_volume(study_dir)
        result['original_slices'] = image_volume.shape[0]
        result['original_spacing_dicom'] = dicom_spacing

        # Resample image volume to target spacing
        image_volume_resampled = resample_to_standard_spacing_3d(
            image_volume, dicom_spacing, target_spacing, order=1
        )
        result['resampled_slices'] = image_volume_resampled.shape[0]
        result['resampled_shape'] = image_volume_resampled.shape

        # Load segmentation volume
        seg_path = os.path.join(SEGMENTATION_PATH, f"{study_id}.nii")
        if not os.path.exists(seg_path):
            seg_path = os.path.join(SEGMENTATION_PATH, f"{study_id}.nii.gz")

        if not os.path.exists(seg_path):
            result['status'] = 'no_segmentation'
            result['error'] = f"Segmentation file not found for {study_id}"
            return (image_volume_resampled, None, result) if return_data else result

        seg_volume, nifti_spacing = load_nifti_volume(seg_path)
        result['original_spacing_nifti'] = nifti_spacing

        # Verify shapes match (at least in H, W dimensions)
        if (image_volume.shape[1] != seg_volume.shape[1] or
                image_volume.shape[2] != seg_volume.shape[2]):
            print(f"Warning: Shape mismatch for {study_id}")
            print(f"  Image shape: {image_volume.shape}")
            print(f"  Seg shape:   {seg_volume.shape}")

        # Resample segmentation volume
        seg_volume_resampled = resample_to_standard_spacing_3d(
            seg_volume, nifti_spacing, target_spacing, order=0
        )
        result['status'] = 'success'

        if return_data:
            return image_volume_resampled, seg_volume_resampled, result
        else:
            return result

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        return (None, None, result) if return_data else result


def aggregate_training_data(target_spacing=None, verbose=True):
    """
    Aggregate and process all training data, returning ready-to-use volumes

    Args:
        target_spacing: Target spacing for resampling (default: TARGET_SPACING)
        verbose: Whether to print progress messages

    Returns:
        Tuple of (image_volumes, seg_volumes, study_ids, results_dict)
        - image_volumes: List of numpy arrays (D, H, W) - resampled image volumes
        - seg_volumes: List of numpy arrays (D, H, W) - resampled segmentation volumes
        - study_ids: List of study IDs corresponding to the volumes
        - results_dict: Dictionary mapping study_id to processing results/statistics
    """
    if target_spacing is None:
        target_spacing = TARGET_SPACING

    training_studies = [d for d in os.listdir(TRAINING_PATH) if os.path.isdir(os.path.join(TRAINING_PATH, d))]
    segmentation_files = [f for f in os.listdir(SEGMENTATION_PATH) if f.endswith(('.nii', '.nii.gz'))]
    segmentation_studies = [f.replace('.nii.gz', '').replace('.nii', '') for f in segmentation_files]
    overlapping_studies = sorted(list(set(training_studies).intersection(set(segmentation_studies))))
    if not FULL_DATA:
        training_studies = overlapping_studies

    if verbose:
        print(f"Found {len(training_studies)} training studies")

    image_volumes = []
    seg_volumes = []
    study_ids = []
    results = []
    
    for idx, study_id in enumerate(training_studies, 1):
        if verbose:
            print(f"[{idx}/{len(training_studies)}] Processing: {study_id}")

        image_vol, seg_vol, result = process_study(
            study_id, target_spacing, return_data=True
        )
        results.append(result)
        if result['status'] != 'error':
            image_volumes.append(image_vol)
            seg_volumes.append(seg_vol)
            study_ids.append(study_id)
            if verbose:
                print(
                    f"  ✓ Success: {result['original_slices']} → "
                    f"{result['resampled_slices']} slices"
                )
        else:
            if verbose:
                print(f"  ✗ Failed: {result['error']}")

    results_dict = {r['study_id']: r for r in results}
    return image_volumes, seg_volumes, study_ids, results_dict

def main():
    """Main function to process studies and save aggregated data"""
    print("="*60)
    print("3D Data Aggregation and Resampling")
    print("="*60)
    
    # Aggregate training data
    image_volumes, seg_volumes, study_ids, results_dict = aggregate_training_data()

    training_studies = [d for d in os.listdir(TRAINING_PATH) if os.path.isdir(os.path.join(TRAINING_PATH, d))]
    segmentation_files = [f for f in os.listdir(SEGMENTATION_PATH) if f.endswith(('.nii', '.nii.gz'))]
    segmentation_studies = [f.replace('.nii.gz', '').replace('.nii', '') for f in segmentation_files]
    overlapping_studies = sorted(list(set(training_studies).intersection(set(segmentation_studies))))
    

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print(f"\nSaving aggregated data to: {OUTPUT_PATH}")
    
    # Create directories for NIfTI files
    images_nii_dir = os.path.join(OUTPUT_PATH, 'images_nii')
    seg_nii_dir = os.path.join(OUTPUT_PATH, 'segmentations_nii')
    os.makedirs(images_nii_dir, exist_ok=True)
    os.makedirs(seg_nii_dir, exist_ok=True)
    print(f"Saving {len(study_ids)} volumes as .nii.gz files...")
    # Create affine matrix with target spacing (1.0, 1.0, 1.0) mm
    affine = np.eye(4)
    affine[0, 0] = TARGET_SPACING[2]  # x spacing
    affine[1, 1] = TARGET_SPACING[1]  # y spacing
    affine[2, 2] = TARGET_SPACING[0]  # z spacing
    
    for study_id, img_vol, seg_vol in tqdm(zip(study_ids, image_volumes, seg_volumes), 
                                  total=len(study_ids), 
                                  desc="Saving volumes"):
        # Apply HU windowing for training images before saving.
        img_vol_windowed = np.clip(img_vol, -200, 1800).astype(np.float32)
        img_nii = nib.Nifti1Image(img_vol_windowed, affine)
        img_nii_path = os.path.join(images_nii_dir, f"{study_id}.nii")
        nib.save(img_nii, img_nii_path)
        
        if study_id in overlapping_studies:
            # Save segmentation volume as .nii file (NIfTI format)
            seg_nii = nib.Nifti1Image(seg_vol, affine)
            seg_nii_path = os.path.join(seg_nii_dir, f"{study_id}.nii")
            nib.save(seg_nii, seg_nii_path)
    
    print("\n" + "="*60)
    print("Aggregation Complete!")
    print(f"Total volumes saved: {len(study_ids)}")
    print(f"Format: .nii (NIfTI)")
    print(f"Output directory: {OUTPUT_PATH}")
    print("="*60)
    
    return image_volumes, study_ids


if __name__ == '__main__':
    main()

