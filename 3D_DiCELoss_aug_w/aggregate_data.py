"""
Aggregate and resample 3D medical imaging data
Processes volumes from first vertebrae slice to last slice, resamples to standard spacing
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import pydicom
from scipy.ndimage import zoom
import warnings
warnings.filterwarnings('ignore')

# Import functions and augmentation from data_process
from data_process import (
    CTVertebral3DAugmentation,
    get_spacing_from_dicom_series,
    get_spacing_from_nifti_3d,
    resample_to_standard_spacing_3d,
)

# Data paths
DATA_PATH = '../../../../../vast/s222440401'
TRAINING_PATH = DATA_PATH + '/training_images'
SEGMENTATION_PATH = DATA_PATH + '/segmentations'
CSV_PATH = 'yolo_inference_results.csv'
TARGET_SPACING = (1.0, 1.0, 1.0)  # (z, y, x) in mm

print("="*60)
print("3D Data Aggregation and Resampling")
print("="*60)
print(f"Training images path: {TRAINING_PATH}")
print(f"Segmentation path: {SEGMENTATION_PATH}")
print(f"Target spacing: {TARGET_SPACING} mm")
print("="*60 + "\n")


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
    
    return volume, spacing, sorted_dicom_paths


def load_nifti_volume(seg_path):
    """Load NIfTI segmentation volume"""
    nii = nib.load(seg_path)
    seg_volume = nii.get_fdata()
    
    # Apply coordinate correction (same as in data_process.py)
    seg_volume = seg_volume[:, ::-1, ::-1].transpose(2, 1, 0)
    seg_volume = seg_volume.astype(np.int64)
    
    # Get spacing
    spacing = get_spacing_from_nifti_3d(nii)
    
    return seg_volume, spacing


def process_study(study_id, first_slice_idx, target_spacing, return_data=False):
    """
    Process a single study: crop from first vertebrae slice to end, then resample
    
    Args:
        study_id: Study identifier
        first_slice_idx: Index of first slice with vertebrae (0-indexed)
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
        'first_slice_idx': first_slice_idx,
        'original_slices': 0,
        'cropped_slices': 0,
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
        
        # Check if first_slice_idx is valid
        if first_slice_idx >= image_volume.shape[0]:
            result['status'] = 'error'
            result['error'] = f"First slice index {first_slice_idx} >= total slices {image_volume.shape[0]}"
            return (None, None, result) if return_data else result
        
        # Crop from first vertebrae slice to end
        image_volume_cropped = image_volume[first_slice_idx:, :, :]
        result['cropped_slices'] = image_volume_cropped.shape[0]
        
        # Load segmentation volume
        seg_path = os.path.join(SEGMENTATION_PATH, f"{study_id}.nii")
        if not os.path.exists(seg_path):
            seg_path = os.path.join(SEGMENTATION_PATH, f"{study_id}.nii.gz")
        
        if not os.path.exists(seg_path):
            result['status'] = 'error'
            result['error'] = f"Segmentation file not found for {study_id}"
            return (None, None, result) if return_data else result
        
        seg_volume, nifti_spacing = load_nifti_volume(seg_path)
        result['original_spacing_nifti'] = nifti_spacing
        
        # Crop segmentation from same slice
        if first_slice_idx >= seg_volume.shape[0]:
            result['status'] = 'error'
            result['error'] = f"First slice index {first_slice_idx} >= segmentation slices {seg_volume.shape[0]}"
            return (None, None, result) if return_data else result
        
        seg_volume_cropped = seg_volume[first_slice_idx:, :, :]
        
        # Verify shapes match (at least in H, W dimensions)
        if (image_volume_cropped.shape[1] != seg_volume_cropped.shape[1] or 
            image_volume_cropped.shape[2] != seg_volume_cropped.shape[2]):
            print(f"Warning: Shape mismatch for {study_id}")
            print(f"  Image shape: {image_volume_cropped.shape}")
            print(f"  Seg shape: {seg_volume_cropped.shape}")
        
        
        # Resample image volume
        image_volume_resampled = resample_to_standard_spacing_3d(
            image_volume_cropped, dicom_spacing, target_spacing, order=1
        )
        
        # Resample segmentation volume
        seg_volume_resampled = resample_to_standard_spacing_3d(
            seg_volume_cropped, nifti_spacing, target_spacing, order=0
        )
        
        result['resampled_slices'] = image_volume_resampled.shape[0]
        result['resampled_shape'] = image_volume_resampled.shape
        result['status'] = 'success'
        
        if return_data:
            return image_volume_resampled, seg_volume_resampled, result
        else:
            return result
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        return (None, None, result) if return_data else result


def aggregate_training_data(csv_path=None, target_spacing=None, verbose=True, n_aug_per_study=0):
    """
    Aggregate and process all training data, returning ready-to-use volumes
    
    Args:
        csv_path: Path to CSV file with first slice indices (default: CSV_PATH)
        target_spacing: Target spacing for resampling (default: TARGET_SPACING)
        verbose: Whether to print progress messages
        n_aug_per_study: How many augmented copies to create per successful study
    
    Returns:
        Tuple of (image_volumes, seg_volumes, study_ids, report_dict)
        - image_volumes: List of numpy arrays (D, H, W) - resampled image volumes
        - seg_volumes: List of numpy arrays (D, H, W) - resampled segmentation volumes
        - study_ids: List of study IDs corresponding to the volumes
        - report_dict: Dictionary with processing statistics
    """
    if csv_path is None:
        csv_path = CSV_PATH
    if target_spacing is None:
        target_spacing = TARGET_SPACING
    
    # Load CSV with first slice indices
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    if verbose:
        print(f"Loaded {len(df)} studies from CSV")
    
    # Find overlapping studies
    training_studies = [d for d in os.listdir(TRAINING_PATH) 
                       if os.path.isdir(os.path.join(TRAINING_PATH, d))]
    segmentation_files = [f for f in os.listdir(SEGMENTATION_PATH) 
                         if f.endswith(('.nii', '.nii.gz'))]
    segmentation_studies = [f.replace('.nii.gz', '').replace('.nii', '') 
                            for f in segmentation_files]
    overlapping_studies = sorted(list(set(training_studies).intersection(set(segmentation_studies))))
    
    if verbose:
        print(f"Found {len(overlapping_studies)} overlapping studies")
    
    # Create dictionary from CSV
    first_slice_dict = {}
    for _, row in df.iterrows():
        study_id = str(row['StudyInstanceUID'])
        first_slice_dict[study_id] = int(row['slice_number'])
    
    # Process each study
    image_volumes = []
    seg_volumes = []
    study_ids = []
    results = []

    # Offline augmentation pipeline (if requested)
    augmentor = None
    if n_aug_per_study > 0:
        augmentor = CTVertebral3DAugmentation(p=1.0)  # always apply when called
        if verbose:
            print(f"\nOffline augmentation enabled: {n_aug_per_study} augmented copies per study")
    
    for idx, study_id in enumerate(overlapping_studies, 1):
        if verbose:
            print(f"[{idx}/{len(overlapping_studies)}] Processing: {study_id}")
        
        # Get first slice index (0-indexed, so subtract 1 from CSV value)
        if study_id in first_slice_dict:
            first_slice_idx = first_slice_dict[study_id] - 1  # Convert to 0-indexed
            if first_slice_idx < 0:
                first_slice_idx = 0
        else:
            if verbose:
                print(f"  Warning: {study_id} not found in CSV, using slice 0")
            first_slice_idx = 0
        
        image_vol, seg_vol, result = process_study(
            study_id, first_slice_idx, target_spacing, return_data=True
        )
        results.append(result)
        
        if result['status'] == 'success':
            # Original (un-augmented) volume
            image_volumes.append(image_vol)
            seg_volumes.append(seg_vol)
            study_ids.append(study_id)
            if verbose:
                print(
                    f"  ✓ Success: {result['original_slices']} → "
                    f"{result['cropped_slices']} → {result['resampled_slices']} slices"
                )

            # Optional offline augmented copies
            if augmentor is not None and n_aug_per_study > 0:
                # Normalize image volume to [0, 1] for augmentation (augmentor expects normalized images)
                img_normalized = np.clip(image_vol, -200, 1800)
                img_normalized = (img_normalized - (-200)) / (1800 - (-200))  # Scale to [0, 1]
                
                for k in range(n_aug_per_study):
                    img_aug, seg_aug = augmentor(img_normalized.copy(), seg_vol.copy())
                    # Convert back to HU scale for storage (dataset will normalize again)
                    img_aug_hu = img_aug * (1800 - (-200)) + (-200)
                    image_volumes.append(img_aug_hu)
                    seg_volumes.append(seg_aug)
                    study_ids.append(f"{study_id}_aug{k+1}")
                if verbose:
                    print(f"    Added {n_aug_per_study} augmented copies for {study_id}")
        else:
            if verbose:
                print(f"  ✗ Failed: {result['error']}")
    
    # Generate report
    successful_studies = [r for r in results if r['status'] == 'success']
    failed_studies = [r for r in results if r['status'] == 'error']
    
    report_dict = {
        'total_studies': len(results),
        'successful': len(successful_studies),
        'failed': len(failed_studies),
        'results': results
    }
    
    if successful_studies:
        resampled_slices = [r['resampled_slices'] for r in successful_studies]
        cropped_slices = [r['cropped_slices'] for r in successful_studies]
        original_slices = [r['original_slices'] for r in successful_studies]
        
        report_dict['statistics'] = {
            'resampled_slices': {
                'mean': float(np.mean(resampled_slices)),
                'median': float(np.median(resampled_slices)),
                'min': int(np.min(resampled_slices)),
                'max': int(np.max(resampled_slices)),
                'std': float(np.std(resampled_slices))
            },
            'cropped_slices': {
                'mean': float(np.mean(cropped_slices)),
                'median': float(np.median(cropped_slices)),
                'min': int(np.min(cropped_slices)),
                'max': int(np.max(cropped_slices))
            },
            'original_slices': {
                'mean': float(np.mean(original_slices)),
                'median': float(np.median(original_slices)),
                'min': int(np.min(original_slices)),
                'max': int(np.max(original_slices))
            }
        }
    
    if verbose:
        n_original = len(successful_studies)
        n_augmented = len(image_volumes) - n_original
        if n_augmented > 0:
            print(f"\n✓ Successfully processed {n_original} original studies")
            print(f"  Created {n_augmented} augmented copies")
            print(f"  Total volumes: {len(image_volumes)} (original + augmented)")
        else:
            print(f"\n✓ Successfully processed {len(image_volumes)} studies")
        if successful_studies:
            stats = report_dict['statistics']['resampled_slices']
            print(f"  Average resampled slices: {stats['mean']:.1f}")
            print(f"  Range: {stats['min']} - {stats['max']} slices")
    
    return image_volumes, seg_volumes, study_ids, report_dict


def main():
    """Main function to process studies and generate reports"""
    print("="*60)
    print("3D Data Aggregation and Resampling")
    print("="*60)
    
    # Aggregate training data
    image_volumes, seg_volumes, study_ids, report_dict = aggregate_training_data()
    
    # Generate detailed report
    print("\n" + "="*60)
    print("Processing Report")
    print("="*60)
    
    print(f"\nTotal studies processed: {report_dict['total_studies']}")
    print(f"Successful: {report_dict['successful']}")
    print(f"Failed: {report_dict['failed']}")
    
    
    print("\n" + "="*60)
    print("Aggregation Complete!")
    print("="*60)
    
    return image_volumes, seg_volumes, study_ids, report_dict


if __name__ == '__main__':
    main()

