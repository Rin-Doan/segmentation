import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import pydicom
from scipy.ndimage import zoom, rotate, gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 3D Data Augmentation for CT Vertebrae Segmentation
# ============================================================================
class CTVertebral3DAugmentation:
    """
    3D Data augmentation pipeline for CT vertebrae segmentation
    
    Implements anatomically-plausible 3D augmentations:
    - Horizontal flip (bilateral symmetry)
    - Small rotations (patient positioning)
    - Scaling (size variation)
    - Intensity variations (scanner differences)
    - Gaussian noise (robustness)
    - Gamma correction (contrast settings)
    
    Args:
        p: Probability of applying augmentation (default: 0.5)
    """
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, label):
        """
        Apply 3D augmentation pipeline
        
        Args:
            image: 3D numpy array (D, H, W), normalized to [0, 1]
            label: 3D numpy array (D, H, W), integer class labels
        
        Returns:
            Augmented image and label
        """
        if np.random.random() > self.p:
            return image, label
        
        # 1. Horizontal Flip (left-right, 50% chance)
        if np.random.random() > 0.5:
            image = np.flip(image, axis=2).copy()  # Flip along width
            label = np.flip(label, axis=2).copy()
        
        # 2. Axial Rotation (around z-axis, -10° to +10°, 50% chance)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-10, 10)
            # Rotate each slice in the axial plane
            image = rotate(image, angle, axes=(1, 2), reshape=False, order=1, mode='constant', cval=0)
            label = rotate(label, angle, axes=(1, 2), reshape=False, order=0, mode='constant', cval=0)
        
        # 3. Scaling (95% to 105%, 40% chance)
        if np.random.random() > 0.6:
            scale = np.random.uniform(0.95, 1.05)
            original_shape = image.shape
            image = zoom(image, scale, order=1)
            label = zoom(label, scale, order=0)
            # Crop/pad back to original shape to maintain consistency
            image = crop_or_pad_to_size(image, original_shape)
            label = crop_or_pad_to_size(label, original_shape)
        
        # 4. Intensity Scaling and Shifting (60% chance)
        if np.random.random() > 0.4:
            scale = np.random.uniform(0.9, 1.1)
            shift = np.random.uniform(-0.1, 0.1)
            image = image * scale + shift
            image = np.clip(image, 0, 1)
        
        # 5. Gaussian Noise (40% chance)
        if np.random.random() > 0.6:
            noise = np.random.normal(0, 0.01, image.shape)
            image = image + noise
            image = np.clip(image, 0, 1)
        
        # 6. Gamma Correction (30% chance)
        if np.random.random() > 0.7:
            gamma = np.random.uniform(0.85, 1.15)
            image = np.power(image, gamma)
        
        return image, label


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
        
        # Get slice spacing (z-direction)
        if hasattr(ds_first, 'SliceThickness'):
            spacing_z = float(ds_first.SliceThickness)
        elif len(dicom_files) >= 2:
            # Calculate from ImagePositionPatient
            ds_second = pydicom.dcmread(dicom_files[1], stop_before_pixels=True, force=True)
            if hasattr(ds_first, 'ImagePositionPatient') and hasattr(ds_second, 'ImagePositionPatient'):
                pos1 = np.array(ds_first.ImagePositionPatient)
                pos2 = np.array(ds_second.ImagePositionPatient)
                spacing_z = float(np.linalg.norm(pos2 - pos1))
            else:
                spacing_z = 1.0
        else:
            spacing_z = 1.0
        
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


def crop_or_pad_to_size(volume, target_shape):
    """
    Crop or pad volume to target shape (simplified and robust version)
    
    Args:
        volume: 3D numpy array (D, H, W)
        target_shape: (D, H, W) desired shape
    
    Returns:
        Volume with exact target shape
    """
    current_shape = np.array(volume.shape)
    target_shape = np.array(target_shape)
    
    # Create output array with target shape
    output = np.zeros(target_shape, dtype=volume.dtype)
    
    # Calculate crop/pad indices for each dimension
    for dim in range(3):
        if current_shape[dim] < target_shape[dim]:
            # Need padding - this dimension is too small
            pass
        elif current_shape[dim] > target_shape[dim]:
            # Need cropping - this dimension is too large
            pass
    
    # Compute slices for source (volume) and destination (output)
    src_slices = []
    dst_slices = []
    
    for dim in range(3):
        if current_shape[dim] >= target_shape[dim]:
            # Crop: take center of volume
            start = (current_shape[dim] - target_shape[dim]) // 2
            src_slices.append(slice(start, start + target_shape[dim]))
            dst_slices.append(slice(None))
        else:
            # Pad: place volume in center of output
            start = (target_shape[dim] - current_shape[dim]) // 2
            src_slices.append(slice(None))
            dst_slices.append(slice(start, start + current_shape[dim]))
    
    # Copy data
    output[dst_slices[0], dst_slices[1], dst_slices[2]] = \
        volume[src_slices[0], src_slices[1], src_slices[2]]
    
    return output


# Part 3: 3D Data Loading and Preprocessing
class Medical3DSegmentationDataset(Dataset):
    """
    Dataset for 3D volumetric segmentation
    
    Loads full 3D volumes from DICOM series and NIfTI segmentations
    """
    
    def __init__(self, study_ids, training_path, segmentation_path, 
                 target_spacing=(2.0, 1.0, 1.0), 
                 target_shape=(64, 256, 256),
                 augment=False, augment_p=0.5, 
                 transform=None):
        """
        Args:
            study_ids: List of study IDs to load
            training_path: Path to DICOM directories
            segmentation_path: Path to NIfTI segmentation files
            target_spacing: (z, y, x) physical spacing in mm
            target_shape: (D, H, W) output volume shape in pixels
            augment: Whether to apply data augmentation
            augment_p: Probability of applying augmentation
        """
        self.study_ids = study_ids
        self.training_path = training_path
        self.segmentation_path = segmentation_path
        self.transform = transform
        self.target_spacing = target_spacing
        self.target_shape = target_shape
        self.augment = augment
        
        # Initialize augmentation pipeline if needed
        if self.augment:
            self.augmentor = CTVertebral3DAugmentation(p=augment_p)
            print(f"  3D Data augmentation ENABLED (p={augment_p})")
        else:
            print(f"  3D Data augmentation DISABLED")
        
        self.samples = self._prepare_samples()
        
    def _prepare_samples(self):
        samples=[]
        
        for study_id in self.study_ids:
            # Load segmentation file (try .nii.gz first, then .nii)
            seg_path = os.path.join(self.segmentation_path, f"{study_id}.nii.gz")
            if not os.path.exists(seg_path):
                seg_path = os.path.join(self.segmentation_path, f"{study_id}.nii")
            
            if not os.path.exists(seg_path):
                continue
            
            # Get DICOM directory
            dicom_dir = os.path.join(self.training_path, study_id)
            if not os.path.isdir(dicom_dir):
                continue
            
            # Check if DICOM files actually exist in the directory
            dicom_files = []
            for root, _, files in os.walk(dicom_dir):
                for file in files:
                    if file.lower().endswith('.dcm'):
                        dicom_files.append(os.path.join(root, file))
            
            if not dicom_files:
                print(f"Warning: No DICOM files found for {study_id}, skipping")
                continue
            
            # For 3D, create one sample per study (entire volume)
            # DICOM files will be loaded in _load_dicom_volume() using os.walk()
            samples.append({
                'dicom_dir': dicom_dir,
                'seg_path': seg_path,
                'study_id': study_id
            })
        print(f"Created {len(samples)} samples from {len(self.study_ids)} studies")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _load_dicom_volume(self, study_dir):
        """Load entire DICOM series as 3D volume"""
        # Find all DICOM files
        dicom_files = []
        for root, _, files in os.walk(study_dir):
            for file in files:
                if file.lower().endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))
        
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
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load DICOM volume
            image_volume, dicom_spacing = self._load_dicom_volume(sample['dicom_dir'])
            
            # HU windowing and normalization
            image_volume = np.clip(image_volume, -200, 1800)
            image_volume = (image_volume - (-200)) / (1800 - (-200))  # Scale to [0, 1]
            
        except Exception as e:
            print(f"Error loading DICOM volume for {sample['study_id']}: {e}")
            # Return zero volume
            image_volume = np.zeros(self.target_shape, dtype=np.float32)
            dicom_spacing = self.target_spacing
        
        try:
            # Load segmentation volume
            nii = nib.load(sample['seg_path'])
            seg_volume = nii.get_fdata()
            
            # Apply coordinate correction (same as 2D version)
            seg_volume = seg_volume[:, ::-1, ::-1].transpose(2, 1, 0)
            seg_volume = seg_volume.astype(np.int64)
            
            # Get spacing
            nifti_spacing = get_spacing_from_nifti_3d(nii)
            
        except Exception as e:
            print(f"Error loading segmentation for {sample['study_id']}: {e}")
            seg_volume = np.zeros(self.target_shape, dtype=np.int64)
            nifti_spacing = self.target_spacing
        
        # Resample to standard physical spacing
        image_volume = resample_to_standard_spacing_3d(
            image_volume, dicom_spacing, self.target_spacing, order=1
        )
        seg_volume = resample_to_standard_spacing_3d(
            seg_volume, nifti_spacing, self.target_spacing, order=0
        )
        
        # Crop or pad to target shape
        image_volume = crop_or_pad_to_size(image_volume, self.target_shape)
        seg_volume = crop_or_pad_to_size(seg_volume, self.target_shape)
        
        # Remap labels: Keep 0-7 (background + C1-C7), map 8-14 -> 8
        # This handles class imbalance by grouping lower vertebrae
        seg_volume = np.where(seg_volume > 7, 8, seg_volume)
        
        # Apply data augmentation (only during training)
        if self.augment:
            image_volume, seg_volume = self.augmentor(image_volume, seg_volume)
        
        # Ensure contiguous arrays
        image_volume = np.ascontiguousarray(image_volume)
        seg_volume = np.ascontiguousarray(seg_volume)
        
        # Convert to tensors
        # Image shape: (1, D, H, W) - needs channel dimension for model input
        # Label shape: (D, H, W) - CrossEntropyLoss expects no channel dimension for targets
        # Note: If using MONAI's DiceCELoss, label would need channel dim with to_onehot_y=True
        image_tensor = torch.FloatTensor(image_volume).unsqueeze(0)  # Add channel dimension
        seg_tensor = torch.LongTensor(seg_volume)  # No channel dimension for CrossEntropyLoss
        
        return image_tensor, seg_tensor
