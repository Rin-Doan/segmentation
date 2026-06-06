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
class Augmentation3D:
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
        # If p < 1.0, skip augmentation with probability (1-p)
        if np.random.random() > self.p:
            return image, label
        
        # 1. Horizontal Flip (left-right, 50% chance)
        if np.random.random() > 0.3:
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
        if np.random.random() > 0.4:
            noise = np.random.normal(0, 0.01, image.shape)
            image = image + noise
            image = np.clip(image, 0, 1)
        
        # 6. Gamma Correction (30% chance)
        if np.random.random() > 0.7:
            gamma = np.random.uniform(0.85, 1.15)
            image = np.power(image, gamma)
        
        return image, label

def crop_or_pad_to_size(volume, target_shape):
    """
    Crop or pad volume to target shape (simplified and robust version).

    Along depth (z / axis 0): crop from slice 0 (drop inferior slices) or pad
    below. Along y and x: center crop or center pad.
    
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
            # z (depth): crop from the first slice downward; discard the bottom
            # y, x: center crop
            if dim == 0:
                start = 0
            else:
                start = (current_shape[dim] - target_shape[dim]) // 2
            src_slices.append(slice(start, start + target_shape[dim]))
            dst_slices.append(slice(None))
        else:
            # z: pad below the volume (content stays at the top)
            # y, x: center pad
            if dim == 0:
                start = 0
            else:
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
                 n_aug_per_study=0,
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
            n_aug_per_study: How many augmented copies to create per successful study
        """
        self.study_ids = study_ids
        self.training_path = training_path
        self.segmentation_path = segmentation_path
        self.transform = transform
        self.target_spacing = target_spacing
        self.target_shape = target_shape
        self.augment = augment
        self.n_aug_per_study = n_aug_per_study
        # Initialize augmentation pipeline if needed
        if self.augment:
            self.augmentor = Augmentation3D(p=augment_p)
            print(f"  3D Data augmentation ENABLED (p={augment_p})")
        else:
            print(f"  3D Data augmentation DISABLED")
        
        # Initialize offline augmentation if needed
        self.offline_augmentor = None
        if self.n_aug_per_study > 0:
            self.offline_augmentor = Augmentation3D(p=1.0)  # Always apply when creating offline copies
            print(f"  Offline augmentation ENABLED: {n_aug_per_study} copies per study")
        
        self.samples = self._prepare_samples()
        
    def _prepare_samples(self):
        """Prepare list of valid study samples"""
        samples = []
        
        for study_id in self.study_ids:
            # Check both .nii.gz and .nii, because aggregation scripts may save either.
            img_path_nii_gz = os.path.join(self.training_path, f"{study_id}.nii.gz")
            img_path_nii = os.path.join(self.training_path, f"{study_id}.nii")
            seg_path_nii_gz = os.path.join(self.segmentation_path, f"{study_id}.nii.gz")
            seg_path_nii = os.path.join(self.segmentation_path, f"{study_id}.nii")

            img_path = img_path_nii_gz if os.path.exists(img_path_nii_gz) else img_path_nii
            seg_path = seg_path_nii_gz if os.path.exists(seg_path_nii_gz) else seg_path_nii

            # Check if NIfTI files exist
            if os.path.exists(img_path) and os.path.exists(seg_path):
                # Original (un-augmented) sample
                samples.append({
                    'study_id': study_id,
                    'img_path': img_path,
                    'seg_path': seg_path,
                    'is_augmented': False,
                })
                
                # Create offline augmented copies if requested
                if self.offline_augmentor is not None and self.n_aug_per_study > 0:
                    for k in range(self.n_aug_per_study):
                        samples.append({
                            'study_id': f"{study_id}_aug{k+1}",
                            'img_path': img_path,      # Same source file
                            'seg_path': seg_path,      # Same source file
                            'is_augmented': True,      # Mark as augmented copy
                        })

        
        print(f"Created {len(samples)} 3D volume samples from {len(self.study_ids)} studies")
        if self.n_aug_per_study > 0:
            print(f"  (Including {self.n_aug_per_study} offline augmented copies per study)")
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
        
        return volume
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load from NIfTI file
            img_nii = nib.load(sample['img_path'])
            image_volume = img_nii.get_fdata()
            # Convert from (x, y, z) to (z, y, x) format
            image_volume = image_volume[:, ::-1, ::-1].transpose(2, 1, 0)

            
            # HU windowing and normalization
            image_volume = np.clip(image_volume, -200, 1800)
            image_volume = (image_volume - (-200)) / (1800 - (-200))  # Scale to [0, 1]
            
        except Exception as e:
            print(f"Error loading image volume for {sample['study_id']}: {e}")
            # Return zero volume
            image_volume = np.zeros(self.target_shape, dtype=np.float32)
        
        try:
            # Load segmentation volume
            nii = nib.load(sample['seg_path'])
            seg_volume = nii.get_fdata()
            
            # Apply coordinate correction (same as 2D version)
            seg_volume = seg_volume[:, ::-1, ::-1].transpose(2, 1, 0)
            seg_volume = seg_volume.astype(np.int64)
            
            
        except Exception as e:
            print(f"Error loading segmentation for {sample['study_id']}: {e}")
            seg_volume = np.zeros(self.target_shape, dtype=np.int64)

        # Crop or pad to target shape
        image_volume = crop_or_pad_to_size(image_volume, self.target_shape)
        seg_volume = crop_or_pad_to_size(seg_volume, self.target_shape)
        
        # Remap labels: Keep 0-7 (background + C1-C7), map 8-14 -> 8
        # This handles class imbalance by grouping lower vertebrae
        seg_volume = np.where(seg_volume > 7, 8, seg_volume)
        
        # Apply offline augmentation if this is an augmented copy
        if sample.get('is_augmented', False) and self.offline_augmentor is not None:
            # Apply augmentation with p=1.0 (always apply for offline augmented copies)
            image_volume, seg_volume = self.offline_augmentor(image_volume, seg_volume)
        
        # Apply online data augmentation (only during training, for non-offline-augmented samples)
        elif self.augment and not sample.get('is_augmented', False):
            image_volume, seg_volume = self.augmentor(image_volume, seg_volume)
        
        # Ensure contiguous arrays
        image_volume = np.ascontiguousarray(image_volume)
        seg_volume = np.ascontiguousarray(seg_volume)
        
        # Convert to tensors
        # Shape: (1, D, H, W) for both image and label
        # Note: MONAI's DiceCELoss with to_onehot_y=True requires target to have channel dim
        image_tensor = torch.FloatTensor(image_volume).unsqueeze(0)  # Add channel dimension
        seg_tensor = torch.LongTensor(seg_volume).unsqueeze(0)  # Add channel dimension
        
        return image_tensor, seg_tensor