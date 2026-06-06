import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import pydicom
from scipy.ndimage import zoom, rotate, gaussian_filter, map_coordinates
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Data Augmentation for CT Vertebrae Segmentation
# ============================================================================
class CTVertebralAugmentation:
    """
    Data augmentation pipeline specifically for CT vertebrae segmentation
    
    Implements anatomically-plausible augmentations:
    - Horizontal flip (bilateral symmetry)
    - Small rotations (patient positioning)
    - Scaling (size variation)
    - Translation (ROI positioning)
    - Intensity variations (scanner differences)
    - Gaussian noise (robustness)
    - Elastic deformation (anatomical variation)
    - Gamma correction (contrast settings)
    
    Args:
        p: Probability of applying augmentation (default: 0.8)
    """
    
    def __init__(self, p=0.8):
        self.p = p
    
    def __call__(self, image, label):
        """
        Apply augmentation pipeline
        
        Args:
            image: 2D numpy array (H, W), normalized to [0, 1]
            label: 2D numpy array (H, W), integer class labels
        
        Returns:
            Augmented image and label
        """
        if np.random.random() > self.p:
            return image, label
        
        # 1. Horizontal Flip (50% chance)
        if np.random.random() > 0.5:
            image = np.fliplr(image)
            label = np.fliplr(label)
        
        # 2. Rotation (-15° to +15°, 70% chance)
        if np.random.random() > 0.3:
            angle = np.random.uniform(-15, 15)
            image = rotate(image, angle, reshape=False, order=1, mode='constant', cval=0)
            label = rotate(label, angle, reshape=False, order=0, mode='constant', cval=0)
        
        # 3. Scaling (90% to 110%, 60% chance)
        if np.random.random() > 0.4:
            scale = np.random.uniform(0.9, 1.1)
            image, label = self._scale_and_crop(image, label, scale)
        
        # 4. Translation (±10%, 60% chance)
        if np.random.random() > 0.4:
            h, w = image.shape
            shift_x = np.random.randint(int(-0.1*w), int(0.1*w))
            shift_y = np.random.randint(int(-0.1*h), int(0.1*h))
            image = np.roll(image, shift=(shift_y, shift_x), axis=(0, 1))
            label = np.roll(label, shift=(shift_y, shift_x), axis=(0, 1))
        
        # 5. Intensity Scaling and Shifting (70% chance)
        if np.random.random() > 0.3:
            scale = np.random.uniform(0.9, 1.1)
            shift = np.random.uniform(-0.1, 0.1)
            image = image * scale + shift
            image = np.clip(image, 0, 1)
        
        # 6. Gaussian Noise (50% chance)
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.02, image.shape)
            image = image + noise
            image = np.clip(image, 0, 1)
        
        # 7. Gamma Correction (30% chance)
        if np.random.random() > 0.7:
            gamma = np.random.uniform(0.8, 1.2)
            image = np.power(image, gamma)
        
        # 8. Elastic Deformation (20% chance, computationally expensive)
        if np.random.random() > 0.8:
            image, label = self._elastic_transform(image, label)
        
        return image, label
    
    def _scale_and_crop(self, image, label, scale):
        """Apply scaling and crop/pad back to original size"""
        h, w = image.shape
        image_scaled = zoom(image, scale, order=1)
        label_scaled = zoom(label, scale, order=0)
        
        new_h, new_w = image_scaled.shape
        
        # Crop or pad to original size
        if new_h > h:
            start = (new_h - h) // 2
            image_scaled = image_scaled[start:start+h, :]
            label_scaled = label_scaled[start:start+h, :]
        elif new_h < h:
            pad = (h - new_h) // 2
            image_scaled = np.pad(image_scaled, ((pad, h-new_h-pad), (0, 0)), mode='constant')
            label_scaled = np.pad(label_scaled, ((pad, h-new_h-pad), (0, 0)), mode='constant')
        
        if new_w > w:
            start = (new_w - w) // 2
            image_scaled = image_scaled[:, start:start+w]
            label_scaled = label_scaled[:, start:start+w]
        elif new_w < w:
            pad = (w - new_w) // 2
            image_scaled = np.pad(image_scaled, ((0, 0), (pad, w-new_w-pad)), mode='constant')
            label_scaled = np.pad(label_scaled, ((0, 0), (pad, w-new_w-pad)), mode='constant')
        
        return image_scaled, label_scaled
    
    def _elastic_transform(self, image, label, alpha=150, sigma=12):
        """
        Apply elastic deformation
        Simulates natural anatomical variation and slight motion
        """
        random_state = np.random.RandomState(None)
        shape = image.shape
        
        # Generate random displacement fields
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        
        # Create meshgrid and apply displacement
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (y + dy).reshape(-1), (x + dx).reshape(-1)
        
        # Apply deformation
        image_deformed = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
        label_deformed = map_coordinates(label, indices, order=0, mode='reflect').reshape(shape)
        
        return image_deformed, label_deformed

# Spatial Standardization Functions
def get_spacing_from_dicom(ds):

    try:
        
        # In-plane spacing (x, y) from PixelSpacing
        if hasattr(ds, 'PixelSpacing'):
            pixel_spacing = ds.PixelSpacing
            # PixelSpacing is [row_spacing, column_spacing]
            spacing_row = float(pixel_spacing[0])
            spacing_col = float(pixel_spacing[1])
            return (spacing_row, spacing_col)
        else:
            # Fallback: no spacing info available
            return (1.0, 1.0)
    except Exception as e:
        print(f"Warning: Could not extract spacing from DICOM: {e}")
        return (1.0, 1.0)

def get_spacing_from_nifti(nii):

    try:
        # NIfTI header contains voxel dimensions
        pixdim = nii.header.get_zooms()
        # For 3D volume: (x, y, z) spacing
        # After our transpose (2, 1, 0), the order changes
        # Original: (x, y, z) -> After transpose: (z, y, x)
        # So for 2D slice: we need (y, x) spacing
        spacing_row = float(pixdim[1])  # y-spacing
        spacing_col = float(pixdim[0])  # x-spacing
        return (spacing_row, spacing_col)
    except Exception as e:
        print(f"Warning: Could not extract spacing from NIfTI: {e}")
        return (1.0, 1.0)

def resample_to_standard_spacing(image, original_spacing, target_spacing=(1.0, 1.0), order=1):
    # Calculate zoom factors to achieve target spacing
    zoom_factors = (
        original_spacing[0] / target_spacing[0],  # row dimension
        original_spacing[1] / target_spacing[1]   # col dimension
    )
    
    # Only resample if spacing is significantly different (avoid unnecessary interpolation)
    if abs(zoom_factors[0] - 1.0) < 0.01 and abs(zoom_factors[1] - 1.0) < 0.01:
        return image
    
    # Resample using scipy's zoom
    resampled = zoom(image, zoom_factors, order=order)
    
    return resampled

# Part 3: Data Loading and Preprocessing
class MedicalSegmentationDataset(Dataset):

    def __init__(self, study_ids, training_path, segmentation_path, skip_slice, 
                 target_spacing=(1.0, 1.0), augment=False, augment_p=0.8, transform=None):
        self.study_ids = study_ids
        self.training_path = training_path
        self.segmentation_path = segmentation_path
        self.transform = transform
        self.skip_slice = skip_slice
        self.target_spacing = target_spacing  # Standard physical spacing in mm
        self.augment = augment
        
        # Initialize augmentation pipeline if needed
        if self.augment:
            self.augmentor = CTVertebralAugmentation(p=augment_p)
            print(f"  Data augmentation ENABLED (p={augment_p})")
        else:
            print(f"  Data augmentation DISABLED")
        
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
            
            # Get all DICOM files sorted
            dicom_files = sorted([f for f in os.listdir(dicom_dir) if f.endswith('.dcm')])
            
            # Load NIfTI to get number of slices
            nii = nib.load(seg_path)
            seg_volume = nii.get_fdata()
            seg_corrected = seg_volume[:, ::-1, ::-1].transpose(2, 1, 0)
            
            # Match each DICOM slice to segmentation slice
            max_slices = min(len(dicom_files), seg_corrected.shape[0])
            for i in range(0, max_slices, self.skip_slice):
                if i < len(dicom_files) and i < seg_corrected.shape[0]:
                    dicom_path = os.path.join(dicom_dir, dicom_files[i])
                    samples.append({
                        'dicom_path': dicom_path,
                        'seg_slice': i,
                        'study_id': study_id
                    })
        print(f"Created {len(samples)} samples from {len(self.study_ids)} studies")
        return samples

    
    def __len__(self):
        return len(self.samples)
    
    def _load_and_preprocess_slice(self, dicom_path, target_spacing):
        """Helper method to load and preprocess a single DICOM slice"""
        try:
            ds = pydicom.dcmread(dicom_path)
            image = ds.pixel_array.astype(np.float32)
            
            # Extract original spacing from DICOM
            dicom_spacing = get_spacing_from_dicom(ds)
            
            # HU truncation and scaling
            image = np.clip(image, -200, 1800)  # truncate HU
            image = (image - (-200)) / (1800 - (-200))  # scale to 0-1
            
            return image, dicom_spacing
            
        except Exception as e:
            print(f"Error loading DICOM {dicom_path}: {e}")
            # Return zero image if loading fails
            image = np.zeros((512, 512), dtype=np.float32)
            dicom_spacing = (1.0, 1.0)  # Default spacing for error case
            return image, dicom_spacing
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Use configured target spacing
        target_spacing = self.target_spacing
        
        # Get study directory and sorted DICOM paths
        study_dir = os.path.join(self.training_path, sample['study_id'])
        dicom_files = []
        for root, _, files in os.walk(study_dir):
            for file in files:
                if file.lower().endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))
        
        # Sort DICOM files by InstanceNumber
        try:
            dicom_meta = []
            for dcm_path in dicom_files:
                ds = pydicom.dcmread(dcm_path, stop_before_pixels=True, force=True)
                instance_num = getattr(ds, 'InstanceNumber', 0)
                dicom_meta.append((instance_num, dcm_path))
            dicom_meta.sort(key=lambda x: x[0])
            sorted_dicom_paths = [path for _, path in dicom_meta]
        except Exception as e:
            sorted_dicom_paths = sorted(dicom_files)
        
        # Get current slice index
        current_idx = sample['seg_slice']
        
        # Load 3 neighboring slices (previous, current, next)
        # Handle edge cases: duplicate first/last slice if at boundaries
        prev_idx = max(0, current_idx - 1)
        next_idx = min(len(sorted_dicom_paths) - 1, current_idx + 1)
        
        # Load the three slices
        prev_image, prev_spacing = self._load_and_preprocess_slice(
            sorted_dicom_paths[prev_idx], target_spacing)
        current_image, current_spacing = self._load_and_preprocess_slice(
            sorted_dicom_paths[current_idx], target_spacing)
        next_image, next_spacing = self._load_and_preprocess_slice(
            sorted_dicom_paths[next_idx], target_spacing)
        
        # Load corresponding segmentation slice (only for current slice)
        seg_path = os.path.join(self.segmentation_path, f"{sample['study_id']}.nii")
        if not os.path.exists(seg_path):
            seg_path = os.path.join(self.segmentation_path, f"{sample['study_id']}.nii.gz")
        
        nii = nib.load(seg_path)
        seg_volume = nii.get_fdata()
        seg_corrected = seg_volume[:, ::-1, ::-1].transpose(2, 1, 0)
        seg_slice = seg_corrected[sample['seg_slice']].astype(np.int64)
        
        # Remap labels: Keep 0-7 (background + C1-C7), merge 8-14 → 8 (other vertebrae)
        # This addresses class imbalance by grouping rare lower vertebrae classes
        seg_slice = np.where(seg_slice > 7, 8, seg_slice)
        
        # Extract spacing from NIfTI
        nifti_spacing = get_spacing_from_nifti(nii)
        
        # Resample all three slices to standard physical spacing using their respective spacings
        prev_image = resample_to_standard_spacing(prev_image, prev_spacing, target_spacing, order=1)
        current_image = resample_to_standard_spacing(current_image, current_spacing, target_spacing, order=1)
        next_image = resample_to_standard_spacing(next_image, next_spacing, target_spacing, order=1)
        seg_slice = resample_to_standard_spacing(seg_slice, nifti_spacing, target_spacing, order=0)
        
        # Resize to network input size (512x512 pixels)
        target_size = (512, 512)
        
        # Resize all three image slices
        for img_array in [prev_image, current_image, next_image]:
            if img_array.shape != target_size:
                zoom_factors = (target_size[0]/img_array.shape[0], target_size[1]/img_array.shape[1])
                if img_array is prev_image:
                    prev_image = zoom(img_array, zoom_factors, order=1)
                elif img_array is current_image:
                    current_image = zoom(img_array, zoom_factors, order=1)
                else:
                    next_image = zoom(img_array, zoom_factors, order=1)
            
        if seg_slice.shape != target_size:
            zoom_factors = (target_size[0]/seg_slice.shape[0], target_size[1]/seg_slice.shape[1])
            seg_slice = zoom(seg_slice, zoom_factors, order=0)  # Nearest neighbor for labels
        
        # Apply data augmentation (only during training) BEFORE stacking
        # This ensures all 3 slices receive the same geometric transformations
        if self.augment:
            # Apply the same augmentation to all three slices
            # Save random state to ensure identical transforms
            random_state = np.random.get_state()
            
            # Augment first slice
            np.random.set_state(random_state)
            prev_image_aug, seg_slice_aug = self.augmentor(prev_image, seg_slice)
            
            # Augment second slice with same random state for geometric transforms
            np.random.set_state(random_state)
            current_image_aug, _ = self.augmentor(current_image, seg_slice)
            
            # Augment third slice with same random state
            np.random.set_state(random_state)
            next_image_aug, _ = self.augmentor(next_image, seg_slice)
            
            # Use augmented versions
            prev_image = prev_image_aug
            current_image = current_image_aug
            next_image = next_image_aug
            seg_slice = seg_slice_aug
        
        # Stack the three slices as channels for 2.5D input
        image_25d = np.stack([prev_image, current_image, next_image], axis=0)  # Shape: (3, H, W)
        
        # Ensure arrays are contiguous before converting to tensors
        image_25d = np.ascontiguousarray(image_25d)
        seg_slice = np.ascontiguousarray(seg_slice)
        
        # Convert to tensors - image_25d already has channel dimension (3, H, W)
        image_tensor = torch.FloatTensor(image_25d)
        seg_tensor = torch.LongTensor(seg_slice)
        
        return image_tensor, seg_tensor
