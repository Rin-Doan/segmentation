import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from scipy.ndimage import zoom, rotate
from PIL import Image
from collections import Counter
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
    Dataset for 3D volumetric segmentation from cropped PNG slices
    
    Loads 3D volumes by stacking cropped PNG images and their masks
    """
    
    def __init__(self, study_ids, cropped_dataset_dir, split='train',
                 target_spacing=(2.0, 1.0, 1.0), 
                 target_shape=(64, 256, 256),
                 augment=False, augment_p=0.5):
        """
        Args:
            study_ids: List of study IDs to load
            cropped_dataset_dir: Path to cropped_yolo_dataset directory
            split: 'train' or 'val' split from cropped dataset
            target_spacing: (z, y, x) physical spacing in mm (used for resampling)
            target_shape: (D, H, W) output volume shape in pixels
            augment: Whether to apply data augmentation
            augment_p: Probability of applying augmentation
        """
        self.study_ids = study_ids
        self.cropped_dataset_dir = cropped_dataset_dir
        self.split = split
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
        """Prepare list of valid study samples from cropped PNG dataset"""
        samples = []
        # Check both train and val directories since studies might be in either
        train_images_dir = os.path.join(self.cropped_dataset_dir, 'train', 'images')
        train_masks_dir = os.path.join(self.cropped_dataset_dir, 'train', 'masks')
        val_images_dir = os.path.join(self.cropped_dataset_dir, 'val', 'images')
        val_masks_dir = os.path.join(self.cropped_dataset_dir, 'val', 'masks')
        
        for study_id in self.study_ids:
            # Find all slices for this study - check both train and val directories
            image_files = []
            masks_dir = None
            
            # First check the assigned split directory
            images_dir = train_images_dir if self.split == 'train' else val_images_dir
            masks_dir_candidate = train_masks_dir if self.split == 'train' else val_masks_dir
            
            if os.path.exists(images_dir):
                for filename in os.listdir(images_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        if filename.startswith(f"{study_id}_slice_"):
                            image_files.append(os.path.join(images_dir, filename))
                            masks_dir = masks_dir_candidate
            
            # If not found in assigned split, check the other split
            if not image_files:
                other_images_dir = val_images_dir if self.split == 'train' else train_images_dir
                other_masks_dir = val_masks_dir if self.split == 'train' else train_masks_dir
                
                if os.path.exists(other_images_dir):
                    for filename in os.listdir(other_images_dir):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            if filename.startswith(f"{study_id}_slice_"):
                                image_files.append(os.path.join(other_images_dir, filename))
                                masks_dir = other_masks_dir
            
            if not image_files:
                print(f"Warning: No cropped images found for {study_id} in either train or val split")
                continue
            
            # Sort by slice index
            image_files.sort(key=lambda x: int(x.rsplit('_slice_', 1)[1].rsplit('.', 1)[0]))
            
            samples.append({
                'study_id': study_id,
                'image_files': image_files,
                'masks_dir': masks_dir
            })
        
        print(f"Created {len(samples)} 3D volume samples from {len(self.study_ids)} studies")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _load_cropped_volume(self, image_files, masks_dir, study_id):
        """Load 3D volume from cropped PNG slices"""
        # First pass: load all slices and find target shape
        raw_slices = []
        raw_mask_slices = []
        
        for img_path in image_files:
            try:
                # Load image (grayscale PNG)
                img_pil = Image.open(img_path).convert('L')
                img_array = np.array(img_pil, dtype=np.float32)
                raw_slices.append(img_array)
                
                # Load corresponding mask
                basename = os.path.basename(img_path)
                mask_filename = basename.rsplit('.', 1)[0]  # Remove extension
                
                # Try NIfTI first, then PNG
                mask_path_nii = os.path.join(masks_dir, f"{mask_filename}.nii.gz")
                mask_path_nii_alt = os.path.join(masks_dir, f"{mask_filename}.nii")
                mask_path_png = os.path.join(masks_dir, f"{mask_filename}.png")
                
                mask_slice = None
                if os.path.exists(mask_path_nii):
                    nii = nib.load(mask_path_nii)
                    mask_slice = nii.get_fdata().astype(np.int64)
                    if mask_slice.ndim == 3:
                        mask_slice = mask_slice[:, :, 0]  # Take first slice if 3D
                elif os.path.exists(mask_path_nii_alt):
                    nii = nib.load(mask_path_nii_alt)
                    mask_slice = nii.get_fdata().astype(np.int64)
                    if mask_slice.ndim == 3:
                        mask_slice = mask_slice[:, :, 0]
                elif os.path.exists(mask_path_png):
                    mask_pil = Image.open(mask_path_png)
                    mask_slice = np.array(mask_pil, dtype=np.int64)
                
                if mask_slice is None:
                    # Create zero mask if not found
                    mask_slice = np.zeros(img_array.shape, dtype=np.int64)
                
                raw_mask_slices.append(mask_slice)
                
            except Exception as e:
                print(f"Error loading slice {img_path}: {e}")
                continue
        
        if not raw_slices:
            raise ValueError(f"No valid slices found for {study_id}")
        
        # Determine target shape (use most common shape, or max dimensions)
        shapes = [s.shape for s in raw_slices]
        # Find the most common shape
        shape_counts = Counter(shapes)
        target_shape_2d = shape_counts.most_common(1)[0][0]
        
        # Alternatively, use max dimensions to avoid cropping
        # max_h = max(s.shape[0] for s in raw_slices)
        # max_w = max(s.shape[1] for s in raw_slices)
        # target_shape_2d = (max_h, max_w)
        
        # Second pass: resize all slices to target shape
        slices = []
        mask_slices = []
        
        for img_array, mask_slice in zip(raw_slices, raw_mask_slices):
            # Resize image if needed
            if img_array.shape != target_shape_2d:
                zoom_factors = (target_shape_2d[0] / img_array.shape[0],
                              target_shape_2d[1] / img_array.shape[1])
                img_array = zoom(img_array, zoom_factors, order=1)
            
            # Resize mask if needed
            if mask_slice.shape != target_shape_2d:
                zoom_factors = (target_shape_2d[0] / mask_slice.shape[0],
                              target_shape_2d[1] / mask_slice.shape[1])
                mask_slice = zoom(mask_slice, zoom_factors, order=0)  # Nearest neighbor for labels
                mask_slice = mask_slice.astype(np.int64)
            
            slices.append(img_array)
            mask_slices.append(mask_slice)
        
        # Stack into 3D volume (D, H, W)
        image_volume = np.stack(slices, axis=0)
        seg_volume = np.stack(mask_slices, axis=0)
        
        # Apply HU normalization (same as original DICOM model)
        # PNG images were created from DICOM with:
        #   1. HU windowing: clip to [-200, 1800]
        #   2. Normalize: (image - (-200)) / (1800 - (-200)) -> [0, 1]
        #   3. Scale to PNG: image * 255 -> [0, 255]
        # So PNG values [0, 255] represent HU-windowed and normalized values
        # We reverse step 3 to get back to [0, 1] normalized HU range
        image_volume = image_volume / 255.0  # Convert PNG [0, 255] back to [0, 1]
        
        # Apply explicit HU windowing normalization (same as 3D_DiCELoss model)
        # This ensures exact same normalization as original DICOM model
        # Even though PNG already has windowing, we apply it again for consistency
        image_volume = np.clip(image_volume, 0.0, 1.0)
        # The values are already in the correct [0, 1] range representing
        # HU values [-200, 1800] after windowing and normalization
        
        # Use default spacing for cropped data (assume 2mm slice thickness)
        spacing = (2.0, 1.0, 1.0)  # (z, y, x) in mm
        
        return image_volume, seg_volume, spacing
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load from cropped dataset
        try:
            image_volume, seg_volume, spacing = self._load_cropped_volume(
                sample['image_files'], 
                sample['masks_dir'], 
                sample['study_id']
            )
        except Exception as e:
            print(f"Error loading cropped volume for {sample['study_id']}: {e}")
            image_volume = np.zeros(self.target_shape, dtype=np.float32)
            seg_volume = np.zeros(self.target_shape, dtype=np.int64)
            spacing = self.target_spacing
        
        # Resample to standard physical spacing
        image_volume = resample_to_standard_spacing_3d(
            image_volume, spacing, self.target_spacing, order=1
        )
        seg_volume = resample_to_standard_spacing_3d(
            seg_volume, spacing, self.target_spacing, order=0
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
        # Shape: (1, D, H, W) for both image and label
        # Note: MONAI's DiceCELoss with to_onehot_y=True requires target to have channel dim
        image_tensor = torch.FloatTensor(image_volume).unsqueeze(0)  # Add channel dimension
        seg_tensor = torch.LongTensor(seg_volume).unsqueeze(0)  # Add channel dimension
        
        return image_tensor, seg_tensor
