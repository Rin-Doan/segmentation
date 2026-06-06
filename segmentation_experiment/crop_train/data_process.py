import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from PIL import Image
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
            image: 2D numpy array (H, W) or 3D array (H, W, 3) for RGB, normalized to [0, 1]
            label: 2D numpy array (H, W), integer class labels
        
        Returns:
            Augmented image and label
        """
        if np.random.random() > self.p:
            return image, label
        
        # Handle RGB images (3D) vs grayscale (2D)
        is_rgb = image.ndim == 3
        if is_rgb:
            h, w, c = image.shape
        else:
            h, w = image.shape
        
        # 1. Horizontal Flip (50% chance)
        if np.random.random() > 0.5:
            image = np.fliplr(image)
            label = np.fliplr(label)
        
        # 2. Rotation (-15° to +15°, 70% chance)
        if np.random.random() > 0.3:
            angle = np.random.uniform(-15, 15)
            if is_rgb:
                # Rotate each channel separately
                rotated_channels = []
                for channel_idx in range(c):
                    rotated_channels.append(rotate(image[:, :, channel_idx], angle, reshape=False, order=1, mode='constant', cval=0))
                image = np.stack(rotated_channels, axis=-1)
            else:
                image = rotate(image, angle, reshape=False, order=1, mode='constant', cval=0)
            label = rotate(label, angle, reshape=False, order=0, mode='constant', cval=0)
        
        # 3. Scaling (90% to 110%, 60% chance)
        if np.random.random() > 0.4:
            scale = np.random.uniform(0.9, 1.1)
            image, label = self._scale_and_crop(image, label, scale)
        
        # 4. Translation (±10%, 60% chance)
        if np.random.random() > 0.4:
            shift_x = np.random.randint(int(-0.1*w), int(0.1*w))
            shift_y = np.random.randint(int(-0.1*h), int(0.1*h))
            if is_rgb:
                image = np.roll(image, shift=(shift_y, shift_x, 0), axis=(0, 1, 2))
            else:
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
        is_rgb = image.ndim == 3
        if is_rgb:
            h, w, c = image.shape
            zoom_factors = (scale, scale, 1)
        else:
            h, w = image.shape
            zoom_factors = (scale, scale)
        
        image_scaled = zoom(image, zoom_factors, order=1)
        label_scaled = zoom(label, scale, order=0)
        
        # Get spatial dimensions (handle both RGB and grayscale)
        if is_rgb:
            new_h, new_w, _ = image_scaled.shape
        else:
            new_h, new_w = image_scaled.shape
        
        # Crop or pad to original size
        if new_h > h:
            start = (new_h - h) // 2
            if is_rgb:
                image_scaled = image_scaled[start:start+h, :, :]
            else:
                image_scaled = image_scaled[start:start+h, :]
            label_scaled = label_scaled[start:start+h, :]
        elif new_h < h:
            pad = (h - new_h) // 2
            if is_rgb:
                image_scaled = np.pad(image_scaled, ((pad, h-new_h-pad), (0, 0), (0, 0)), mode='constant')
            else:
                image_scaled = np.pad(image_scaled, ((pad, h-new_h-pad), (0, 0)), mode='constant')
            label_scaled = np.pad(label_scaled, ((pad, h-new_h-pad), (0, 0)), mode='constant')
        
        if new_w > w:
            start = (new_w - w) // 2
            if is_rgb:
                image_scaled = image_scaled[:, start:start+w, :]
            else:
                image_scaled = image_scaled[:, start:start+w]
            label_scaled = label_scaled[:, start:start+w]
        elif new_w < w:
            pad = (w - new_w) // 2
            if is_rgb:
                image_scaled = np.pad(image_scaled, ((0, 0), (pad, w-new_w-pad), (0, 0)), mode='constant')
            else:
                image_scaled = np.pad(image_scaled, ((0, 0), (pad, w-new_w-pad)), mode='constant')
            label_scaled = np.pad(label_scaled, ((0, 0), (pad, w-new_w-pad)), mode='constant')
        
        return image_scaled, label_scaled
    
    def _elastic_transform(self, image, label, alpha=150, sigma=12):
        """
        Apply elastic deformation
        Simulates natural anatomical variation and slight motion
        """
        random_state = np.random.RandomState(None)
        is_rgb = image.ndim == 3
        
        if is_rgb:
            h, w, c = image.shape
            spatial_shape = (h, w)
        else:
            h, w = image.shape
            spatial_shape = (h, w)
        
        # Generate random displacement fields (2D for spatial dimensions)
        dx = gaussian_filter((random_state.rand(*spatial_shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*spatial_shape) * 2 - 1), sigma) * alpha
        
        # Create meshgrid and apply displacement
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = (y + dy).reshape(-1), (x + dx).reshape(-1)
        
        # Apply deformation
        if is_rgb:
            # Deform each channel separately
            deformed_channels = []
            for channel_idx in range(c):
                channel_deformed = map_coordinates(image[:, :, channel_idx], indices, order=1, mode='reflect').reshape(spatial_shape)
                deformed_channels.append(channel_deformed)
            image_deformed = np.stack(deformed_channels, axis=-1)
        else:
            image_deformed = map_coordinates(image, indices, order=1, mode='reflect').reshape(spatial_shape)
        
        label_deformed = map_coordinates(label, indices, order=0, mode='reflect').reshape(spatial_shape)
        
        return image_deformed, label_deformed

# ============================================================================
# Data Loading for Cropped YOLO Dataset
# ============================================================================
class CroppedSegmentationDataset(Dataset):
    """
    Dataset for loading cropped images and masks from cropped_yolo_dataset.
    
    The cropped dataset contains:
    - Images: PNG files in {split}/images/
    - Masks: NIfTI files in {split}/masks/
    
    Args:
        split: 'train' or 'val'
        cropped_dataset_dir: Path to cropped_yolo_dataset directory
        augment: Whether to apply data augmentation (default: False)
        augment_p: Probability of applying augmentation (default: 0.8)
        target_size: Target size for resizing (height, width). Default: (512, 512)
    """
    
    def __init__(self, split='train', cropped_dataset_dir=None, augment=False, augment_p=0.8, target_size=(512, 512)):
        self.split = split
        self.augment = augment
        self.target_size = target_size
        
        # Set default path if not provided
        if cropped_dataset_dir is None:
            DATA_PATH = "../../../../../vast/s222440401"
            cropped_dataset_dir = os.path.join(DATA_PATH, "cropped_yolo_dataset")
        
        self.images_dir = os.path.join(cropped_dataset_dir, split, "images")
        self.masks_dir = os.path.join(cropped_dataset_dir, split, "masks")
        
        if not os.path.exists(self.images_dir):
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.masks_dir):
            raise ValueError(f"Masks directory not found: {self.masks_dir}")
        
        # Initialize augmentation pipeline if needed
        if self.augment:
            self.augmentor = CTVertebralAugmentation(p=augment_p)
            print(f"  Data augmentation ENABLED (p={augment_p})")
        else:
            print(f"  Data augmentation DISABLED")
        
        # Get list of image files
        self.image_files = sorted([f for f in os.listdir(self.images_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"Created {len(self.image_files)} samples from {split} set")
    
    def __len__(self):
        return len(self.image_files)
    
    def _load_image(self, image_path):
        """Load cropped image from PNG file (RGB - multi-window HU transformation)"""
        img = Image.open(image_path).convert("RGB")
        image = np.array(img).astype(np.float32)
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        return image
    
    def _load_mask(self, mask_path):
        """Load cropped mask from NIfTI file"""
        nii = nib.load(mask_path)
        mask = nii.get_fdata().astype(np.int64)
        # Handle 2D or 3D (if 3D, take first slice)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        # Ensure 2D
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
        # Remap values > 8 to class 8 (other vertebrae)
        mask = np.where(mask > 8, 8, mask)
        return mask
    
    def __getitem__(self, idx):
        # Get image file
        img_file = self.image_files[idx]
        base_name = os.path.splitext(img_file)[0]
        
        # Load image
        img_path = os.path.join(self.images_dir, img_file)
        image = self._load_image(img_path)
        
        # Load mask (try .nii.gz first, then .nii)
        mask_path = os.path.join(self.masks_dir, f"{base_name}.nii.gz")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.masks_dir, f"{base_name}.nii")
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for {img_file}: {mask_path}")
        
        mask = self._load_mask(mask_path)
        
        # Ensure image and mask have same spatial shape (image is RGB: H, W, 3)
        image_spatial_shape = image.shape[:2]
        if image_spatial_shape != mask.shape:
            # Resize mask to match image (shouldn't happen, but just in case)
            zoom_factors = (image_spatial_shape[0] / mask.shape[0], image_spatial_shape[1] / mask.shape[1])
            mask = zoom(mask, zoom_factors, order=0).astype(np.int64)
        
        # Resize to fixed size for batching (cropped images have variable sizes)
        # This ensures all images in a batch have the same dimensions
        if image_spatial_shape != self.target_size:
            zoom_factors = (self.target_size[0] / image_spatial_shape[0], self.target_size[1] / image_spatial_shape[1], 1)
            image = zoom(image, zoom_factors, order=1).astype(np.float32)  # Linear interpolation for image
            mask = zoom(mask, (self.target_size[0] / mask.shape[0], self.target_size[1] / mask.shape[1]), order=0).astype(np.int64)  # Nearest neighbor for labels
        
        # Apply data augmentation (only during training)
        if self.augment:
            image, mask = self.augmentor(image, mask)
        
        # Ensure arrays are contiguous before converting to tensors
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)
        
        # Convert to tensors
        # Image is RGB: (H, W, 3) -> (3, H, W) for PyTorch
        image_tensor = torch.FloatTensor(image).permute(2, 0, 1)  # Convert (H, W, 3) to (3, H, W)
        mask_tensor = torch.LongTensor(mask)
        
        return image_tensor, mask_tensor

