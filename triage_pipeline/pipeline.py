# TRIAGE PIPELINE

# Import Libraries
from find_first_slice import find_first_slice
from aggregate_data import aggregate_data
from data_process import front_crop_or_pad_to_size
import nibabel as nib
import os
import numpy as np
import torch
from monai.networks.nets import UNet
from scipy.ndimage import label as ndimage_label

FIRST_TIME_RUN = False
SAVE_DATA = True
AGG_DATA_PATH = '/vast/s222440401/triage_database/agg_data/images_nii'
BINARY_MODEL_PATH = './models/binary_model.pth'
ML_MODEL_PATH = './models/ml_model.pth'
ML_DATA_PATH = '/vast/s222440401/triage_database/ml_data/ml_predictions'
BINARY_MASK_PATH = '/vast/s222440401/triage_database/ml_data/binary_masks'
MASKED_DATA_PATH = '/vast/s222440401/triage_database/ml_data/masked_data'

NUM_PIXELS = 10000
_CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.int32)
ML_NUM_CLASSES = 9
HU_MIN, HU_MAX = -200, 1800
ML_BG_VALUE = (0 - HU_MIN) / (HU_MAX - HU_MIN)
TARGET_SHAPE = (256, 256, 256)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Pipeline Functions

def load_binary_model(model_path: str = BINARY_MODEL_PATH) -> torch.nn.Module:
    """Load trained MONAI 3D U-Net (same architecture as segmentation.py)."""
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="batch",
        dropout=0.1,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def load_ml_model(model_path: str = ML_MODEL_PATH) -> torch.nn.Module:
    """Load trained multiclass 3D U-Net (same architecture as segmentation_ml.py)."""
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=ML_NUM_CLASSES,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="batch",
        dropout=0.1,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image_volume: np.ndarray) -> np.ndarray:
    """HU window + normalise to [0, 1] + front z-crop/pad to TARGET_SHAPE."""
    image_volume = np.clip(image_volume, HU_MIN, HU_MAX)
    image_volume = (image_volume - HU_MIN) / (HU_MAX - HU_MIN)
    image_volume = front_crop_or_pad_to_size(image_volume, TARGET_SHAPE)
    return np.ascontiguousarray(image_volume, dtype=np.float32)

def run_inference(model: torch.nn.Module, image_volume: np.ndarray) -> np.ndarray:
    """Run a single forward pass and return the binary prediction (D, H, W)."""
    image_tensor = torch.from_numpy(image_volume).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        pred = torch.argmax(outputs, dim=1).squeeze(0)
    return pred.cpu().numpy().astype(np.uint8)

def filter_small_binary_components(
    binary_mask: np.ndarray, min_pixels: int = NUM_PIXELS
) -> np.ndarray:
    """Remove 26-connected components smaller than min_pixels from a binary mask."""
    labeled, _ = ndimage_label(binary_mask, structure=_CONNECTIVITY_26)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # exclude background
    small_labels = np.where(counts < min_pixels)[0]
    result = binary_mask.copy()
    for lbl in small_labels:
        result[labeled == lbl] = 0
    return result

def keep_largest_component(ml_pred: np.ndarray, num_classes: int = ML_NUM_CLASSES) -> np.ndarray:
    """For each foreground class, retain only the largest 26-connected component."""
    result = np.zeros_like(ml_pred)
    for cls in range(1, num_classes):
        mask = (ml_pred == cls)
        if not mask.any():
            continue
        labeled, num_features = ndimage_label(mask, structure=_CONNECTIVITY_26)
        if num_features == 0:
            continue
        # Find the label with the most voxels (argmax over counts; label 0 is background).
        counts = np.bincount(labeled.ravel())
        counts[0] = 0  # exclude background label
        largest_label = counts.argmax()
        result[labeled == largest_label] = cls
    return result

# PREPARING DATA FOR PIPELINE
if FIRST_TIME_RUN:
    # 1: First Slice Mapping
    find_first_slice()
    # 2: Data Aggregation
    aggregate_data()

if SAVE_DATA:
    os.makedirs(BINARY_MASK_PATH, exist_ok=True)
    os.makedirs(MASKED_DATA_PATH, exist_ok=True)
    os.makedirs(ML_DATA_PATH, exist_ok=True)

# LOADING MODELS

model = load_binary_model(BINARY_MODEL_PATH)
print(f"Loaded model ({sum(p.numel() for p in model.parameters()):,} parameters)\n")

ml_model = load_ml_model(ML_MODEL_PATH)
print(f"Loaded ML model ({sum(p.numel() for p in ml_model.parameters()):,} parameters)\n")


# Pipeline Main 

image_files = [f for f in os.listdir(AGG_DATA_PATH) if f.endswith((".nii", ".nii.gz"))]
print(f"Found {len(image_files)} image volume(s) in {AGG_DATA_PATH}\n")

for file in image_files:
    print(f"Processing {file} ...")
    study_id = file.replace(".nii.gz", "").replace(".nii", "")
    img_nii = nib.load(os.path.join(AGG_DATA_PATH, file))
    affine = img_nii.affine
    raw_volume = img_nii.get_fdata()  # (D, H, W) HU, z along axis 0    
    image_volume = preprocess_image(raw_volume)

# 3: Binary Mask Generation
    binary_mask = run_inference(model, image_volume)
    binary_mask = filter_small_binary_components(binary_mask)
    # Apply the predicted mask to the (preprocessed) image to keep only vertebrae.
    masked_volume = (image_volume * binary_mask).astype(np.float32)

# 4: Mulitlevel Segmentation

    # Multiclass vertebra-level segmentation (matches bm_segmentation_z).
    # Build the ML input the same way generate_bm_data.py did: keep vertebra
    # voxels, set background to the trained background value (0 HU -> ML_BG_VALUE).
    ml_input = np.where(binary_mask > 0, image_volume, ML_BG_VALUE).astype(np.float32)

    ml_pred = run_inference(ml_model, ml_input)

    # Re-impose the known background using the binary mask (evaluation_ml.py):
    # force every voxel outside the vertebra mask to class 0.
    ml_pred[binary_mask == 0] = 0
    # Post-process: per class, keep only the largest 26-connected component.
    ml_pred = keep_largest_component(ml_pred)
    
    nib.save(nib.Nifti1Image(ml_pred, affine), os.path.join(ML_DATA_PATH, file))
    print(f"  saved ml result   -> {os.path.join(ML_DATA_PATH, file)}")
   
    if SAVE_DATA:
        nib.save(nib.Nifti1Image(binary_mask, affine), os.path.join(BINARY_MASK_PATH, file))
        print(f"  saved binary mask -> {os.path.join(BINARY_MASK_PATH, file)}")

        nib.save(nib.Nifti1Image(masked_volume, affine), os.path.join(MASKED_DATA_PATH, file))
        print(f"  saved masked data -> {os.path.join(MASKED_DATA_PATH, file)}")


