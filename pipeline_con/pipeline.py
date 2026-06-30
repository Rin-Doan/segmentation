#import the create_aggregated_data function from the aggregate_data.py file
from aggregate.aggregate_data import create_aggregated_data
from aggregate.data_process import front_crop_or_pad_to_size
import csv
import nibabel as nib
import numpy as np
import os
import torch
from scipy.ndimage import label as ndimage_label
from PIL import Image
from monai.networks.nets import UNet
from ultralytics import YOLO
from first_slice_detection import find_first_slice, slice_to_image, longest_present_run

DATA_PATH = '/vast/s222440401/pipeline'
AGG_DATA_PATH = DATA_PATH + '/agg_data/images_nii'
BINARY_MASK_PATH = DATA_PATH + '/binary_mask'
MASKED_DATA_PATH = DATA_PATH + '/masked_data'
ML_DATA_PATH = DATA_PATH + '/ml_data'

# Records the YOLO first-slice (trim offset) per study so the evaluation can
# apply the same z-crop to the ground-truth segmentations.
FIRST_SLICE_CSV = './first_slices.csv'

BINARY_MODEL_PATH ='./models/binary_model.pth'
YOLO_MODEL_PATH ='./models/best.pt'
ML_MODEL_PATH ='./models/ml_model.pth'

# Inference settings — must match training/evaluation preprocessing.
TARGET_SHAPE = (256, 256, 256)
# HU window applied before normalising to [0, 1] (same as data_process.py).
HU_MIN, HU_MAX = -200, 1800

# Multiclass (vertebra-level) model: 0=background, 1-7=C1-C7, 8=other.
ML_NUM_CLASSES = 9
# Background intensity the ML model was trained on: masked images set background
# to 0 HU in generate_bm_data.py, which normalises to (0 - HU_MIN)/(HU_MAX - HU_MIN).
ML_BG_VALUE = (0 - HU_MIN) / (HU_MAX - HU_MIN)

# YOLO first-slice detection settings (match find_first_slice.py).
YOLO_CONF = 0.70
YOLO_IMGSZ = 512
YOLO_BATCH = 32
# Allowed consecutive sub-threshold slices inside a run (bridges confidence dips).
YOLO_MAX_GAP = 3
SAVE_DATA = True
FIRST_TIME_RUN = False
NUM_PIXELS = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Ultralytics device spec: GPU index (0) when available, else CPU.
yolo_device = 0 if torch.cuda.is_available() else "cpu"

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


_CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.int32)


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


def run_inference(model: torch.nn.Module, image_volume: np.ndarray) -> np.ndarray:
    """Run a single forward pass and return the binary prediction (D, H, W)."""
    image_tensor = torch.from_numpy(image_volume).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        pred = torch.argmax(outputs, dim=1).squeeze(0)
    return pred.cpu().numpy().astype(np.uint8)

model = load_binary_model(BINARY_MODEL_PATH)
print(f"Loaded model ({sum(p.numel() for p in model.parameters()):,} parameters)\n")

yolo_model = YOLO(YOLO_MODEL_PATH)
print(f"Loaded YOLO first-slice detector: {YOLO_MODEL_PATH}\n")

ml_model = load_ml_model(ML_MODEL_PATH)
print(f"Loaded ML model ({sum(p.numel() for p in ml_model.parameters()):,} parameters)\n")

if SAVE_DATA:
    os.makedirs(BINARY_MASK_PATH, exist_ok=True)
    os.makedirs(MASKED_DATA_PATH, exist_ok=True)
    os.makedirs(ML_DATA_PATH, exist_ok=True)

if FIRST_TIME_RUN:
    create_aggregated_data()

image_files = [f for f in os.listdir(AGG_DATA_PATH) if f.endswith((".nii", ".nii.gz"))]
print(f"Found {len(image_files)} image volume(s) in {AGG_DATA_PATH}\n")

# Effective z-trim offset per study, written to FIRST_SLICE_CSV for the evaluation.
first_slice_records = []

for file in image_files:
    print(f"Processing {file} ...")
    study_id = file.replace(".nii.gz", "").replace(".nii", "")
    img_nii = nib.load(os.path.join(AGG_DATA_PATH, file))
    affine = img_nii.affine
    raw_volume = img_nii.get_fdata()  # (D, H, W) HU, z along axis 0

    # run yolo inference: locate the first vertebra slice on the raw HU volume
    # (YOLO does its own windowing) and trim everything above it before segmenting.
    first_slice = find_first_slice(yolo_model, raw_volume)
    if first_slice >= 0:
        print(f"  YOLO first vertebra slice: {first_slice}/{raw_volume.shape[0]}")
        raw_volume = raw_volume[first_slice:]
    else:
        print("  YOLO found no vertebra slice; using full volume")

    # Effective trim applied to the volume (0 when no vertebra slice was found),
    # so the evaluation can align the ground truth identically.
    trim_offset = first_slice if first_slice >= 0 else 0
    first_slice_records.append({"study_id": study_id, "first_slice": trim_offset})

    image_volume = preprocess_image(raw_volume)

    binary_mask = run_inference(model, image_volume)
    binary_mask = filter_small_binary_components(binary_mask)

    # Apply the predicted mask to the (preprocessed) image to keep only vertebrae.
    masked_volume = (image_volume * binary_mask).astype(np.float32)

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
        

with open(FIRST_SLICE_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["study_id", "first_slice"])
    writer.writeheader()
    writer.writerows(first_slice_records)
print(f"\nSaved first-slice offsets for {len(first_slice_records)} study(ies) -> {FIRST_SLICE_CSV}")

print(f"\nDone. Processed {len(image_files)} volume(s).")