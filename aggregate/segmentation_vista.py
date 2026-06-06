# ============================================================================
# 3D U-Net Training Script for Vertebrae Segmentation (MONAI Version)
# ============================================================================

import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
import nibabel as nib
from data_process import Medical3DSegmentationDataset
from segmentation_models import ARCH_CHOICES, build_segmentation_model, count_parameters, default_checkpoint_path

# MONAI imports
from monai.losses import DiceCELoss

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="3D vertebra segmentation — compare backbones (see segmentation_models.py).")
parser.add_argument(
    "--model",
    type=str,
    default="vista3d",
    choices=ARCH_CHOICES,
    help=f"Architecture: {', '.join(ARCH_CHOICES)}. "
    "'nnunet' = MONAI DynUNet (nnU-Net-style). "
    "'vista3d' = SegResNetDS2 auto branch (VISTA3D encoder family; see module docstring).",
)
parser.add_argument(
    "--pretrained",
    type=str,
    default="",
    help="Path to pretrained checkpoint (.pth/.pt) to fine-tune from.",
)
parser.add_argument(
    "--use_monai_pretrained",
    action="store_true",
    help="Download/load official MONAI Model Zoo VISTA3D bundle weights when --pretrained is not set.",
)
parser.add_argument(
    "--bundle_dir",
    type=str,
    default="./monai_bundles",
    help="Directory to store downloaded MONAI bundles.",
)
parser.add_argument(
    "--freeze_encoder",
    action="store_true",
    help="Freeze encoder parameters at start (recommended for fine-tuning).",
)
parser.add_argument(
    "--unfreeze_epoch",
    type=int,
    default=-1,
    help="Epoch to unfreeze all layers; set < 0 to keep encoder frozen.",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=120,
    help="Number of fine-tuning epochs.",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-4,
    help="Learning rate for fine-tuning.",
)
parser.add_argument(
    "--target_depth",
    type=int,
    default=128,
    help="Target depth for cropped/padded volumes.",
)
args = parser.parse_args()
SELECTED_ARCH = args.model


def _find_checkpoint_in_dir(root_dir: str) -> str:
    """Find a likely checkpoint file inside a directory."""
    root = Path(root_dir)
    if not root.exists():
        return ""

    candidates = []
    for path in root.rglob("*"):
        if path.suffix.lower() in {".pt", ".pth"}:
            name = path.name.lower()
            score = 0
            if "model" in name:
                score += 2
            if "vista" in name:
                score += 1
            candidates.append((score, path.stat().st_size, str(path)))

    if not candidates:
        return ""

    # Prefer better-named and larger files.
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def _load_checkpoint_flexible(net: torch.nn.Module, checkpoint_path: str, map_device: torch.device) -> tuple[int, int, int]:
    """
    Load checkpoint keys that match by both name and shape.
    Returns: (matched, missing_after_load, unexpected_from_checkpoint)
    """
    state = torch.load(checkpoint_path, map_location=map_device)
    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        elif "model" in state and isinstance(state["model"], dict):
            state = state["model"]

    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format in {checkpoint_path}")

    model_state = net.state_dict()
    filtered = {}
    unexpected = 0
    for key, value in state.items():
        clean_key = key[7:] if key.startswith("module.") else key
        if clean_key in model_state and model_state[clean_key].shape == value.shape:
            filtered[clean_key] = value
        else:
            unexpected += 1

    missing, _ = net.load_state_dict(filtered, strict=False)
    return len(filtered), len(missing), unexpected


def _maybe_download_monai_vista_weights(bundle_dir: str) -> str:
    """Download MONAI VISTA3D bundle and return resolved checkpoint path."""
    from monai.bundle import download as monai_bundle_download

    os.makedirs(bundle_dir, exist_ok=True)
    monai_bundle_download(name="vista3d", bundle_dir=bundle_dir)

    bundle_path = Path(bundle_dir) / "vista3d"
    ckpt_path = _find_checkpoint_in_dir(str(bundle_path))
    if not ckpt_path:
        raise FileNotFoundError(f"No .pt/.pth checkpoint found under {bundle_path}")
    return ckpt_path

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data paths
DATA_PATH = '../../../../../vast/s222440401'
TRAINING_PATH = DATA_PATH + '/agg_data/images_nii'
SEGMENTATION_PATH = DATA_PATH + '/agg_data/segmentations_nii'

print(f"Training images path: {TRAINING_PATH}")
print(f"Segmentation path: {SEGMENTATION_PATH}")


# ============================================================================
# Model Initialization (selectable backbone)
# ============================================================================

NUM_CLASSES = 9  # background + C1-C7 + other

print("\n" + "="*60)
print(f"Initializing model: {SELECTED_ARCH}")
print("="*60)

model = build_segmentation_model(SELECTED_ARCH, spatial_dims=3, in_channels=1, out_channels=NUM_CLASSES)
model = model.to(device)

total_params, trainable_params = count_parameters(model)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print("="*60 + "\n")

CHECKPOINT_PATH = default_checkpoint_path(SELECTED_ARCH)
pretrained_path = args.pretrained.strip()
if not pretrained_path and args.use_monai_pretrained and SELECTED_ARCH == "vista3d":
    print("No --pretrained provided. Downloading official MONAI VISTA3D bundle...")
    pretrained_path = _maybe_download_monai_vista_weights(args.bundle_dir)
    print(f"Resolved MONAI pretrained checkpoint: {pretrained_path}")

if pretrained_path:
    print(f"Loading pretrained checkpoint: {pretrained_path}")
    matched, missing, unexpected = _load_checkpoint_flexible(model, pretrained_path, device)
    print("Loaded checkpoint with flexible key+shape matching.")
    print(f"Matched keys: {matched}")
    print(f"Missing model keys after load: {missing}")
    print(f"Skipped checkpoint keys (name/shape mismatch): {unexpected}")


# ============================================================================
# Prepare Training Data (Using Pre-Aggregated Data)
# ============================================================================

print("\n" + "="*60)
print("Loading Pre-Aggregated Training Data from NIfTI Files")
print("="*60)

# Load NIfTI files from saved directories
print("Loading NIfTI files from saved directories...")

# Get all image and segmentation files
image_files = [f for f in os.listdir(TRAINING_PATH) if f.endswith(('.nii', '.nii.gz'))]
seg_files = [f for f in os.listdir(SEGMENTATION_PATH) if f.endswith(('.nii', '.nii.gz'))]

# Extract study IDs (remove .nii or .nii.gz extension)
def get_study_id(filename):
    return filename.replace('.nii.gz', '').replace('.nii', '')
image_study_ids = {get_study_id(f): f for f in image_files}
seg_study_ids = {get_study_id(f): f for f in seg_files}

# Find overlapping studies (studies that have both image and segmentation)
overlapping_studies = sorted(list(set(image_study_ids.keys()).intersection(set(seg_study_ids.keys()))))

print(f"Found {len(overlapping_studies)} overlapping studies")
print(f"First 5 studies: {overlapping_studies[:5]}")
# Split studies for training/validation
train_studies, val_studies = train_test_split(overlapping_studies, test_size=0.2, random_state=42)
print(f"Training studies: {len(train_studies)}")
print(f"Validation studies: {len(val_studies)}")


# Create datasets
print("\n" + "="*60)
print("Creating 3D Datasets from Pre-Aggregated Data")
print("="*60)

# For 3D, we use:
# - Lower resolution (64x256x256) to fit in memory
# - Smaller batch size

print("\nCreating training dataset...")
train_dataset = Medical3DSegmentationDataset(
    study_ids=train_studies,
    training_path=TRAINING_PATH,
    segmentation_path=SEGMENTATION_PATH,
    target_shape=(args.target_depth, 256, 256),        # (D, H, W) output shape
    augment=True,                       # Enable 3D augmentation
    augment_p=0.5                       # 50% augmentation probability
)

print("\nCreating validation dataset...")
val_dataset = Medical3DSegmentationDataset(
    study_ids=val_studies,
    training_path=TRAINING_PATH,
    segmentation_path=SEGMENTATION_PATH,
    target_shape=(args.target_depth, 256, 256),        # Same shape
    augment=False                       # No augmentation for validation
)

# Create data loaders
# NOTE: 3D volumes are MUCH larger, so batch_size must be small 
batch_size = 1  
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"\nTraining batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print("="*60 + "\n")


# ============================================================================
# Training Setup
# ============================================================================

# Loss function: DiceCELoss combines Dice loss + CrossEntropy for better segmentation
# This is superior to plain CrossEntropyLoss for medical image segmentation
criterion = DiceCELoss(
    to_onehot_y=True,    # Convert labels to one-hot encoding
    softmax=True,        # Apply softmax to predictions
    squared_pred=False,  # Use standard Dice formula
    lambda_dice=1.0,     # Weight for Dice loss
    lambda_ce=1.0        # Weight for CrossEntropy loss
)

# Optimizer (lower learning rate for 3D)
def set_finetune_trainable_params(net, freeze_encoder=True):
    """
    Freeze most layers for stable fine-tuning, keep segmentation head/decoder trainable.
    """
    for p in net.parameters():
        p.requires_grad = True

    if not freeze_encoder:
        return

    for name, p in net.named_parameters():
        if name.startswith("backbone.encoder"):
            p.requires_grad = False


set_finetune_trainable_params(model, freeze_encoder=args.freeze_encoder)
trainable_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Fine-tune trainable params at start: {trainable_now:,}")

optimizer = optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=args.lr,
    weight_decay=1e-5,
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10)

# Training history
train_losses = []
val_losses = []
best_val_loss = float('inf')

print("Training setup complete")
print(f"Loss function: DiceCELoss (Dice + CrossEntropy)")
print(f"Optimizer: Adam (lr={args.lr})")
print(f"Scheduler: ReduceLROnPlateau")
print(f"Batch size: {batch_size}")
print(f"Fine-tuning mode: {'ON' if bool(pretrained_path) else 'OFF (training from scratch)'}")
print(f"Freeze encoder at start: {args.freeze_encoder}")
print(f"Unfreeze epoch: {args.unfreeze_epoch}")


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        if batch_idx % 5 == 0:
            print(f'  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    return total_loss / len(train_loader)


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(val_loader)


print("Training functions defined\n")


# ============================================================================
# START TRAINING
# ============================================================================

num_epochs = args.epochs

print("="*60)
print(f"🚀 STARTING 3D SEGMENTATION FINE-TUNING ({SELECTED_ARCH})")
print("="*60)
print(f"Epochs: {num_epochs}")
print(f"Device: {device}")
print(f"Batch size: {batch_size}")
print(f"Input shape: (1, {args.target_depth}, 256, 256)")
print(f"Output classes: 9 (background + C1-C7 + other)")
print("="*60 + "\n")

for epoch in range(num_epochs):
    if args.freeze_encoder and args.unfreeze_epoch >= 0 and epoch == args.unfreeze_epoch:
        print("\n🔓 Unfreezing all layers for full-network fine-tuning...")
        for p in model.parameters():
            p.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.5, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10)
        trainable_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params after unfreeze: {trainable_now:,}")

    print(f"\n{'='*60}")
    print(f"📊 Epoch {epoch+1}/{num_epochs}")
    print(f"{'='*60}")
    
    # Training
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    
    # Validation
    val_loss = validate_epoch(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"\n✅ New best model saved! Val Loss: {val_loss:.4f}")
    
    print(f"\n📈 Epoch Summary:")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")
    

print("\n" + "="*60)
print("🎉 TRAINING COMPLETED!")
print("="*60)
print(f"🏆 Best validation loss: {best_val_loss:.4f}")
print(f"💾 Best model saved as: {CHECKPOINT_PATH}")
print("="*60 + "\n")


# ============================================================================
# Save Final Results
# ============================================================================

print("💾 Saving training results...")

print("\n" + "="*60)
print("✅ All results saved successfully!")
print("="*60)
