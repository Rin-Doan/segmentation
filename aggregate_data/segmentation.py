# ============================================================================
# 3D U-Net Training Script for Vertebrae Segmentation (MONAI Version)
# ============================================================================

import os
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

# MONAI imports
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data paths
DATA_PATH = '/vast/s222440401'
TRAINING_PATH = DATA_PATH + '/agg_data/images_nii'
SEGMENTATION_PATH = DATA_PATH + '/agg_data/segmentations_nii'
AUG_NUM = 3
NUM_EPOCHS = 200
TARGET_SHAPE = (256, 256, 256)
BATCH_SIZE = 4

print(f"Training images path: {TRAINING_PATH}")
print(f"Segmentation path: {SEGMENTATION_PATH}")


# ============================================================================
# Model Initialization (MONAI 3D U-Net)
# ============================================================================

# Create MONAI 3D U-Net model
print("\n" + "="*60)
print("Initializing MONAI 3D U-Net Model")
print("="*60)

model = UNet(
    spatial_dims=3,           # 3D convolutions
    in_channels=1,            # Grayscale CT input
    out_channels=2,           # Number of segmentation classes (0: background, 1: vertebrae)
    channels=(32, 64, 128, 256, 512),  # Feature channels at each level (5 levels)
    strides=(2, 2, 2, 2),     # Downsampling stride (2x2x2 pooling)
    num_res_units=2,          # Number of residual units per level
    norm='batch',             # Batch normalization
    dropout=0.1,              # Dropout for regularization
)
model = model.to(device)

# Print model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print("="*60 + "\n")


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
    target_shape=TARGET_SHAPE,
    augment=True,
    augment_p=1.0,
    n_aug_per_study=AUG_NUM,
)

print("\nCreating validation dataset...")
val_dataset = Medical3DSegmentationDataset(
    study_ids=val_studies,
    training_path=TRAINING_PATH,
    segmentation_path=SEGMENTATION_PATH,
    target_shape=TARGET_SHAPE,
    augment=False,
)

# Create data loaders
# NOTE: 3D volumes are MUCH larger, so batch_size must be small 
batch_size = BATCH_SIZE  
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
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10)

# Dice metric for binary segmentation (foreground class only, index 1)
dice_metric = DiceMetric(include_background=False, reduction="mean")
post_pred = AsDiscrete(argmax=True, to_onehot=2)   # one-hot from argmax
post_label = AsDiscrete(to_onehot=2)               # one-hot from integer label

# Training history
train_losses = []
val_losses = []
val_dices = []
best_val_dice = 0.0

print("Training setup complete")
print(f"Loss function: DiceCELoss (Dice + CrossEntropy)")
print(f"Optimizer: Adam (lr=0.01)")
print(f"Scheduler: ReduceLROnPlateau")
print(f"Batch size: {batch_size}")


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
    """Validate for one epoch, returns (mean_loss, mean_dice)"""
    model.eval()
    total_loss = 0
    dice_metric.reset()

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Convert outputs and labels to one-hot for Dice computation
            outputs_onehot = torch.stack([post_pred(o) for o in outputs])
            labels_onehot = torch.stack([post_label(l) for l in labels])
            dice_metric(y_pred=outputs_onehot, y=labels_onehot)

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return total_loss / len(val_loader), mean_dice


print("Training functions defined\n")


# ============================================================================
# START TRAINING
# ============================================================================

num_epochs = NUM_EPOCHS  # More epochs for 3D (but with fewer samples per epoch)

print("="*60)
print(f"🚀 STARTING 3D U-NET TRAINING")
print("="*60)
print(f"Epochs: {num_epochs}")
print(f"Device: {device}")
print(f"Batch size: {batch_size}")
print(f"Input shape: (1, 256, 256, 256)")
print(f"Output classes: 2 (background + vertebrae)")
print("="*60 + "\n")

for epoch in range(num_epochs):
    print(f"\n{'='*60}")
    print(f"📊 Epoch {epoch+1}/{num_epochs}")
    print(f"{'='*60}")
    
    # Training
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    
    # Validation
    val_loss, val_dice = validate_epoch(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    val_dices.append(val_dice)

    # Learning rate scheduling (driven by loss)
    scheduler.step(val_loss)

    # Save best model based on Dice score (higher = better)
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), 'best_unet3d_model.pth')
        print(f"\n✅ New best model saved! Val Dice: {val_dice:.4f}")

    print(f"\n📈 Epoch Summary:")
    print(f"  Train Loss:     {train_loss:.4f}")
    print(f"  Val Loss:       {val_loss:.4f}")
    print(f"  Val Dice:       {val_dice:.4f}")
    print(f"  Best Val Dice:  {best_val_dice:.4f}")
    print(f"  Current LR:     {optimizer.param_groups[0]['lr']:.6f}")
    

print("\n" + "="*60)
print("🎉 TRAINING COMPLETED!")
print("="*60)
print(f"🏆 Best validation Dice: {best_val_dice:.4f}")
print(f"💾 Best model saved as: best_unet3d_model.pth")
print("="*60 + "\n")


# ============================================================================
# Save Final Results
# ============================================================================

print("💾 Saving training results...")

print("\n" + "="*60)
print("✅ All results saved successfully!")
print("="*60)
