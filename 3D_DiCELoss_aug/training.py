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
from data_process import Aggregated3DSegmentationDataset
from aggregate_data import aggregate_training_data

# MONAI imports
from monai.networks.nets import UNet
from monai.losses import DiceCELoss

warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data paths
DATA_PATH = '../../../../../vast/s222440401'
TRAINING_PATH = DATA_PATH + '/training_images'
SEGMENTATION_PATH = DATA_PATH + '/segmentations_test'
CSV_PATH = 'yolo_inference_results.csv'

print(f"Training images path: {TRAINING_PATH}")
print(f"Segmentation path: {SEGMENTATION_PATH}")
print(f"CSV path: {CSV_PATH}")


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
    out_channels=9,           # Number of segmentation classes (0-8: background + C1-C7 + other)
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
print("Loading Pre-Aggregated Training Data")
print("="*60)

# Aggregate all training data using aggregate_data.py
print("Aggregating training data from aggregate_data.py...")
image_volumes, seg_volumes, study_ids, report_dict = aggregate_training_data(
    csv_path=CSV_PATH,
    target_spacing=(1.0, 1.0, 1.0),
    verbose=True,
    n_aug_per_study=2,   # add 1 offline augmented copy per study to increase samples
)

print(f"\n✓ Loaded {len(image_volumes)} pre-processed volumes")
if report_dict.get('statistics'):
    stats = report_dict['statistics']['resampled_slices']
    print(f"  Average resampled slices: {stats['mean']:.1f}")
    print(f"  Range: {stats['min']} - {stats['max']} slices")

# Split data by study_ids for training/validation
train_indices, val_indices = train_test_split(
    range(len(study_ids)), 
    test_size=0.2, 
    random_state=42
)

train_image_volumes = [image_volumes[i] for i in train_indices]
train_seg_volumes = [seg_volumes[i] for i in train_indices]
train_study_ids = [study_ids[i] for i in train_indices]

val_image_volumes = [image_volumes[i] for i in val_indices]
val_seg_volumes = [seg_volumes[i] for i in val_indices]
val_study_ids = [study_ids[i] for i in val_indices]

print(f"\nTraining samples: {len(train_image_volumes)}")
print(f"Validation samples: {len(val_image_volumes)}")

# Create datasets
print("\n" + "="*60)
print("Creating 3D Datasets from Pre-Aggregated Data")
print("="*60)

# For 3D, we use:
# - Lower resolution (64x256x256) to fit in memory
# - Smaller batch size

print("\nCreating training dataset...")
train_dataset = Aggregated3DSegmentationDataset(
    image_volumes=train_image_volumes,
    seg_volumes=train_seg_volumes,
    study_ids=train_study_ids,
    target_shape=(128, 256, 256),        # (D, H, W) output shape
    augment=True,                       # Enable 3D augmentation
    augment_p=0.5                       # 50% augmentation probability
)

print("\nCreating validation dataset...")
val_dataset = Aggregated3DSegmentationDataset(
    image_volumes=val_image_volumes,
    seg_volumes=val_seg_volumes,
    study_ids=val_study_ids,
    target_shape=(128, 256, 256),        # Same shape
    augment=False                       # No augmentation for validation
)

# Create data loaders
# NOTE: 3D volumes are MUCH larger, so batch_size must be small 
batch_size = 4  
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

# Training history
train_losses = []
val_losses = []
best_val_loss = float('inf')

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

num_epochs = 500  # More epochs for 3D (but with fewer samples per epoch)

print("="*60)
print(f"🚀 STARTING 3D U-NET TRAINING")
print("="*60)
print(f"Epochs: {num_epochs}")
print(f"Device: {device}")
print(f"Batch size: {batch_size}")
print(f"Input shape: (1, 128, 256, 256)")
print(f"Output classes: 9 (background + C1-C7 + other)")
print("="*60 + "\n")

for epoch in range(num_epochs):
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
        torch.save(model.state_dict(), 'best_unet3d_model.pth')
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
print(f"💾 Best model saved as: best_unet3d_model.pth")
print("="*60 + "\n")


# ============================================================================
# Save Final Results
# ============================================================================

print("💾 Saving training results...")

print("\n" + "="*60)
print("✅ All results saved successfully!")
print("="*60)
