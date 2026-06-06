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
from data_process import Medical3DSegmentationDataset

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
SEGMENTATION_PATH = DATA_PATH + '/segmentations'

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
# Prepare Training Data
# ============================================================================

# Find overlapping studies
training_studies = [d for d in os.listdir(TRAINING_PATH) if os.path.isdir(os.path.join(TRAINING_PATH, d))]
segmentation_files = [f for f in os.listdir(SEGMENTATION_PATH) if f.endswith(('.nii', '.nii.gz'))]
segmentation_studies = [f.replace('.nii.gz', '').replace('.nii', '') for f in segmentation_files]
overlapping_studies = sorted(list(set(training_studies).intersection(set(segmentation_studies))))

print(f"Found {len(overlapping_studies)} overlapping studies")
print(f"First 5 studies: {overlapping_studies[:5]}")

# Split studies for training/validation
train_studies, val_studies = train_test_split(overlapping_studies, test_size=0.2, random_state=42)
print(f"Training studies: {len(train_studies)}")
print(f"Validation studies: {len(val_studies)}")

# Create datasets
print("\n" + "="*60)
print("Creating 3D Datasets")
print("="*60)

# For 3D, we use:
# - Lower resolution (64x256x256) to fit in memory
# - Larger physical spacing (2mm z-axis, 1mm in-plane)
# - Smaller batch size

print("\nCreating training dataset...")
train_dataset = Medical3DSegmentationDataset(
    study_ids=train_studies,
    training_path=TRAINING_PATH,
    segmentation_path=SEGMENTATION_PATH,
    target_spacing=(2.0, 1.0, 1.0),    # (z, y, x) spacing in mm
    target_shape=(64, 256, 256),        # (D, H, W) output shape
    augment=True,                       # Enable 3D augmentation
    augment_p=0.5                       # 50% augmentation probability
)

print("\nCreating validation dataset...")
val_dataset = Medical3DSegmentationDataset(
    study_ids=val_studies,
    training_path=TRAINING_PATH,
    segmentation_path=SEGMENTATION_PATH,
    target_spacing=(2.0, 1.0, 1.0),    # Same spacing
    target_shape=(64, 256, 256),        # Same shape
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
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Training history
train_losses = []
val_losses = []
best_val_loss = float('inf')

print("Training setup complete")
print(f"Loss function: DiceCELoss (Dice + CrossEntropy)")
print(f"Optimizer: Adam (lr=5e-5)")
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

num_epochs = 300  # More epochs for 3D (but with fewer samples per epoch)

print("="*60)
print(f"🚀 STARTING 3D U-NET TRAINING")
print("="*60)
print(f"Epochs: {num_epochs}")
print(f"Device: {device}")
print(f"Batch size: {batch_size}")
print(f"Input shape: (1, 64, 256, 256)")
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
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss
        }, checkpoint_path)
        print(f"  💾 Checkpoint saved: {checkpoint_path}")

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

# Save final model (last epoch)
torch.save(model.state_dict(), 'final_unet3d_model.pth')
print("✓ Final model saved as 'final_unet3d_model.pth'")

# Save complete checkpoint
checkpoint = {
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'best_val_loss': best_val_loss,
    'model_config': {
        'model_type': 'MONAI_UNet',
        'spatial_dims': 3,
        'in_channels': 1,
        'out_channels': 9,
        'channels': (32, 64, 128, 256, 512),
        'strides': (2, 2, 2, 2),
        'num_res_units': 2,
        'norm': 'batch',
        'dropout': 0.1
    },
    'training_config': {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': 5e-5,
        'weight_decay': 1e-5,
        'loss_function': 'DiceCELoss',
        'target_spacing': (2.0, 1.0, 1.0),
        'target_shape': (64, 256, 256)
    }
}
torch.save(checkpoint, 'training_checkpoint_3d.pth')
print("✓ Full checkpoint saved as 'training_checkpoint_3d.pth'")

# Save training history
np.savez('training_history_3d.npz', 
         train_losses=np.array(train_losses),
         val_losses=np.array(val_losses),
         best_val_loss=best_val_loss)
print("✓ Training history saved as 'training_history_3d.npz'")

# Plot and save training curves
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', marker='o', linewidth=2)
plt.plot(val_losses, label='Validation Loss', marker='s', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('3D U-Net Training and Validation Loss', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(val_losses, label='Validation Loss', marker='s', linewidth=2, color='orange')
plt.axhline(y=best_val_loss, color='r', linestyle='--', label=f'Best: {best_val_loss:.4f}')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation Loss', fontsize=12)
plt.title('Validation Loss Over Time', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves_3d.png', dpi=300, bbox_inches='tight')
print("✓ Training curves saved as 'training_curves_3d.png'")
plt.close()

# Save summary text file
with open('training_summary_3d.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("MONAI 3D U-Net Training Summary\n")
    f.write("="*60 + "\n\n")
    f.write(f"Model Architecture: MONAI 3D U-Net\n")
    f.write(f"  - Framework: MONAI (Medical Open Network for AI)\n")
    f.write(f"  - Input channels: 1 (grayscale CT)\n")
    f.write(f"  - Output classes: 9 (background + C1-C7 + other vertebrae)\n")
    f.write(f"  - Feature channels: (32, 64, 128, 256, 512)\n")
    f.write(f"  - Residual units: 2 per level\n")
    f.write(f"  - Normalization: Batch normalization\n")
    f.write(f"  - Dropout: 0.1\n")
    f.write(f"  - Total parameters: {total_params:,}\n")
    f.write(f"  - Trainable parameters: {trainable_params:,}\n\n")
    f.write(f"Data Configuration:\n")
    f.write(f"  - Target spacing: (2.0, 1.0, 1.0) mm\n")
    f.write(f"  - Target shape: (64, 256, 256) voxels\n")
    f.write(f"  - Training studies: {len(train_studies)}\n")
    f.write(f"  - Validation studies: {len(val_studies)}\n\n")
    f.write(f"Training Configuration:\n")
    f.write(f"  - Epochs: {num_epochs}\n")
    f.write(f"  - Batch size: {batch_size}\n")
    f.write(f"  - Learning rate: 5e-5\n")
    f.write(f"  - Weight decay: 1e-5\n")
    f.write(f"  - Loss function: DiceCELoss (Dice + CrossEntropy)\n")
    f.write(f"  - Optimizer: Adam\n")
    f.write(f"  - Scheduler: ReduceLROnPlateau\n")
    f.write(f"  - Augmentation: Enabled (p=0.5)\n\n")
    f.write(f"Results:\n")
    f.write(f"  - Best validation loss: {best_val_loss:.6f}\n")
    f.write(f"  - Final training loss: {train_losses[-1]:.6f}\n")
    f.write(f"  - Final validation loss: {val_losses[-1]:.6f}\n")
    f.write(f"  - Final learning rate: {optimizer.param_groups[0]['lr']:.6f}\n\n")
    f.write(f"Files Saved:\n")
    f.write(f"  - best_unet3d_model.pth (best model weights)\n")
    f.write(f"  - final_unet3d_model.pth (final epoch weights)\n")
    f.write(f"  - training_checkpoint_3d.pth (full checkpoint)\n")
    f.write(f"  - training_history_3d.npz (loss history)\n")
    f.write(f"  - training_curves_3d.png (visualization)\n")
    f.write("="*60 + "\n")

print("✓ Training summary saved as 'training_summary_3d.txt'")

print("\n" + "="*60)
print("✅ All results saved successfully!")
print("="*60)
