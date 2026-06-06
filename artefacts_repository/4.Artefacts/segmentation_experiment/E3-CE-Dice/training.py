# Part 1: Import necessary libraries    
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
from segmentation_models_pytorch import Unet
from data_process import MedicalSegmentationDataset
import time
from computational_metrics import get_memory_usage
from codecarbon import EmissionsTracker
from training_report import generate_training_reports
from losses import CombinedLoss
warnings.filterwarnings('ignore')

# Part 2: Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Part 3: Data paths training parameters
DATA_PATH = '../../../../../vast/s222440401'
TRAINING_PATH = DATA_PATH + '/training_images'
SEGMENTATION_PATH = DATA_PATH + '/segmentations'
EPOCHS = 20
BATCH_SIZE = 64
SKIP_SLICE = 1
LEARNING_RATE = 0.001
NUM_CLASSES = 9
NUM_WORKERS = 8  # Reduced from 8 to prevent OOM (each worker loads data in memory)
CODECARBON_AVAILABLE = True
AUGMENT = True

# Part 4: Model
model = Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES
)
model = model.to(device)

# Part 5: Prepare Training Data
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
train_dataset = MedicalSegmentationDataset(train_studies, TRAINING_PATH, SEGMENTATION_PATH, SKIP_SLICE, augment=AUGMENT, augment_p=0.8)
val_dataset = MedicalSegmentationDataset(val_studies, TRAINING_PATH, SEGMENTATION_PATH, SKIP_SLICE, augment=False, augment_p=0.8)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

# Part 6: Training Setup

# CLASS WEIGHTS - Addressing severe class imbalance
# These weights are based on inverse frequency to give rare classes more importance
# Run analyze_class_distribution.py to get optimal weights for your data
class_weights = torch.FloatTensor([
    0.1,    # Class 0: Background (very frequent, low weight)
    5.0,    # Class 1: C1 (rare, high weight)
    5.0,    # Class 2: C2
    5.0,    # Class 3: C3
    5.0,    # Class 4: C4
    5.0,    # Class 5: C5
    5.0,    # Class 6: C6
    5.0,    # Class 7: C7
    3.0     # Class 8: Other vertebrae (moderate weight)
]).to(device)
# Training loss function, optimizer, and scheduler
criterion = CombinedLoss(
    ce_weight=0.5,          # Weight for Focal/CE term (disabled)
    dice_weight=0.5,        # Weight for Dice term (100%)
    class_weights=class_weights,
    use_focal=False,         # Use Focal Loss instead of CE
    focal_gamma=2.0         # Focusing parameter
)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training history
train_losses = []
val_losses = []
best_val_loss = float('inf')

print("Training setup complete")
print(f"Loss function: {criterion}")
print(f"Optimizer: {optimizer}")
print(f"Scheduler: {scheduler}")

# Part 7: Training Functions
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()   
        if batch_idx % 5 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    return total_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Validation")):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# Part 8: START TRAINING NOW!
print(f"🚀 STARTING TRAINING for {EPOCHS} epochs...")
print(f"Device: {device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Initialize efficiency tracking
training_start_time = time.time()
epoch_times = []
train_epoch_times = []
val_epoch_times = []
memory_usage_history = []
convergence_epoch = None
convergence_threshold = 0.001  # Consider converged if val_loss change < 0.001 for 3 epochs

# Get initial memory usage
initial_memory = get_memory_usage()
print(f"Initial RAM: {initial_memory['ram_gb']:.2f} GB")
if torch.cuda.is_available():
    print(f"Initial GPU Memory: {initial_memory['gpu_allocated_gb']:.2f} GB allocated, {initial_memory['gpu_reserved_gb']:.2f} GB reserved")

# Initialize energy/carbon tracking
emissions_tracker = None
if CODECARBON_AVAILABLE:
    try:
        emissions_tracker = EmissionsTracker(
            output_dir="./",
            output_file="emissions.csv",
            log_level="error",
            measure_power_secs=30,  # Measure power every 30 seconds
            save_to_file=True
        )
        emissions_tracker.start()
        print("🌍 Energy/carbon tracking enabled")
    except Exception as e:
        print(f"⚠️  Warning: Failed to initialize energy tracker: {e}")
        emissions_tracker = None
else:
    print("🌍 Energy/carbon tracking disabled (codecarbon not available)")

print("=" * 60)

# Training loop
for epoch in range(EPOCHS):
    epoch_start = time.time()
    print(f"\n📊 Epoch {epoch+1}/{EPOCHS}")
    print("-" * 50)
    
    # Training
    train_epoch_start = time.time()
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    train_epoch_time = time.time() - train_epoch_start
    train_epoch_times.append(train_epoch_time)
    train_losses.append(train_loss)

    # Validation
    val_epoch_start = time.time()
    val_loss = validate_epoch(model, val_loader, criterion, device)
    val_epoch_time = time.time() - val_epoch_start
    val_epoch_times.append(val_epoch_time)
    val_losses.append(val_loss)
    
    # Track epoch time
    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)
    
    # Track memory
    memory_usage = get_memory_usage()
    memory_usage_history.append(memory_usage)
    
    # Check for convergence (only after at least 3 epochs)
    if len(val_losses) >= 3:
        recent_val_losses = val_losses[-3:]  # Check last 3 epochs
        if max(recent_val_losses) - min(recent_val_losses) < convergence_threshold:
            if convergence_epoch is None:
                convergence_epoch = epoch + 1
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_unet_model.pth')
        print(f"✅ New best model saved! Val Loss: {val_loss:.4f}")
    
    print(f"📈 Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"📉 Current LR: {optimizer.param_groups[0]['lr']:.6f}")

print("\n🎉 TRAINING COMPLETED!")
print(f"🏆 Best validation loss: {best_val_loss:.4f}")
print("💾 Model saved as 'best_unet_model.pth'")

# Part 9: Generate training reports
print("\n📊 Generating training reports...")
emissions_data = generate_training_reports(
    epoch_times=epoch_times,
    memory_usage_history=memory_usage_history,
    convergence_epoch=convergence_epoch,
    emissions_tracker=emissions_tracker,
    csv_dir='training_reports'
)

print("=" * 60)

# Plot and save training curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='s')  
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time')
plt.legend()
plt.grid(True)
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
print("✓ Training curves saved as 'training_curves.png'")
plt.close()