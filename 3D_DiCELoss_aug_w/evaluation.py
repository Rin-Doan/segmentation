"""
Comprehensive 3D U-Net Segmentation Model Evaluation Script
Evaluates 3D model performance on validation set with detailed metrics and visualizations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.networks.nets import UNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from data_process import Aggregated3DSegmentationDataset
from aggregate_data import aggregate_training_data
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import torchmetrics (F1Score = Dice coefficient for segmentation)
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex, MulticlassAccuracy

# Configuration
DATA_PATH = '../../../../../vast/s222440401'
TRAINING_PATH = DATA_PATH + '/training_images'
SEGMENTATION_PATH = DATA_PATH + '/segmentations_test'
CSV_PATH = 'yolo_inference_results.csv'
MODEL_PATH = 'best_unet3d_model.pth'
BATCH_SIZE = 1  # 3D volumes are large, use batch_size=1
NUM_CLASSES = 9  # Background + C1-C7 + other vertebrae
EVAL_DIR = "evaluation_report_3d"

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# ============================================================================
# 1. LOAD MODEL
# ============================================================================
def load_model(model_path=MODEL_PATH):
    """Load trained MONAI 3D U-Net model"""
    print("Loading 3D U-Net model...")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=NUM_CLASSES,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm='batch',
        dropout=0.1,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✓ MONAI 3D U-Net loaded from {model_path}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    return model

# ============================================================================
# 2. METRICS FUNCTIONS (Using torchmetrics - OPTIMIZED)
# ============================================================================
# Initialize metrics - PER CLASS (for detailed report)  
# Note: F1Score is mathematically equivalent to Dice coefficient
dice_per_class = MulticlassF1Score(num_classes=NUM_CLASSES, average='none').to(device)
iou_per_class = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average='none').to(device)
acc_per_class = MulticlassAccuracy(num_classes=NUM_CLASSES, average='none').to(device)

# Initialize metrics - MACRO/MICRO AVERAGE (for overall score)
dice_macro = MulticlassF1Score(num_classes=NUM_CLASSES, average='macro').to(device)
iou_macro = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average='macro').to(device)
acc_micro = MulticlassAccuracy(num_classes=NUM_CLASSES, average='micro').to(device)

def calculate_metrics(pred, target):
    """
    Calculate all metrics using torchmetrics (handles 3D volumes)
    Returns both per-class and averaged metrics
    """
    # For 3D: pred and target are (D, H, W) or (1, D, H, W)
    # Flatten to 1D for metric calculation
    if pred.dim() == 4:  # (1, D, H, W) - remove channel dimension
        pred = pred.squeeze(0)
        target = target.squeeze(0)
    
    # Flatten 3D to 1D for torchmetrics
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Per-class metrics
    dice_scores = dice_per_class(pred_flat, target_flat).cpu().numpy()
    iou_scores = iou_per_class(pred_flat, target_flat).cpu().numpy()
    class_acc = acc_per_class(pred_flat, target_flat).cpu().numpy()
    
    # Overall metrics
    dice_avg = dice_macro(pred_flat, target_flat).item()
    iou_avg = iou_macro(pred_flat, target_flat).item()
    pixel_acc = acc_micro(pred_flat, target_flat).item()
    
    # Reset metrics for next calculation
    dice_per_class.reset()
    iou_per_class.reset()
    acc_per_class.reset()
    dice_macro.reset()
    iou_macro.reset()
    acc_micro.reset()
    
    return {
        'dice_per_class': dice_scores,
        'iou_per_class': iou_scores,
        'class_accuracy': class_acc,
        'mean_dice': dice_avg,
        'mean_iou': iou_avg,
        'pixel_accuracy': pixel_acc
    }

# ============================================================================
# 3. EVALUATE ON DATASET
# ============================================================================
def evaluate_model(model, dataloader, num_classes=NUM_CLASSES):
    """Evaluate 3D model on entire dataset"""
    model.eval()
    
    all_dice_scores = []
    all_iou_scores = []
    all_pixel_acc = []
    all_class_acc = []
    
    all_preds = []
    all_targets = []
    
    print("Evaluating 3D model on validation set...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Processing 3D volumes"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Get predictions (output shape: B, C, D, H, W)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)  # (B, D, H, W)
            
            # Calculate metrics for each volume in batch
            for pred, target in zip(preds, labels):
                # pred: (D, H, W), target: (1, D, H, W)
                metrics = calculate_metrics(pred, target)
                
                all_dice_scores.append(metrics['dice_per_class'])
                all_iou_scores.append(metrics['iou_per_class'])
                all_pixel_acc.append(metrics['pixel_accuracy'])
                all_class_acc.append(metrics['class_accuracy'])
                
                # Store for confusion matrix (sample to avoid memory issues with 3D)
                if len(all_preds) < 100000:  # Limit samples for confusion matrix
                    all_preds.extend(pred.cpu().numpy().flatten()[::50])  # Sample every 50th voxel (3D is larger)
                    all_targets.extend(target.cpu().numpy().flatten()[::50])
    
    results = {
        'dice_scores': np.array(all_dice_scores),
        'iou_scores': np.array(all_iou_scores),
        'pixel_accuracy': np.array(all_pixel_acc),
        'class_accuracy': np.array(all_class_acc),
        'predictions': np.array(all_preds),
        'targets': np.array(all_targets)
    }
    
    print(f"✓ Evaluation complete on {len(all_pixel_acc)} 3D volumes\n")
    return results

# ============================================================================
# 4. VISUALIZATION FUNCTIONS
# ============================================================================
def plot_sample_predictions(model, dataset, num_samples=6, save_path='sample_predictions.png'):
    """Visualize sample predictions (middle slice from 3D volumes)"""
    print(f"Generating sample predictions from 3D volumes ({num_samples} samples)...")
    model.eval()
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            image, label = dataset[sample_idx]
            image_input = image.unsqueeze(0).to(device)  # (1, 1, D, H, W)
            
            # Get prediction
            output = model(image_input)  # (1, C, D, H, W)
            pred = torch.argmax(output, dim=1).squeeze(0)  # (D, H, W)
            
            # Calculate metrics using torchmetrics
            metrics = calculate_metrics(pred, label.to(device))
            
            # Get middle slice for visualization
            # image: (1, D, H, W), label: (1, D, H, W), pred: (D, H, W)
            mid_slice = image.shape[1] // 2
            
            img_np = image[0, mid_slice].cpu().numpy()  # Middle slice
            label_np = label.squeeze(0)[mid_slice].cpu().numpy()
            pred_np = pred[mid_slice].cpu().numpy()
            
            # Plot
            axes[idx, 0].imshow(img_np, cmap='gray')
            axes[idx, 0].set_title(f'Input (slice {mid_slice})', fontsize=12, fontweight='bold')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(label_np, cmap='tab20', vmin=0, vmax=8)
            axes[idx, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(pred_np, cmap='tab20', vmin=0, vmax=8)
            axes[idx, 2].set_title(f'Prediction\nAcc: {metrics["pixel_accuracy"]:.3f} | Dice: {metrics["mean_dice"]:.3f}', 
                                  fontsize=12, fontweight='bold')
            axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Sample predictions saved to '{save_path}'\n")
    plt.close()

def plot_metrics_per_class(dice_scores, iou_scores, class_acc, save_path='metrics_per_class.png'):
    """Plot per-class metrics"""
    print("Generating per-class metrics plot...")
    
    mean_dice = np.nanmean(dice_scores, axis=0)
    mean_iou = np.nanmean(iou_scores, axis=0)
    mean_class_acc = np.nanmean(class_acc, axis=0)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    classes = range(NUM_CLASSES)
    
    # Dice scores
    axes[0].bar(classes, mean_dice, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Dice Score', fontsize=12)
    axes[0].set_title('Mean Dice Score per Class', fontsize=14, fontweight='bold')
    axes[0].set_xticks(classes)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, 1])
    for i, v in enumerate(mean_dice):
        if not np.isnan(v):
            axes[0].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
    
    # IoU scores
    axes[1].bar(classes, mean_iou, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Class', fontsize=12)
    axes[1].set_ylabel('IoU Score', fontsize=12)
    axes[1].set_title('Mean IoU Score per Class', fontsize=14, fontweight='bold')
    axes[1].set_xticks(classes)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0, 1])
    for i, v in enumerate(mean_iou):
        if not np.isnan(v):
            axes[1].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
    
    # Class accuracy
    axes[2].bar(classes, mean_class_acc, color='seagreen', alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Class', fontsize=12)
    axes[2].set_ylabel('Accuracy', fontsize=12)
    axes[2].set_title('Mean Accuracy per Class', fontsize=14, fontweight='bold')
    axes[2].set_xticks(classes)
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].set_ylim([0, 1])
    for i, v in enumerate(mean_class_acc):
        if not np.isnan(v):
            axes[2].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Per-class metrics saved to '{save_path}'\n")
    plt.close()

def plot_confusion_matrix(targets, predictions, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    print("Generating confusion matrix...")
    
    cm = confusion_matrix(targets, predictions, labels=range(NUM_CLASSES))
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Raw counts
    im1 = axes[0].imshow(cm, cmap='Blues', aspect='auto')
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xticks(range(NUM_CLASSES))
    axes[0].set_yticks(range(NUM_CLASSES))
    plt.colorbar(im1, ax=axes[0])
    
    # Normalized
    im2 = axes[1].imshow(cm_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xticks(range(NUM_CLASSES))
    axes[1].set_yticks(range(NUM_CLASSES))
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to '{save_path}'\n")
    plt.close()

# ============================================================================
# 5. SAVE REPORT
# ============================================================================
def save_evaluation_report(results, save_path='evaluation_report.txt'):
    """Save detailed evaluation report"""
    print("Generating evaluation report...")
    
    dice_scores = results['dice_scores']
    iou_scores = results['iou_scores']
    pixel_acc = results['pixel_accuracy']
    class_acc = results['class_accuracy']
    
    mean_dice = np.nanmean(dice_scores, axis=0)
    std_dice = np.nanstd(dice_scores, axis=0)
    mean_iou = np.nanmean(iou_scores, axis=0)
    std_iou = np.nanstd(iou_scores, axis=0)
    mean_pixel_acc = np.mean(pixel_acc)
    std_pixel_acc = np.std(pixel_acc)
    mean_class_acc = np.nanmean(class_acc, axis=0)
    
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("3D U-NET SEGMENTATION MODEL EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Model: MONAI 3D U-Net\n")
        f.write(f"Framework: MONAI (Medical Open Network for AI)\n")
        f.write(f"Metrics Library: torchmetrics (optimized)\n")
        f.write(f"Dice Metric: F1Score (mathematically equivalent)\n")
        f.write(f"Number of test volumes: {len(pixel_acc)}\n")
        f.write(f"Number of classes: {NUM_CLASSES} (0=background, 1-7=C1-C7, 8=other)\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean Pixel Accuracy:     {mean_pixel_acc:.6f} ± {std_pixel_acc:.6f}\n")
        f.write(f"Mean Dice Score (macro): {np.nanmean(mean_dice):.6f}\n")
        f.write(f"Mean IoU Score (macro):  {np.nanmean(mean_iou):.6f}\n\n")
        
        f.write("PER-CLASS METRICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Class':<8} {'Dice':<20} {'IoU':<20} {'Accuracy':<12}\n")
        f.write("-"*80 + "\n")
        
        for cls in range(NUM_CLASSES):
            dice_str = f"{mean_dice[cls]:.4f}±{std_dice[cls]:.4f}" if not np.isnan(mean_dice[cls]) else "N/A"
            iou_str = f"{mean_iou[cls]:.4f}±{std_iou[cls]:.4f}" if not np.isnan(mean_iou[cls]) else "N/A"
            acc_str = f"{mean_class_acc[cls]:.4f}" if not np.isnan(mean_class_acc[cls]) else "N/A"
            f.write(f"{cls:<8} {dice_str:<20} {iou_str:<20} {acc_str:<12}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY:\n")
        f.write("="*80 + "\n")
        valid_dice = [d for d in mean_dice if not np.isnan(d)]
        valid_iou = [i for i in mean_iou if not np.isnan(i)]
        if valid_dice:
            f.write(f"Best class (Dice):  Class {np.nanargmax(mean_dice)} ({np.nanmax(mean_dice):.4f})\n")
            f.write(f"Worst class (Dice): Class {np.nanargmin(mean_dice)} ({np.nanmin(mean_dice):.4f})\n")
        if valid_iou:
            f.write(f"Best class (IoU):   Class {np.nanargmax(mean_iou)} ({np.nanmax(mean_iou):.4f})\n")
            f.write(f"Worst class (IoU):  Class {np.nanargmin(mean_iou)} ({np.nanmin(mean_iou):.4f})\n")
    
    print(f"✓ Evaluation report saved to '{save_path}'\n")
    
    # Print summary to console
    print("="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Samples evaluated:     {len(pixel_acc)}")
    print(f"Mean Pixel Accuracy:   {mean_pixel_acc:.4f} ± {std_pixel_acc:.4f}")
    print(f"Mean Dice Score:       {np.nanmean(mean_dice):.4f}")
    print(f"Mean IoU Score:        {np.nanmean(mean_iou):.4f}")
    print("="*80)
    print()

def save_metrics_csv(results, save_path='metrics_per_class.csv'):
    """Save per-class metrics to CSV"""
    print("Saving metrics to CSV...")
    
    mean_dice = np.nanmean(results['dice_scores'], axis=0)
    mean_iou = np.nanmean(results['iou_scores'], axis=0)
    mean_class_acc = np.nanmean(results['class_accuracy'], axis=0)
    
    # Count samples per class
    class_counts = [0] * NUM_CLASSES
    for target in results['targets']:
        if 0 <= target < NUM_CLASSES:
            class_counts[int(target)] += 1
    
    df = pd.DataFrame({
        'Class': range(NUM_CLASSES),
        'Dice_Score': mean_dice,
        'IoU_Score': mean_iou,
        'Accuracy': mean_class_acc,
        'Sample_Count': class_counts
    })
    
    df.to_csv(save_path, index=False, float_format='%.6f')
    print(f"✓ Metrics CSV saved to '{save_path}'\n")

# ============================================================================
# 6. MAIN EVALUATION PIPELINE
# ============================================================================
def main():
    # Create evaluation directory if it doesn't exist
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    print("="*80)
    print("3D U-NET SEGMENTATION MODEL EVALUATION")
    print("MONAI 3D U-Net with torchmetrics (F1Score=Dice, JaccardIndex=IoU)")
    print("="*80)
    print()
    
    # Load model
    model = load_model(MODEL_PATH)
    
    # Prepare dataset using pre-aggregated data
    print("Loading pre-aggregated validation data...")
    print("Aggregating training data from aggregate_data.py...")
    image_volumes, seg_volumes, study_ids, report_dict = aggregate_training_data(
        csv_path=CSV_PATH,
        target_spacing=(1.0, 1.0, 1.0),
        verbose=False  # Less verbose for evaluation
    )
    
    print(f"✓ Loaded {len(image_volumes)} pre-processed volumes")
    
    # Split data by indices for validation (same split as training)
    train_indices, val_indices = train_test_split(
        range(len(study_ids)), 
        test_size=0.2, 
        random_state=42
    )
    
    # Get validation volumes
    val_image_volumes = [image_volumes[i] for i in val_indices]
    val_seg_volumes = [seg_volumes[i] for i in val_indices]
    val_study_ids = [study_ids[i] for i in val_indices]
    
    print(f"  Validation samples: {len(val_image_volumes)}")
    
    # Create validation dataset from pre-aggregated data
    val_dataset = Aggregated3DSegmentationDataset(
        image_volumes=val_image_volumes,
        seg_volumes=val_seg_volumes,
        study_ids=val_study_ids,
        target_shape=(128, 256, 256),  # Match training.py
        augment=False  # No augmentation for evaluation
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"✓ 3D Validation dataset prepared")
    print(f"  Studies: {len(val_study_ids)}")
    print(f"  3D Volumes: {len(val_dataset)}")
    print(f"  Volume shape: (128, 256, 256)")
    print()
    
    # Evaluate model
    results = evaluate_model(model, val_loader, num_classes=NUM_CLASSES)
    
    # Generate visualizations
    plot_sample_predictions(model, val_dataset, num_samples=8, 
                           save_path=os.path.join(EVAL_DIR, 'sample_predictions.png'))
    plot_metrics_per_class(results['dice_scores'], results['iou_scores'], 
                          results['class_accuracy'], save_path=os.path.join(EVAL_DIR, 'metrics_per_class.png'))
    plot_confusion_matrix(results['targets'], results['predictions'], 
                         save_path=os.path.join(EVAL_DIR, 'confusion_matrix.png'))
    
    # Save reports
    save_evaluation_report(results, save_path=os.path.join(EVAL_DIR, 'evaluation_report.txt'))
    save_metrics_csv(results, save_path=os.path.join(EVAL_DIR, 'metrics_per_class.csv'))
    
    # Save raw metrics
    np.savez(os.path.join(EVAL_DIR, 'evaluation_metrics.npz'),
             dice_scores=results['dice_scores'],
             iou_scores=results['iou_scores'],
             pixel_accuracy=results['pixel_accuracy'],
             class_accuracy=results['class_accuracy'])
    print(f"✓ Raw metrics saved to '{os.path.join(EVAL_DIR, 'evaluation_metrics.npz')}'\n")
    
    print("="*80)
    print("3D EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nGenerated files in '{EVAL_DIR}/' directory:")
    print("  1. evaluation_report.txt     - Detailed 3D metrics report")
    print("  2. metrics_per_class.csv     - Per-class metrics table")
    print("  3. evaluation_metrics.npz    - Raw metrics data")
    print("  4. sample_predictions.png    - Sample 3D volume slices")
    print("  5. metrics_per_class.png     - Per-class performance charts")
    print("  6. confusion_matrix.png      - Confusion matrix")
    print("="*80)

if __name__ == "__main__":
    main()