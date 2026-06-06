"""
Comprehensive Segmentation Model Evaluation Script (Optimized with torchmetrics)
Evaluates model performance on validation set with detailed metrics and visualizations
Adapted for cropped YOLO dataset
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from segmentation_models_pytorch import Unet
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from data_process import CroppedSegmentationDataset
import pandas as pd
import time
import warnings
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    print("⚠️  Warning: codecarbon not installed. Energy tracking will be disabled.")
warnings.filterwarnings('ignore')

# Import torchmetrics (F1Score = Dice coefficient for segmentation)
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex, MulticlassAccuracy

# Configuration
DATA_PATH = '../../../../vast/s222440401'
CROPPED_DATASET_DIR = os.path.join(DATA_PATH, 'cropped_yolo_dataset')
MODEL_PATH = 'best_unet_model.pth'
BATCH_SIZE = 16
NUM_CLASSES = 9  # 0=Background, 1-7=C1-C7, 8=Other vertebrae merged
EVAL_DIR = "evaluation_report"

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# ============================================================================
# 1. LOAD MODEL
# ============================================================================
def load_model(model_path=MODEL_PATH):
    """Load trained U-Net model"""
    print("Loading model...")
    model = Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=3,  # RGB - multi-window HU transformation
        classes=NUM_CLASSES
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded from {model_path}")
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
    Calculate all metrics using torchmetrics
    Returns both per-class and averaged metrics
    """
    # Ensure correct dimensions for torchmetrics
    if pred.dim() == 2:  # Single image
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    # Per-class metrics
    dice_scores = dice_per_class(pred, target).cpu().numpy()
    iou_scores = iou_per_class(pred, target).cpu().numpy()
    class_acc = acc_per_class(pred, target).cpu().numpy()
    
    # Overall metrics
    dice_avg = dice_macro(pred, target).item()
    iou_avg = iou_macro(pred, target).item()
    pixel_acc = acc_micro(pred, target).item()
    
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
    """Evaluate model on entire dataset"""
    model.eval()
    
    all_dice_scores = []
    all_iou_scores = []
    all_pixel_acc = []
    all_class_acc = []
    
    all_preds = []
    all_targets = []
    
    # Inference timing
    inference_times = []
    total_inference_start = time.time()
    
    print("Evaluating model on validation set...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Processing batches"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Time inference
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_start = time.time()
            
            # Get predictions
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_time = time.time() - inference_start
            inference_times.append(inference_time)
            
            # Calculate metrics for each image in batch
            for pred, target in zip(preds, labels):
                metrics = calculate_metrics(pred, target)
                
                all_dice_scores.append(metrics['dice_per_class'])
                all_iou_scores.append(metrics['iou_per_class'])
                all_pixel_acc.append(metrics['pixel_accuracy'])
                all_class_acc.append(metrics['class_accuracy'])
                
                # Store for confusion matrix (sample to avoid memory issues)
                if len(all_preds) < 100000:  # Limit samples for confusion matrix
                    all_preds.extend(pred.cpu().numpy().flatten()[::10])  # Sample every 10th pixel
                    all_targets.extend(target.cpu().numpy().flatten()[::10])
    
    total_inference_time = time.time() - total_inference_start
    
    # Calculate inference statistics
    avg_inference_time = np.mean(inference_times)
    median_inference_time = np.median(inference_times)
    std_inference_time = np.std(inference_times)
    fps = len(inference_times) / total_inference_time  # batches per second
    samples_per_second = len(all_pixel_acc) / total_inference_time
    
    results = {
        'dice_scores': np.array(all_dice_scores),
        'iou_scores': np.array(all_iou_scores),
        'pixel_accuracy': np.array(all_pixel_acc),
        'class_accuracy': np.array(all_class_acc),
        'predictions': np.array(all_preds),
        'targets': np.array(all_targets),
        'inference_stats': {
            'total_time_seconds': total_inference_time,
            'avg_batch_time_ms': avg_inference_time * 1000,
            'median_batch_time_ms': median_inference_time * 1000,
            'std_batch_time_ms': std_inference_time * 1000,
            'batches_per_second': fps,
            'samples_per_second': samples_per_second,
            'ms_per_sample': (total_inference_time / len(all_pixel_acc)) * 1000
        }
    }
    
    return results

# ============================================================================
# 4. VISUALIZATION FUNCTIONS
# ============================================================================
def plot_sample_predictions(model, dataset, num_samples=6, save_path='sample_predictions.png'):
    """Visualize sample predictions"""
    print(f"Generating sample predictions ({num_samples} samples)...")
    model.eval()
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            image, label = dataset[sample_idx]
            image_input = image.unsqueeze(0).to(device)
            
            # Get prediction
            output = model(image_input)
            pred = torch.argmax(output, dim=1).squeeze()
            
            # Calculate metrics using torchmetrics
            metrics = calculate_metrics(pred, label.to(device))
            
            # Convert to numpy
            img_np = image.squeeze().cpu().numpy()
            label_np = label.cpu().numpy()
            pred_np = pred.cpu().numpy()
            
            # Transpose image from (C, H, W) to (H, W, C) for matplotlib
            if img_np.ndim == 3 and img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            
            # Normalize image to [0, 1] range if needed
            if img_np.max() > 1.0:
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            
            # Plot
            axes[idx, 0].imshow(img_np)
            axes[idx, 0].set_title('Input Image', fontsize=12, fontweight='bold')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(label_np, cmap='tab10', vmin=0, vmax=8)
            axes[idx, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(pred_np, cmap='tab10', vmin=0, vmax=8)
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
    class_names = ['Background', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'Other']
    
    # Dice scores
    axes[0].bar(classes, mean_dice, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Dice Score', fontsize=12)
    axes[0].set_title('Mean Dice Score per Class', fontsize=14, fontweight='bold')
    axes[0].set_xticks(classes)
    axes[0].set_xticklabels([f'{i}\n{name}' if i < len(class_names) else f'{i}' for i, name in enumerate(class_names)], fontsize=9)
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
    axes[1].set_xticklabels([f'{i}\n{name}' if i < len(class_names) else f'{i}' for i, name in enumerate(class_names)], fontsize=9)
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
    axes[2].set_xticklabels([f'{i}\n{name}' if i < len(class_names) else f'{i}' for i, name in enumerate(class_names)], fontsize=9)
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
    inference_stats = results.get('inference_stats', {})
    
    mean_dice = np.nanmean(dice_scores, axis=0)
    std_dice = np.nanstd(dice_scores, axis=0)
    mean_iou = np.nanmean(iou_scores, axis=0)
    std_iou = np.nanstd(iou_scores, axis=0)
    mean_pixel_acc = np.mean(pixel_acc)
    std_pixel_acc = np.std(pixel_acc)
    mean_class_acc = np.nanmean(class_acc, axis=0)
    
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SEGMENTATION MODEL EVALUATION REPORT (CROPPED DATASET)\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Model: U-Net with EfficientNet-B0 encoder\n")
        f.write(f"Dataset: Cropped YOLO Dataset\n")
        f.write(f"Metrics Library: torchmetrics (optimized)\n")
        f.write(f"Dice Metric: F1Score (mathematically equivalent)\n")
        f.write(f"Number of test samples: {len(pixel_acc)}\n")
        f.write(f"Number of classes: {NUM_CLASSES}\n\n")
        
        # Add inference performance section
        if inference_stats:
            f.write("INFERENCE PERFORMANCE:\n")
            f.write("-"*80 + "\n")
            f.write(f"Total inference time: {inference_stats.get('total_time_seconds', 0):.2f} seconds\n")
            f.write(f"Average batch time: {inference_stats.get('avg_batch_time_ms', 0):.2f} ms\n")
            f.write(f"Throughput: {inference_stats.get('samples_per_second', 0):.2f} samples/second\n")
            f.write(f"Latency per sample: {inference_stats.get('ms_per_sample', 0):.2f} ms\n")
            f.write("\n")
        
        f.write("OVERALL METRICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean Pixel Accuracy:     {mean_pixel_acc:.6f} ± {std_pixel_acc:.6f}\n")
        f.write(f"Mean Dice Score (macro): {np.nanmean(mean_dice):.6f}\n")
        f.write(f"Mean IoU Score (macro):  {np.nanmean(mean_iou):.6f}\n\n")
        
        f.write("PER-CLASS METRICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Class':<8} {'Dice':<20} {'IoU':<20} {'Accuracy':<12}\n")
        f.write("-"*80 + "\n")
        
        class_names = ['Background', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'Other']
        for cls in range(NUM_CLASSES):
            dice_str = f"{mean_dice[cls]:.4f}±{std_dice[cls]:.4f}" if not np.isnan(mean_dice[cls]) else "N/A"
            iou_str = f"{mean_iou[cls]:.4f}±{std_iou[cls]:.4f}" if not np.isnan(mean_iou[cls]) else "N/A"
            acc_str = f"{mean_class_acc[cls]:.4f}" if not np.isnan(mean_class_acc[cls]) else "N/A"
            cls_name = class_names[cls] if cls < len(class_names) else f'Class {cls}'
            f.write(f"{cls} ({cls_name[:3]}) {dice_str:<20} {iou_str:<20} {acc_str:<12}\n")
        
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

# ============================================================================
# 6. MAIN EVALUATION PIPELINE
# ============================================================================
def main():
    """Main evaluation pipeline"""
    print("="*80)
    print("SEGMENTATION MODEL EVALUATION (CROPPED DATASET)")
    print("="*80)
    print()
    
    # Create evaluation directory
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    # Load model
    model = load_model()
    
    # Load validation dataset
    print("Loading validation dataset...")
    val_dataset = CroppedSegmentationDataset(
        split='val',
        cropped_dataset_dir=CROPPED_DATASET_DIR,
        augment=False
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"✓ Validation dataset loaded: {len(val_dataset)} samples\n")
    
    # Initialize energy tracking for inference
    inference_tracker = None
    if CODECARBON_AVAILABLE:
        try:
            inference_tracker = EmissionsTracker(
                output_dir="./",
                output_file="inference_emissions.csv",
                log_level="error",
                measure_power_secs=30,
                save_to_file=True
            )
            inference_tracker.start()
        except Exception as e:
            print(f"⚠️  Warning: Failed to initialize inference energy tracker: {e}")
    
    # Evaluate model
    results = evaluate_model(model, val_loader)
    
    # Stop energy tracking
    if inference_tracker:
        try:
            inference_tracker.stop()
            emissions_data = inference_tracker.final_emissions_data
            if emissions_data:
                energy = getattr(emissions_data, 'energy_consumed_kWh', 0)
                co2 = getattr(emissions_data, 'emissions', 0)
                results['inference_stats']['energy_kwh'] = energy
                results['inference_stats']['co2_kg'] = co2
        except Exception as e:
            print(f"⚠️  Warning: Failed to get inference emissions: {e}")
    
    # Save metrics to CSV
    dice_scores = results['dice_scores']
    iou_scores = results['iou_scores']
    class_acc = results['class_accuracy']
    
    mean_dice = np.nanmean(dice_scores, axis=0)
    std_dice = np.nanstd(dice_scores, axis=0)
    mean_iou = np.nanmean(iou_scores, axis=0)
    std_iou = np.nanstd(iou_scores, axis=0)
    mean_class_acc = np.nanmean(class_acc, axis=0)
    
    metrics_df = pd.DataFrame({
        'Class': range(NUM_CLASSES),
        'Dice_Mean': mean_dice,
        'Dice_Std': std_dice,
        'IoU_Mean': mean_iou,
        'IoU_Std': std_iou,
        'Accuracy': mean_class_acc
    })
    metrics_df.to_csv(os.path.join(EVAL_DIR, 'metrics_per_class.csv'), index=False)
    print(f"✓ Metrics saved to '{EVAL_DIR}/metrics_per_class.csv'\n")
    
    # Generate visualizations
    plot_sample_predictions(model, val_dataset, num_samples=100, 
                           save_path=os.path.join(EVAL_DIR, 'sample_predictions.png'))
    plot_metrics_per_class(dice_scores, iou_scores, class_acc,
                          save_path=os.path.join(EVAL_DIR, 'metrics_per_class.png'))
    plot_confusion_matrix(results['targets'], results['predictions'],
                         save_path=os.path.join(EVAL_DIR, 'confusion_matrix.png'))
    
    # Save evaluation report
    save_evaluation_report(results, save_path=os.path.join(EVAL_DIR, 'evaluation_report.txt'))
    
    # Save results for later analysis
    np.savez(os.path.join(EVAL_DIR, 'evaluation_metrics.npz'),
             dice_scores=dice_scores,
             iou_scores=iou_scores,
             pixel_accuracy=results['pixel_accuracy'],
             class_accuracy=class_acc)
    print(f"✓ Evaluation metrics saved to '{EVAL_DIR}/evaluation_metrics.npz'\n")
    
    print("="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"Results saved in '{EVAL_DIR}/' directory")
    print(f"  - evaluation_report.txt")
    print(f"  - metrics_per_class.csv")
    print(f"  - sample_predictions.png")
    print(f"  - metrics_per_class.png")
    print(f"  - confusion_matrix.png")
    print(f"  - evaluation_metrics.npz")
    print("="*80)

if __name__ == "__main__":
    main()

