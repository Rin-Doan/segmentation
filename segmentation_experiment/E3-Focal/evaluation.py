"""
Comprehensive Segmentation Model Evaluation Script (Optimized with torchmetrics)
Evaluates model performance on validation set with detailed metrics and visualizations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from segmentation_models_pytorch import Unet
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from data_process import MedicalSegmentationDataset
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

# Figure output settings (journal-ready: 174 mm wide, 600 DPI)
FIG_WIDTH_MM = 174
FIG_WIDTH_IN = FIG_WIDTH_MM / 25.4  # ≈ 6.85 inches
FIG_DPI = 600

# Configuration
DATA_PATH = '../../../../../vast/s222440401'
TRAINING_PATH = DATA_PATH + '/training_images'
SEGMENTATION_PATH = DATA_PATH + '/segmentations'
MODEL_PATH = 'best_unet_model.pth'
BATCH_SIZE = 16
SKIP_SLICE = 1
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
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
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
    
    print(f"✓ Evaluation complete on {len(all_pixel_acc)} samples")
    print(f"⏱️  Inference: {total_inference_time:.2f}s total, {avg_inference_time*1000:.2f}ms/batch avg, {samples_per_second:.2f} samples/s\n")
    return results

# ============================================================================
# 4. VISUALIZATION FUNCTIONS
# ============================================================================
def plot_sample_predictions(model, dataset, num_samples=6, save_path='sample_predictions.png'):
    """Visualize sample predictions"""
    print(f"Generating sample predictions ({num_samples} samples)...")
    model.eval()
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * (5 * num_samples) / 15))
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
            
            # Handle multi-channel images (2.5D: 3 channels)
            # For visualization, use the middle channel (index 1) for 2.5D
            if len(img_np.shape) == 3 and img_np.shape[0] == 3:
                img_np = img_np[1]  # Take middle channel (the main slice)
            elif len(img_np.shape) == 3 and img_np.shape[0] > 3:
                img_np = img_np[img_np.shape[0] // 2]  # Take middle channel if more than 3
            
            # Plot
            axes[idx, 0].imshow(img_np, cmap='gray')
            axes[idx, 0].set_title('Input Image', fontsize=12, fontweight='bold')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(label_np, cmap='tab20', vmin=0, vmax=8)
            axes[idx, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(pred_np, cmap='tab20', vmin=0, vmax=8)
            axes[idx, 2].set_title(f'Prediction\nAcc: {metrics["pixel_accuracy"]:.3f} | Dice: {metrics["mean_dice"]:.3f}', 
                                  fontsize=12, fontweight='bold')
            axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    print(f"✓ Sample predictions saved to '{save_path}'\n")
    plt.close()

def plot_metrics_per_class(dice_scores, iou_scores, class_acc, save_path='metrics_per_class.png'):
    """Plot per-class metrics"""
    print("Generating per-class metrics plot...")
    
    mean_dice = np.nanmean(dice_scores, axis=0)
    mean_iou = np.nanmean(iou_scores, axis=0)
    mean_class_acc = np.nanmean(class_acc, axis=0)
    
    fig, axes = plt.subplots(3, 1, figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 12 / 14))
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
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    print(f"✓ Per-class metrics saved to '{save_path}'\n")
    plt.close()

def plot_confusion_matrix(targets, predictions, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    print("Generating confusion matrix...")
    
    cm = confusion_matrix(targets, predictions, labels=range(NUM_CLASSES))
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 8 / 20))
    
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
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
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
        f.write("SEGMENTATION MODEL EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Model: U-Net with ResNet34 encoder\n")
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

def benchmark_inference_latency(model, dataset, num_samples=100, warmup=10):
    """Benchmark single-image inference latency"""
    print(f"Benchmarking inference latency ({num_samples} samples, {warmup} warmup)...")
    model.eval()
    
    # Warmup
    for i in range(warmup):
        image, _ = dataset[i % len(dataset)]
        image_input = image.unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(image_input)
    
    # Benchmark
    latencies = []
    with torch.no_grad():
        for i in range(num_samples):
            image, _ = dataset[i % len(dataset)]
            image_input = image.unsqueeze(0).to(device)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.time()
            _ = model(image_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            latencies.append((time.time() - start) * 1000)  # Convert to ms
    
    return {
        'mean_latency_ms': np.mean(latencies),
        'median_latency_ms': np.median(latencies),
        'std_latency_ms': np.std(latencies),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
        'fps': 1000 / np.mean(latencies)
    }

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
    print("SEGMENTATION MODEL EVALUATION")
    print("Optimized with torchmetrics (F1Score=Dice, JaccardIndex=IoU)")
    print("="*80)
    print()
    
    # Load model
    model = load_model(MODEL_PATH)
    
    # Prepare dataset
    print("Preparing validation dataset...")
    training_studies = [d for d in os.listdir(TRAINING_PATH) 
                       if os.path.isdir(os.path.join(TRAINING_PATH, d))]
    segmentation_files = [f for f in os.listdir(SEGMENTATION_PATH) 
                         if f.endswith(('.nii', '.nii.gz'))]
    segmentation_studies = [f.replace('.nii.gz', '').replace('.nii', '') 
                           for f in segmentation_files]
    overlapping_studies = sorted(list(set(training_studies).intersection(set(segmentation_studies))))
    
    _, val_studies = train_test_split(overlapping_studies, test_size=0.2, random_state=42)
    
    val_dataset = MedicalSegmentationDataset(val_studies, TRAINING_PATH, SEGMENTATION_PATH, SKIP_SLICE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"✓ Validation dataset prepared")
    print(f"  Studies: {len(val_studies)}")
    print(f"  Samples: {len(val_dataset)}")
    print()
    
    # Initialize energy/carbon tracking for inference
    emissions_tracker = None
    if CODECARBON_AVAILABLE:
        try:
            emissions_tracker = EmissionsTracker(
                output_dir="./training_reports",
                output_file="inference_emissions.csv",
                log_level="error",
                measure_power_secs=30,
                save_to_file=True
            )
            emissions_tracker.start()
            print("🌍 Energy/carbon tracking enabled for inference")
        except Exception as e:
            print(f"⚠️  Warning: Failed to initialize energy tracker: {e}")
            emissions_tracker = None
    
    # Track GPU memory before inference
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    
    # Evaluate model
    results = evaluate_model(model, val_loader, num_classes=NUM_CLASSES)
    
    # Get GPU memory after inference
    avg_gpu_memory = 0
    if torch.cuda.is_available():
        peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        avg_gpu_memory = (initial_gpu_memory + peak_gpu_memory) / 2
    
    # Stop energy tracking
    emissions_data = None
    if emissions_tracker is not None:
        try:
            emissions_tracker.stop()
            emissions_data = emissions_tracker.final_emissions_data
        except Exception as e:
            print(f"⚠️  Warning: Failed to get emissions data: {e}")
            emissions_data = None
    
    # Helper function to safely get emissions attribute
    def get_emissions_attr(emissions_obj, attr_name, default=None):
        """Safely get attribute from EmissionsData object"""
        if emissions_obj is None:
            return default
        try:
            return getattr(emissions_obj, attr_name, default)
        except (AttributeError, TypeError):
            return default
    
    # Benchmark single-image inference latency
    latency_stats = benchmark_inference_latency(model, val_dataset, num_samples=100)
    print("="*80)
    print("INFERENCE LATENCY BENCHMARK")
    print("="*80)
    print(f"Mean latency: {latency_stats['mean_latency_ms']:.2f} ms")
    print(f"Median latency: {latency_stats['median_latency_ms']:.2f} ms")
    print(f"P95 latency: {latency_stats['p95_latency_ms']:.2f} ms")
    print(f"P99 latency: {latency_stats['p99_latency_ms']:.2f} ms")
    print(f"Throughput: {latency_stats['fps']:.2f} FPS")
    print("="*80)
    print()
    
    # Add to results
    results['latency_stats'] = latency_stats
    
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
    
    # Calculate overall metrics for inference.csv
    mean_pixel_acc = np.mean(results['pixel_accuracy'])
    mean_dice = np.nanmean(np.nanmean(results['dice_scores'], axis=0))
    mean_iou = np.nanmean(np.nanmean(results['iou_scores'], axis=0))
    
    # Get inference time from results
    inference_stats = results.get('inference_stats', {})
    total_inference_time = inference_stats.get('total_time_seconds', 0)
    
    # Save inference.csv to training_reports directory
    inference_dir = 'training_reports'
    os.makedirs(inference_dir, exist_ok=True)
    
    inference_data = {
        'Total Inference Time (hours)': [total_inference_time / 3600],
        'Total Inference Time (second)': [total_inference_time],
        'Average GPU Memory (GB)': [avg_gpu_memory],
        'Energy Consumed (kWh)': [get_emissions_attr(emissions_data, 'energy_consumed_kWh')],
        'CO2 Emissions (kg)': [get_emissions_attr(emissions_data, 'emissions')],
        'Accuracy': [mean_pixel_acc],
        'Dice Score': [mean_dice],
        'IoU': [mean_iou],
    }
    
    inference_file = os.path.join(inference_dir, 'inference.csv')
    if os.path.exists(inference_file):
        df_existing = pd.read_csv(inference_file)
        df_new = pd.DataFrame(inference_data)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(inference_file, index=False)
        print(f"✓ Inference report appended to '{inference_file}' (Total runs: {len(df_combined)})")
    else:
        df = pd.DataFrame(inference_data)
        df.to_csv(inference_file, index=False)
        print(f"✓ Inference report saved to '{inference_file}' (New file created)")
    
    print("="*80)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files in 'evaluation_report/' directory:")
    print("  1. evaluation_report.txt     - Detailed metrics report")
    print("  2. metrics_per_class.csv     - Per-class metrics table")
    print("  3. evaluation_metrics.npz    - Raw metrics data")
    print("  4. sample_predictions.png    - Sample predictions visualization")
    print("  5. metrics_per_class.png     - Per-class performance charts")
    print("  6. confusion_matrix.png      - Confusion matrix")
    print("="*80)

if __name__ == "__main__":
    main()