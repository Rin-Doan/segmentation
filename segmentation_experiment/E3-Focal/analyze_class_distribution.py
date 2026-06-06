"""
Analyze Class Distribution in Segmentation Dataset

This script analyzes the class distribution in your training data
and recommends appropriate class weights to address imbalance.
"""

import os
import numpy as np
import nibabel as nib
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATA_PATH = '../../../../vast/s222440401'
TRAINING_PATH = DATA_PATH + '/training_images'
SEGMENTATION_PATH = DATA_PATH + '/segmentations'

def analyze_distribution():
    """Analyze and visualize class distribution"""
    
    print("="*80)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*80)
    print()
    
    # Get studies
    print("Loading study information...")
    training_studies = [d for d in os.listdir(TRAINING_PATH) 
                       if os.path.isdir(os.path.join(TRAINING_PATH, d))]
    segmentation_files = [f for f in os.listdir(SEGMENTATION_PATH) 
                         if f.endswith(('.nii', '.nii.gz'))]
    segmentation_studies = [f.replace('.nii.gz', '').replace('.nii', '') 
                           for f in segmentation_files]
    overlapping_studies = sorted(list(set(training_studies).intersection(set(segmentation_studies))))
    
    train_studies, val_studies = train_test_split(overlapping_studies, test_size=0.2, random_state=42)
    
    print(f"Total studies: {len(overlapping_studies)}")
    print(f"Training studies: {len(train_studies)}")
    print(f"Validation studies: {len(val_studies)}")
    print()
    
    # Analyze training set
    print("Analyzing class distribution in training set...")
    class_counts = Counter()
    total_pixels = 0
    vertebrae_slices = 0
    total_slices = 0
    
    for study_id in train_studies:
        seg_path = os.path.join(SEGMENTATION_PATH, f"{study_id}.nii")
        if not os.path.exists(seg_path):
            seg_path = os.path.join(SEGMENTATION_PATH, f"{study_id}.nii.gz")
        
        if not os.path.exists(seg_path):
            continue
        
        nii = nib.load(seg_path)
        seg_volume = nii.get_fdata()
        seg_corrected = seg_volume[:, ::-1, ::-1].transpose(2, 1, 0)
        
        # Apply same remapping as in dataset
        seg_corrected = np.where(seg_corrected > 7, 8, seg_corrected)
        
        # Count slices with vertebrae
        for i in range(seg_corrected.shape[0]):
            total_slices += 1
            slice_data = seg_corrected[i]
            if np.any((slice_data > 0) & (slice_data <= 8)):
                vertebrae_slices += 1
        
        # Count pixels per class
        unique, counts = np.unique(seg_corrected, return_counts=True)
        for cls, count in zip(unique, counts):
            class_counts[int(cls)] += count
            total_pixels += count
    
    print(f"✓ Analysis complete!")
    print()
    
    # Display results
    print("="*80)
    print("PIXEL-LEVEL CLASS DISTRIBUTION")
    print("="*80)
    class_names = ['Background', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'Other Vert.']
    
    print(f"{'Class':<5} {'Name':<15} {'Pixel Count':<15} {'Percentage':<12} {'Weight (1/freq)':<15}")
    print("-"*80)
    
    class_weights_inv_freq = []
    class_percentages = []
    
    for cls in range(9):
        count = class_counts.get(cls, 0)
        percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
        weight = total_pixels / (9 * count) if count > 0 else 0
        
        class_weights_inv_freq.append(weight)
        class_percentages.append(percentage)
        
        print(f"{cls:<5} {class_names[cls]:<15} {count:<15,} {percentage:<12.4f}% {weight:<15.4f}")
    
    print(f"\nTotal pixels: {total_pixels:,}")
    print(f"Total slices: {total_slices}")
    print(f"Slices with vertebrae: {vertebrae_slices} ({vertebrae_slices/total_slices*100:.1f}%)")
    print()
    
    # Normalize weights
    class_weights_normalized = np.array(class_weights_inv_freq)
    class_weights_normalized = class_weights_normalized / class_weights_normalized.min()
    
    print("="*80)
    print("RECOMMENDED CLASS WEIGHTS")
    print("="*80)
    print("\n1. INVERSE FREQUENCY WEIGHTS (normalized to min=1.0):")
    print("-"*80)
    print("class_weights = torch.FloatTensor([")
    for cls in range(9):
        print(f"    {class_weights_normalized[cls]:.4f},  # Class {cls}: {class_names[cls]}")
    print("]).to(device)")
    print()
    
    # Alternative: Balanced weights (reduce background dominance)
    class_weights_balanced = class_weights_normalized.copy()
    class_weights_balanced[0] = class_weights_balanced[0] * 0.01  # Heavily reduce background weight
    class_weights_balanced[1:] = class_weights_balanced[1:] * 1.5  # Boost vertebrae classes
    
    print("2. BALANCED WEIGHTS (reduced background, boosted vertebrae):")
    print("-"*80)
    print("class_weights = torch.FloatTensor([")
    for cls in range(9):
        print(f"    {class_weights_balanced[cls]:.4f},  # Class {cls}: {class_names[cls]}")
    print("]).to(device)")
    print()
    
    # Manual recommendation
    print("3. MANUAL RECOMMENDATION (based on analysis):")
    print("-"*80)
    print("class_weights = torch.FloatTensor([")
    print(f"    0.1,    # Class 0: Background (very frequent)")
    for cls in range(1, 8):
        print(f"    5.0,    # Class {cls}: {class_names[cls]} (rare)")
    print(f"    3.0     # Class 8: Other Vertebrae (moderate)")
    print("]).to(device)")
    print()
    
    # Calculate imbalance ratio
    max_count = max(class_counts.values())
    min_count = min([c for c in class_counts.values() if c > 0])
    imbalance_ratio = max_count / min_count
    
    print("="*80)
    print("IMBALANCE STATISTICS")
    print("="*80)
    print(f"Most frequent class:  Class 0 ({class_counts[0]:,} pixels)")
    print(f"Least frequent class: Class {min(class_counts, key=lambda k: class_counts[k] if class_counts[k] > 0 else float('inf'))}")
    print(f"Imbalance ratio:      {imbalance_ratio:.2f}x")
    print()
    
    if imbalance_ratio > 100:
        print("⚠️  SEVERE CLASS IMBALANCE DETECTED!")
        print("   Recommendation: Use Focal Loss or Combined Loss with strong class weights")
    elif imbalance_ratio > 10:
        print("⚠️  MODERATE CLASS IMBALANCE")
        print("   Recommendation: Use class-weighted CrossEntropy or Focal Loss")
    else:
        print("✓ MILD CLASS IMBALANCE")
        print("   Recommendation: Standard CrossEntropy should work")
    
    # Create visualization
    create_visualization(class_names, class_percentages, class_weights_normalized)
    
    return class_weights_normalized, class_percentages


def create_visualization(class_names, percentages, weights):
    """Create visualization of class distribution"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Class distribution (percentage)
    colors = ['gray', 'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'pink']
    bars1 = axes[0].bar(range(9), percentages, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Class', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Percentage of Total Pixels', fontsize=12, fontweight='bold')
    axes[0].set_title('Class Distribution (Pixel Count)', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(9))
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars1, percentages)):
        if pct > 0.1:  # Only show label if > 0.1%
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(percentages)*0.01,
                        f'{pct:.2f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Recommended class weights
    bars2 = axes[1].bar(range(9), weights, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Class', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Recommended Weight (normalized)', fontsize=12, fontweight='bold')
    axes[1].set_title('Recommended Class Weights', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(9))
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add weight labels on bars
    for i, (bar, w) in enumerate(zip(bars2, weights)):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(weights)*0.01,
                    f'{w:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('class_distribution_analysis.png', dpi=200, bbox_inches='tight')
    print("="*80)
    print("✓ Visualization saved to 'class_distribution_analysis.png'")
    print("="*80)
    plt.close()


if __name__ == "__main__":
    try:
        weights, percentages = analyze_distribution()
        print("\n✅ Analysis complete! Use the recommended weights in your training script.")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

