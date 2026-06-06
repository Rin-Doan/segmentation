"""
Class Distribution Analysis for RSNA Vertebrae Segmentation Data

This script analyzes and visualizes the distribution of background and vertebrae
pixels in the RSNA dataset to demonstrate the severe class imbalance problem.

Author: Analysis Script
Date: 2024
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# Configuration
DATA_PATH = '../../../../../vast/s222440401'
SEGMENTATION_PATH = os.path.join(DATA_PATH, 'segmentations')
OUTPUT_DIR = Path(__file__).parent
SKIP_SLICE = 1  # Process every slice
NUM_CLASSES = 9  # 0=Background, 1-7=C1-C7, 8=Other vertebrae

# Class labels for visualization
CLASS_LABELS = {
    0: 'Background',
    1: 'C1',
    2: 'C2',
    3: 'C3',
    4: 'C4',
    5: 'C5',
    6: 'C6',
    7: 'C7',
    8: 'Other Vertebrae'
}

# Colors for each class
CLASS_COLORS = {
    0: '#2E86AB',      # Blue for background
    1: '#A23B72',      # Purple
    2: '#F18F01',      # Orange
    3: '#C73E1D',      # Red
    4: '#6A994E',      # Green
    5: '#BC4749',      # Dark red
    6: '#F77F00',      # Dark orange
    7: '#FCBF49',       # Yellow
    8: '#D62828'        # Dark red
}


def load_segmentation_files(segmentation_path):
    """Load all segmentation NIfTI files"""
    print("Loading segmentation files...")
    segmentation_files = []
    
    for f in os.listdir(segmentation_path):
        if f.endswith('.nii.gz') or f.endswith('.nii'):
            study_id = f.replace('.nii.gz', '').replace('.nii', '')
            seg_path = os.path.join(segmentation_path, f)
            segmentation_files.append((study_id, seg_path))
    
    print(f"Found {len(segmentation_files)} segmentation files")
    return segmentation_files


def analyze_single_segmentation(seg_path, skip_slice=1):
    """
    Analyze a single segmentation file and count pixels per class
    
    Returns:
        class_counts: Dictionary with class counts
        total_pixels: Total number of pixels analyzed
        num_slices: Number of slices processed
    """
    try:
        nii = nib.load(seg_path)
        seg_volume = nii.get_fdata()
        # Apply same transformation as in data_process.py
        seg_corrected = seg_volume[:, ::-1, ::-1].transpose(2, 1, 0)
        
        # Initialize class counts
        class_counts = {i: 0 for i in range(NUM_CLASSES)}
        total_pixels = 0
        num_slices = 0
        
        # Process each slice
        for i in range(0, seg_corrected.shape[0], skip_slice):
            seg_slice = seg_corrected[i].astype(np.int64)
            
            # Remap values > 8 to class 8 (same as in data_process.py)
            seg_slice = np.where(seg_slice > 8, 8, seg_slice)
            seg_slice = np.clip(seg_slice, 0, 8)
            
            # Count pixels per class
            unique, counts = np.unique(seg_slice, return_counts=True)
            for cls, count in zip(unique, counts):
                if 0 <= cls < NUM_CLASSES:
                    class_counts[int(cls)] += int(count)
            
            total_pixels += seg_slice.size
            num_slices += 1
        
        return class_counts, total_pixels, num_slices
    
    except Exception as e:
        print(f"Error processing {seg_path}: {e}")
        return None, 0, 0


def analyze_all_segmentations(segmentation_files, max_files=None):
    """
    Analyze all segmentation files and aggregate class distribution
    
    Args:
        segmentation_files: List of (study_id, seg_path) tuples
        max_files: Maximum number of files to process (None for all)
    
    Returns:
        aggregated_counts: Dictionary with total counts per class
        per_file_stats: List of dictionaries with per-file statistics
        total_pixels: Total pixels across all files
    """
    print(f"\nAnalyzing segmentation files...")
    if max_files:
        segmentation_files = segmentation_files[:max_files]
        print(f"Processing first {max_files} files (for faster analysis)")
    
    aggregated_counts = {i: 0 for i in range(NUM_CLASSES)}
    per_file_stats = []
    total_pixels = 0
    total_slices = 0
    
    for study_id, seg_path in tqdm(segmentation_files, desc="Processing files"):
        class_counts, file_pixels, num_slices = analyze_single_segmentation(seg_path, SKIP_SLICE)
        
        if class_counts is not None:
            # Aggregate counts
            for cls in range(NUM_CLASSES):
                aggregated_counts[cls] += class_counts[cls]
            
            total_pixels += file_pixels
            total_slices += num_slices
            
            # Store per-file statistics
            per_file_stats.append({
                'study_id': study_id,
                'total_pixels': file_pixels,
                'num_slices': num_slices,
                **{f'class_{cls}': class_counts[cls] for cls in range(NUM_CLASSES)}
            })
    
    print(f"\nAnalysis complete!")
    print(f"  Total files processed: {len(per_file_stats)}")
    print(f"  Total slices: {total_slices:,}")
    print(f"  Total pixels: {total_pixels:,}")
    
    return aggregated_counts, per_file_stats, total_pixels


def create_class_distribution_plots(aggregated_counts, total_pixels, output_dir):
    """Create comprehensive visualizations of class distribution"""
    
    # Prepare data
    classes = list(range(NUM_CLASSES))
    counts = [aggregated_counts[cls] for cls in classes]
    percentages = [(count / total_pixels * 100) if total_pixels > 0 else 0 
                   for count in counts]
    labels = [CLASS_LABELS[cls] for cls in classes]
    colors = [CLASS_COLORS[cls] for cls in classes]
    
    # Separate background and vertebrae
    background_count = aggregated_counts[0]
    vertebrae_count = sum(aggregated_counts[i] for i in range(1, NUM_CLASSES))
    background_pct = (background_count / total_pixels * 100) if total_pixels > 0 else 0
    vertebrae_pct = (vertebrae_count / total_pixels * 100) if total_pixels > 0 else 0
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Pie Chart - Overall Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    pie_data = [background_pct, vertebrae_pct]
    pie_labels = ['Background', 'Vertebrae (All)']
    pie_colors = [CLASS_COLORS[0], '#FF6B6B']
    wedges, texts, autotexts = ax1.pie(pie_data, labels=pie_labels, colors=pie_colors,
                                       autopct='%1.2f%%', startangle=90,
                                       textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax1.set_title('Background vs Vertebrae Distribution', fontsize=14, fontweight='bold', pad=20)
    
    # 2. Bar Chart - Pixel Counts (Log Scale)
    ax3 = fig.add_subplot(gs[0, 1])
    bars = ax3.bar(classes, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_yscale('log')
    ax3.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Pixel Count (Log Scale)', fontsize=12, fontweight='bold')
    ax3.set_title('Pixel Count per Class (Log Scale)', fontsize=14, fontweight='bold')
    ax3.set_xticks(classes)
    ax3.set_xticklabels([f'{i}\n{CLASS_LABELS[i][:4]}' for i in classes], fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if count > 0:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count/1e6:.2f}M' if count > 1e6 else f'{count/1e3:.1f}K',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 4. Bar Chart - Percentages (Linear Scale)
    ax4 = fig.add_subplot(gs[1, :])
    bars2 = ax4.bar(classes, percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Percentage of Total Pixels (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Class Distribution - Percentage of Total Pixels', fontsize=14, fontweight='bold')
    ax4.set_xticks(classes)
    ax4.set_xticklabels([f'{i}\n{CLASS_LABELS[i]}' for i in classes], fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for bar, pct in zip(bars2, percentages):
        if pct > 0.1:  # Only show label if > 0.1%
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.2f}%', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
     
    # Overall title
    fig.suptitle('RSNA Vertebrae Segmentation - Class Distribution Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = output_dir / 'class_distribution_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved comprehensive visualization: {output_path}")
    plt.close()
    
    # Create a separate figure for imbalance ratio
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Imbalance ratio visualization
    ratio = background_count / vertebrae_count if vertebrae_count > 0 else 0
    ax1.barh(['Background : Vertebrae'], [ratio], color='#FF6B6B', alpha=0.8, height=0.5)
    ax1.set_xlabel('Ratio (Background / Vertebrae)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Class Imbalance Ratio\n{ratio:.1f}:1', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.text(ratio/2, 0, f'{ratio:.1f}x more background\nthan vertebrae pixels',
             ha='center', va='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Log scale comparison
    ax2.bar(['Background', 'Vertebrae (All)'], 
            [background_count, vertebrae_count],
            color=[CLASS_COLORS[0], '#FF6B6B'], alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_yscale('log')
    ax2.set_ylabel('Pixel Count (Log Scale)', fontsize=12, fontweight='bold')
    ax2.set_title('Background vs Vertebrae (Log Scale)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    ax2.text(0, background_count, f'{background_count/1e6:.2f}M',
             ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax2.text(1, vertebrae_count, f'{vertebrae_count/1e6:.2f}M',
             ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path2 = output_dir / 'class_imbalance_ratio.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved imbalance ratio visualization: {output_path2}")
    plt.close()


def save_statistics_csv(aggregated_counts, total_pixels, per_file_stats, output_dir):
    """Save detailed statistics to CSV files"""
    
    # Overall statistics
    stats_data = []
    for cls in range(NUM_CLASSES):
        count = aggregated_counts[cls]
        pct = (count / total_pixels * 100) if total_pixels > 0 else 0
        stats_data.append({
            'Class_ID': cls,
            'Class_Name': CLASS_LABELS[cls],
            'Pixel_Count': count,
            'Percentage': pct,
            'Is_Background': (cls == 0),
            'Is_Vertebrae': (1 <= cls <= 7),
            'Is_Other': (cls == 8)
        })
    
    df_overall = pd.DataFrame(stats_data)
    output_path = output_dir / 'class_distribution_statistics.csv'
    df_overall.to_csv(output_path, index=False, float_format='%.6f')
    print(f"✓ Saved overall statistics: {output_path}")
    
    # Per-file statistics
    if per_file_stats:
        df_per_file = pd.DataFrame(per_file_stats)
        output_path2 = output_dir / 'per_file_class_distribution.csv'
        df_per_file.to_csv(output_path2, index=False)
        print(f"✓ Saved per-file statistics: {output_path2}")
    
    # Summary statistics
    background_count = aggregated_counts[0]
    vertebrae_count = sum(aggregated_counts[i] for i in range(1, NUM_CLASSES))
    other_count = aggregated_counts[8]
    
    summary = {
        'Metric': [
            'Total_Pixels',
            'Background_Pixels',
            'Vertebrae_Pixels',
            'Other_Vertebrae_Pixels',
            'Background_Percentage',
            'Vertebrae_Percentage',
            'Other_Percentage',
            'Imbalance_Ratio',
            'Background_to_Vertebrae_Ratio'
        ],
        'Value': [
            total_pixels,
            background_count,
            vertebrae_count,
            other_count,
            (background_count / total_pixels * 100) if total_pixels > 0 else 0,
            (vertebrae_count / total_pixels * 100) if total_pixels > 0 else 0,
            (other_count / total_pixels * 100) if total_pixels > 0 else 0,
            (background_count / vertebrae_count) if vertebrae_count > 0 else 0,
            f'{background_count:,} : {vertebrae_count:,}'
        ]
    }
    
    df_summary = pd.DataFrame(summary)
    output_path3 = output_dir / 'class_distribution_summary.csv'
    df_summary.to_csv(output_path3, index=False)
    print(f"✓ Saved summary statistics: {output_path3}")
    
    return df_overall, df_summary


def print_summary_statistics(aggregated_counts, total_pixels):
    """Print summary statistics to console"""
    print("\n" + "="*80)
    print("CLASS DISTRIBUTION SUMMARY")
    print("="*80)
    
    background_count = aggregated_counts[0]
    vertebrae_count = sum(aggregated_counts[i] for i in range(1, NUM_CLASSES))
    other_count = aggregated_counts[8]
    
    background_pct = (background_count / total_pixels * 100) if total_pixels > 0 else 0
    vertebrae_pct = (vertebrae_count / total_pixels * 100) if total_pixels > 0 else 0
    other_pct = (other_count / total_pixels * 100) if total_pixels > 0 else 0
    
    print(f"\nTotal Pixels Analyzed: {total_pixels:,}")
    print(f"\nClass Distribution:")
    print(f"  Background (Class 0):     {background_count:>15,} pixels ({background_pct:>6.4f}%)")
    print(f"  Vertebrae (Classes 1-7):  {vertebrae_count:>15,} pixels ({vertebrae_pct:>6.4f}%)")
    print(f"  Other Vertebrae (Class 8): {other_count:>14,} pixels ({other_pct:>6.4f}%)")
    
    print(f"\nPer-Class Breakdown:")
    for cls in range(NUM_CLASSES):
        count = aggregated_counts[cls]
        pct = (count / total_pixels * 100) if total_pixels > 0 else 0
        print(f"  {CLASS_LABELS[cls]:20s} (Class {cls}): {count:>12,} pixels ({pct:>6.4f}%)")
    
    ratio = (background_count / vertebrae_count) if vertebrae_count > 0 else 0
    print(f"\nImbalance Ratio:")
    print(f"  Background : Vertebrae = {ratio:.2f} : 1")
    print(f"  There are {ratio:.1f}x more background pixels than vertebrae pixels")
    
    print("\n" + "="*80)


def main():
    """Main analysis pipeline"""
    print("="*80)
    print("RSNA VERTEBRAE SEGMENTATION - CLASS DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Check if segmentation path exists
    if not os.path.exists(SEGMENTATION_PATH):
        print(f"ERROR: Segmentation path not found: {SEGMENTATION_PATH}")
        print("Please update DATA_PATH in the script to point to your data directory.")
        return
    
    # Load segmentation files
    segmentation_files = load_segmentation_files(SEGMENTATION_PATH)
    
    if len(segmentation_files) == 0:
        print("ERROR: No segmentation files found!")
        return
    
    # Analyze all segmentations
    # For faster analysis, you can limit the number of files
    # Set max_files=None to process all files
    aggregated_counts, per_file_stats, total_pixels = analyze_all_segmentations(
        segmentation_files, max_files=None  # Change to a number to limit files
    )
    
    # Print summary
    print_summary_statistics(aggregated_counts, total_pixels)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_class_distribution_plots(aggregated_counts, total_pixels, OUTPUT_DIR)
    
    # Save statistics
    print("\nSaving statistics...")
    save_statistics_csv(aggregated_counts, total_pixels, per_file_stats, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print("  - class_distribution_analysis.png")
    print("  - class_imbalance_ratio.png")
    print("  - class_distribution_statistics.csv")
    print("  - class_distribution_summary.csv")
    if per_file_stats:
        print("  - per_file_class_distribution.csv")
    print("="*80)


if __name__ == "__main__":
    main()

