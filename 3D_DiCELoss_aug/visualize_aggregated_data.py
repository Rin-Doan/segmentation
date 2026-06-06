"""
Visualize aggregated training data with segmentation overlays
Creates one image per study showing all slices with segmentation masks
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

from aggregate_data import aggregate_training_data

# Class names and colors for visualization
NUM_CLASSES = 9  # Background + C1-C7 + other vertebrae
CLASS_NAMES = ['Background', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'Other']
CLASS_COLORS = [
    '#000000',  # Background - black (transparent)
    '#FF0000',  # C1 - red
    '#00FF00',  # C2 - green
    '#0000FF',  # C3 - blue
    '#FFFF00',  # C4 - yellow
    '#FF00FF',  # C5 - magenta
    '#00FFFF',  # C6 - cyan
    '#FFA500',  # C7 - orange
    '#800080',  # Other - purple
]


def create_colormap():
    """Create a colormap for segmentation visualization"""
    colors = np.zeros((256, 4))  # RGBA
    colors[0] = [0, 0, 0, 0]  # Background - transparent
    
    for i in range(1, NUM_CLASSES):
        # Convert hex to RGB
        hex_color = CLASS_COLORS[i].lstrip('#')
        rgb = tuple(int(hex_color[j:j+2], 16) / 255.0 for j in (0, 2, 4))
        colors[i] = [rgb[0], rgb[1], rgb[2], 0.6]  # 60% opacity
    
    return ListedColormap(colors[:NUM_CLASSES])


def visualize_study_all_slices(image_volume, seg_volume, study_id, save_path=None, 
                                max_slices_per_row=10, slice_spacing=2):
    """
    Visualize all slices of a study with segmentation overlay
    
    Args:
        image_volume: 3D numpy array (D, H, W) - image data
        seg_volume: 3D numpy array (D, H, W) - segmentation labels
        study_id: Study identifier for title
        save_path: Path to save visualization (if None, uses study_id)
        max_slices_per_row: Maximum number of slices to display per row
        slice_spacing: Spacing between slices to display (1 = all slices, 2 = every other slice, etc.)
    """
    D, H, W = image_volume.shape
    print('seg_volume.shape: ', seg_volume.shape)
    
    # Select slices to display
    selected_slices = list(range(0, D, slice_spacing))
    num_slices = len(selected_slices)
    
    # Calculate grid dimensions
    num_rows = int(np.ceil(num_slices / max_slices_per_row))
    num_cols = min(num_slices, max_slices_per_row)
    
    # Create figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    fig.suptitle(f'Study: {study_id}\nAll Slices with Segmentation Overlay ({num_slices} slices shown)',
                 fontsize=14, fontweight='bold')
    
    # Flatten axes for easier indexing
    if num_rows == 1:
        axes = axes.reshape(1, -1) if num_cols > 1 else [axes]
    axes_flat = axes.flatten() if num_rows > 1 else axes
    
    # Create colormap
    cmap = create_colormap()
    
    # Normalize image for display
    img_min, img_max = np.percentile(image_volume, [1, 99])
    image_volume_norm = np.clip((image_volume - img_min) / (img_max - img_min), 0, 1)
    
    # Display each selected slice
    for idx, slice_idx in enumerate(selected_slices):
        if idx >= len(axes_flat):
            break
        
        ax = axes_flat[idx]
        ax.axis('off')
        
        # Display image
        ax.imshow(image_volume_norm[slice_idx], cmap='gray', aspect='auto')
        
        # Overlay segmentation
        seg_slice = seg_volume[slice_idx]
        mask = seg_slice > 0  # Only show non-background pixels
        
        if np.any(mask):
            # Create colored mask
            seg_colored = np.zeros((H, W, 4))  # RGBA
            for class_id in range(1, NUM_CLASSES):
                class_mask = seg_slice == class_id
                if np.any(class_mask):
                    hex_color = CLASS_COLORS[class_id].lstrip('#')
                    rgb = tuple(int(hex_color[j:j+2], 16) / 255.0 for j in (0, 2, 4))
                    seg_colored[class_mask] = [rgb[0], rgb[1], rgb[2], 0.6]
            
            ax.imshow(seg_colored, interpolation='nearest', aspect='auto')
            
            # Count unique vertebrae in this slice
            unique_labels = np.unique(seg_slice[seg_slice > 0])
            num_vertebrae = len(unique_labels)
            ax.set_title(f'Slice {slice_idx}\n({num_vertebrae} vertebrae)', 
                        fontsize=8, pad=2)
        else:
            ax.set_title(f'Slice {slice_idx}\n(No vertebrae)', fontsize=8, pad=2)
    
    # Hide unused subplots
    for idx in range(len(selected_slices), len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    if save_path is None:
        save_path = f'{study_id}_all_slices_visualization.png'
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved visualization to: {save_path}")


def visualize_all_studies(output_dir='visualizations', max_slices_per_row=10, 
                          slice_spacing=2, max_studies=None):
    """
    Visualize all aggregated studies
    
    Args:
        output_dir: Directory to save visualizations
        max_slices_per_row: Maximum slices per row in visualization
        slice_spacing: Spacing between slices (1=all, 2=every other, etc.)
        max_studies: Maximum number of studies to visualize (None = all)
    """
    print("="*60)
    print("Visualizing Aggregated Training Data")
    print("="*60)
    
    # Aggregate training data
    print("\nLoading and processing data...")
    image_volumes, seg_volumes, study_ids, report_dict = aggregate_training_data(verbose=True)


    
    if len(image_volumes) == 0:
        print("Error: No data to visualize!")
        return
    
    # Limit number of studies if specified
    if max_studies is not None:
        image_volumes = image_volumes[:max_studies]
        seg_volumes = seg_volumes[:max_studies]
        study_ids = study_ids[:max_studies]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating visualizations for {len(study_ids)} studies")
    print(f"{'='*60}")
    
    # Visualize each study
    for idx, (image_vol, seg_vol, study_id) in enumerate(zip(image_volumes, seg_volumes, study_ids), 1):
        print(f"\n[{idx}/{len(study_ids)}] Visualizing: {study_id}")
        print(f"  Volume shape: {image_vol.shape}")
        
        save_path = os.path.join(output_dir, f'{study_id}_all_slices_visualization.png')
        visualize_study_all_slices(
            image_vol, seg_vol, study_id, 
            save_path=save_path,
            max_slices_per_row=max_slices_per_row,
            slice_spacing=slice_spacing
        )
    
    print(f"\n{'='*60}")
    print(f"✓ Visualization complete!")
    print(f"  Saved {len(study_ids)} visualizations to: {output_dir}")
    print(f"{'='*60}")


def visualize_single_study(study_id, output_dir='visualizations', 
                           max_slices_per_row=10, slice_spacing=2):
    """
    Visualize a single study
    
    Args:
        study_id: Study ID to visualize
        output_dir: Directory to save visualization
        max_slices_per_row: Maximum slices per row
        slice_spacing: Spacing between slices
    """
    print(f"Visualizing study: {study_id}")
    
    # Aggregate training data
    image_volumes, seg_volumes, study_ids, _ = aggregate_training_data(verbose=False)
    
    # Find the study
    try:
        idx = study_ids.index(study_id)
        image_vol = image_volumes[idx]
        seg_vol = seg_volumes[idx]
    except ValueError:
        print(f"Error: Study {study_id} not found in aggregated data")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, f'{study_id}_all_slices_visualization.png')
    visualize_study_all_slices(
        image_vol, seg_vol, study_id,
        save_path=save_path,
        max_slices_per_row=max_slices_per_row,
        slice_spacing=slice_spacing
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize aggregated training data')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--max_slices_per_row', type=int, default=10,
                       help='Maximum slices per row in visualization')
    parser.add_argument('--slice_spacing', type=int, default=2,
                       help='Spacing between slices (1=all, 2=every other, etc.)')
    parser.add_argument('--max_studies', type=int, default=None,
                       help='Maximum number of studies to visualize (None = all)')
    parser.add_argument('--study_id', type=str, default=None,
                       help='Visualize a single study by ID')
    
    args = parser.parse_args()
    
    if args.study_id:
        visualize_single_study(
            args.study_id,
            output_dir=args.output_dir,
            max_slices_per_row=args.max_slices_per_row,
            slice_spacing=args.slice_spacing
        )
    else:
        visualize_all_studies(
            output_dir=args.output_dir,
            max_slices_per_row=args.max_slices_per_row,
            slice_spacing=args.slice_spacing,
            max_studies=args.max_studies
        )

