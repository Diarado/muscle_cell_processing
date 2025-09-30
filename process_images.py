from cellpose import models, io, core
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
import time
from skimage import segmentation

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


def masks_to_outlines(masks):
    """Convert masks to outlines (replacement for plot.masks_to_outlines)."""
    outlines = segmentation.find_boundaries(masks, mode='inner')
    return outlines


def process_with_cellpose_gpu(image_path, output_dir, use_gpu=True, 
                               diameter=None, flow_threshold=0.4, cellprob_threshold=0.0,
                               skip_existing=True):
    """
    Use Cellpose v4 with GPU acceleration for nuclei segmentation.
    """
    
    # Check if results already exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    base_name = Path(image_path).stem
    masks_file = output_dir / f'{base_name}_masks.npy'
    
    if skip_existing and masks_file.exists():
        print(f"  ✓ Skipping {Path(image_path).name} (already processed)", flush=True)
        # Load existing results
        masks = np.load(masks_file)
        nuclei_count = int(masks.max())
        return nuclei_count, masks, None, 0.0  # Return None for stats, 0 for time
    
    # Initialize model with GPU
    use_gpu = use_gpu and core.use_gpu()
    
    # For Cellpose v4
    model = models.CellposeModel(gpu=use_gpu)
    
    # Read image
    img = io.imread(str(image_path))
    
    # Extract blue channel for nuclei
    if img.ndim == 3 and img.shape[2] >= 3:
        img_blue = img[:, :, 2]  # Blue channel only
    else:
        img_blue = img
    
    print(f"  Processing {Path(image_path).name} (shape: {img_blue.shape})...", flush=True)
    
    # Start timing
    start_time = time.time()
    
    # Run segmentation with GPU
    print(f"  Running segmentation...", flush=True)
    masks, flows, styles = model.eval(
        img_blue,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize=True
    )
    
    # End timing
    elapsed_time = time.time() - start_time
    
    # Count nuclei
    nuclei_count = int(masks.max())
    print(f"  Found {nuclei_count} nuclei in {elapsed_time:.2f}s", flush=True)
    
    # Create visualization
    print(f"  Creating visualization...", flush=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image', fontsize=12)
    axes[0, 0].axis('off')
    
    # Blue channel
    axes[0, 1].imshow(img_blue, cmap='Blues')
    axes[0, 1].set_title('Blue Channel (Nuclei)', fontsize=12)
    axes[0, 1].axis('off')
    
    # Red channel
    if img.ndim == 3:
        axes[0, 2].imshow(img[:, :, 0], cmap='Reds')
        axes[0, 2].set_title('Red Channel (Membrane)', fontsize=12)
    axes[0, 2].axis('off')
    
    # Segmentation masks (colored by nucleus)
    axes[1, 0].imshow(masks, cmap='nipy_spectral', vmax=min(masks.max(), 1000))
    axes[1, 0].set_title(f'Segmented Nuclei (n={nuclei_count})', fontsize=12)
    axes[1, 0].axis('off')
    
    # Flow field visualization
    if len(flows) > 0 and flows[0] is not None:
        axes[1, 1].imshow(flows[0], cmap='RdBu_r')
        axes[1, 1].set_title('Flow Field', fontsize=12)
        axes[1, 1].axis('off')
    else:
        axes[1, 1].axis('off')
    
    # Overlay with outlines
    print(f"  Computing outlines...", flush=True)
    outlines = masks_to_outlines(masks)
    overlay = img.copy() if img.ndim == 3 else np.stack([img_blue]*3, axis=-1)
    overlay[outlines] = [0, 255, 255]  # Cyan outlines
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title(f'Overlay (Time: {elapsed_time:.2f}s)', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save results
    print(f"  Saving results...", flush=True)
    plt.savefig(output_dir / f'{base_name}_cellpose_gpu.png', dpi=300, bbox_inches='tight')
    
    # Save masks
    if masks.max() > 65535:
        np.save(output_dir / f'{base_name}_masks.npy', masks.astype(np.uint32))
    else:
        np.save(output_dir / f'{base_name}_masks.npy', masks.astype(np.uint16))
    
    if len(flows) > 0 and flows[0] is not None:
        np.save(output_dir / f'{base_name}_flows.npy', flows[0])
    
    plt.close()
    
    # Calculate statistics
    print(f"  Calculating statistics...", flush=True)
    stats = calculate_nuclei_stats(masks, img_blue)
    stats['processing_time_sec'] = elapsed_time
    stats['gpu_used'] = use_gpu
    
    print(f"  ✓ Completed {Path(image_path).name}\n", flush=True)
    
    return nuclei_count, masks, stats, elapsed_time


def calculate_nuclei_stats(masks, img):
    """Calculate properties of each nucleus."""
    from skimage.measure import regionprops
    
    # For very large number of masks, sample
    max_labels = 5000  # Process at most 5000 nuclei for stats
    
    if masks.max() > max_labels:
        print(f"    {masks.max()} nuclei detected. Computing stats for first {max_labels}...", flush=True)
        masks_sample = masks.copy()
        masks_sample[masks_sample > max_labels] = 0
        props = regionprops(masks_sample, intensity_image=img)
    else:
        props = regionprops(masks, intensity_image=img)
    
    stats = []
    for i, prop in enumerate(props):
        if i % 1000 == 0 and i > 0:
            print(f"    Processing nucleus {i}/{len(props)}...", flush=True)
        
        stats.append({
            'label': prop.label,
            'area': prop.area,
            'centroid_x': prop.centroid[1],
            'centroid_y': prop.centroid[0],
            'mean_intensity': prop.mean_intensity,
            'max_intensity': prop.max_intensity,
            'min_intensity': prop.min_intensity,
            'perimeter': prop.perimeter,
            'eccentricity': prop.eccentricity,
            'solidity': prop.solidity,
            'major_axis_length': prop.major_axis_length,
            'minor_axis_length': prop.minor_axis_length
        })
    
    return pd.DataFrame(stats)


def batch_process_cellpose_gpu(input_dir, output_dir, use_gpu=True, diameter=None, skip_existing=True):
    """Batch process all images with GPU acceleration."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Find all images
    image_files = list(input_path.glob('*.jpeg')) + \
                  list(input_path.glob('*.jpg')) + \
                  list(input_path.glob('*.png')) + \
                  list(input_path.glob('*.tif'))
    
    print(f"\nFound {len(image_files)} images to process")
    print(f"Using GPU: {use_gpu and core.use_gpu()}")
    print(f"Skip existing: {skip_existing}")
    print(f"Output directory: {output_path}\n")
    print("="*60)
    
    all_results = []
    total_nuclei = 0
    total_time = 0
    successful_images = 0
    skipped_images = 0
    
    # Process each image
    for i, img_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {img_file.name}")
        print("-"*60)
        
        try:
            count, masks, stats, proc_time = process_with_cellpose_gpu(
                img_file, output_path, use_gpu, diameter, skip_existing=skip_existing
            )
            
            # Only append stats if we actually processed (stats is not None)
            if stats is not None:
                stats['image'] = img_file.name
                all_results.append(stats)
                total_time += proc_time
                successful_images += 1
                print(f"✓ Success: {count} nuclei, {proc_time:.2f}s")
            else:
                skipped_images += 1
            
            total_nuclei += count
            
        except Exception as e:
            print(f"✗ Error processing {img_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Combine all statistics
    if all_results:
        print("\n" + "="*60)
        print("Saving combined results...")
        
        combined_stats = pd.concat(all_results, ignore_index=True)
        combined_stats.to_csv(output_path / 'all_nuclei_statistics.csv', index=False)
        print(f"✓ Saved: all_nuclei_statistics.csv")
        
        # Summary statistics per image
        summary = combined_stats.groupby('image').agg({
            'label': 'count',
            'area': ['mean', 'std', 'min', 'max'],
            'mean_intensity': ['mean', 'std'],
            'eccentricity': ['mean', 'std'],
            'processing_time_sec': 'first',
            'gpu_used': 'first'
        }).round(2)
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns]
        summary.to_csv(output_path / 'summary_statistics.csv')
        print(f"✓ Saved: summary_statistics.csv")
    
    # Overall summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total images found: {len(image_files)}")
    print(f"Images processed: {successful_images}")
    print(f"Images skipped: {skipped_images}")
    print(f"Total nuclei detected: {total_nuclei:,}")
    if successful_images > 0:
        print(f"Total processing time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"Average time per image: {total_time/successful_images:.2f}s")
    print(f"GPU used: {use_gpu and core.use_gpu()}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}\n")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return None


# Run the batch processing
if __name__ == "__main__":
    input_directory = "crop_1/crop"
    output_directory = "crop_1/cellpose_gpu_results"
    
    # Check GPU
    if core.use_gpu():
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        use_gpu = True
    else:
        print("✗ No GPU detected, will use CPU")
        use_gpu = False
    
    # Process all images
    results = batch_process_cellpose_gpu(
        input_directory, 
        output_directory, 
        use_gpu=use_gpu,
        diameter=None,  # Auto-detect, or specify like 30 for ~30 pixel diameter
        skip_existing=True  # Set to False to reprocess all images
    )