from cellpose import models, io, plot
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
import time

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


def process_with_cellpose_gpu(image_path, output_dir, model_type='nuclei', use_gpu=True, 
                               diameter=None, flow_threshold=0.4, cellprob_threshold=0.0):
    """
    Use Cellpose with GPU acceleration for nuclei segmentation.
    
    Parameters:
    - image_path: path to input image
    - output_dir: directory to save results
    - model_type: 'nuclei', 'cyto', 'cyto2', 'cyto3'
    - use_gpu: whether to use GPU (automatically detected if True)
    - diameter: expected nucleus diameter in pixels (None = auto-detect)
    - flow_threshold: threshold for flow error (default 0.4)
    - cellprob_threshold: threshold for cell probability (default 0.0)
    """
    
    # Initialize model with GPU
    # gpu=True will automatically use GPU if available, fallback to CPU if not
    model = models.Cellpose(gpu=use_gpu, model_type=model_type)
    
    # Check if GPU is actually being used
    if use_gpu and torch.cuda.is_available():
        print(f"✓ Using GPU for {Path(image_path).name}")
    else:
        print(f"✗ Using CPU for {Path(image_path).name}")
    
    # Read image
    img = io.imread(str(image_path))
    
    # Start timing
    start_time = time.time()
    
    # Define channels
    # channels = [cytoplasm, nucleus]
    # For nuclei in blue channel: [0, 2] means no cytoplasm, nucleus in channel 2
    # For nuclei detection only: [0, 0] uses grayscale
    channels = [0, 2]  # Use blue channel (index 2) for nuclei
    
    # Run segmentation with GPU
    masks, flows, styles, diams = model.eval(
        img, 
        diameter=diameter,
        channels=channels,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize=True,  # Normalize images
        batch_size=8  # Process multiple images at once if doing batch
    )
    
    # End timing
    elapsed_time = time.time() - start_time
    
    # Count nuclei
    nuclei_count = masks.max()
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image', fontsize=12)
    axes[0, 0].axis('off')
    
    # Blue channel
    axes[0, 1].imshow(img[:, :, 2], cmap='Blues')
    axes[0, 1].set_title('Blue Channel (Nuclei)', fontsize=12)
    axes[0, 1].axis('off')
    
    # Red channel
    axes[0, 2].imshow(img[:, :, 0], cmap='Reds')
    axes[0, 2].set_title('Red Channel (Membrane)', fontsize=12)
    axes[0, 2].axis('off')
    
    # Segmentation masks (colored by nucleus)
    axes[1, 0].imshow(masks, cmap='nipy_spectral')
    axes[1, 0].set_title(f'Segmented Nuclei (n={nuclei_count})', fontsize=12)
    axes[1, 0].axis('off')
    
    # Flow field visualization
    axes[1, 1].imshow(flows[0][0], cmap='RdBu_r')
    axes[1, 1].set_title('Flow Field (X)', fontsize=12)
    axes[1, 1].axis('off')
    
    # Overlay with outlines
    outlines = plot.masks_to_outlines(masks)
    overlay = img.copy()
    overlay[outlines > 0] = [0, 255, 255]  # Cyan outlines
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title(f'Overlay (Time: {elapsed_time:.2f}s)', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    base_name = Path(image_path).stem
    
    plt.savefig(output_dir / f'{base_name}_cellpose_gpu.png', dpi=300, bbox_inches='tight')
    np.save(output_dir / f'{base_name}_masks.npy', masks)
    np.save(output_dir / f'{base_name}_flows.npy', flows[0])
    
    plt.close()
    
    # Calculate statistics
    stats = calculate_nuclei_stats(masks, img)
    stats['processing_time_sec'] = elapsed_time
    stats['gpu_used'] = use_gpu and torch.cuda.is_available()
    
    return nuclei_count, masks, stats, elapsed_time


def calculate_nuclei_stats(masks, img):
    """Calculate properties of each nucleus."""
    from skimage.measure import regionprops
    
    props = regionprops(masks, intensity_image=img[:, :, 2])  # Blue channel
    
    stats = []
    for prop in props:
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


def batch_process_cellpose_gpu(input_dir, output_dir, model_type='nuclei', 
                                use_gpu=True, diameter=None):
    """Batch process all images with GPU acceleration."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Find all images
    image_files = list(input_path.glob('*.jpeg')) + \
                  list(input_path.glob('*.jpg')) + \
                  list(input_path.glob('*.png')) + \
                  list(input_path.glob('*.tif'))
    
    print(f"Found {len(image_files)} images to process")
    print(f"Using GPU: {use_gpu and torch.cuda.is_available()}\n")
    
    all_results = []
    total_nuclei = 0
    total_time = 0
    
    # Process with progress bar
    for img_file in tqdm(image_files, desc="Processing images"):
        try:
            count, masks, stats, proc_time = process_with_cellpose_gpu(
                img_file, output_path, model_type, use_gpu, diameter
            )
            
            stats['image'] = img_file.name
            all_results.append(stats)
            total_nuclei += count
            total_time += proc_time
            
            print(f"  {img_file.name}: {count} nuclei in {proc_time:.2f}s")
            
        except Exception as e:
            print(f"  Error processing {img_file.name}: {e}")
    
    # Combine all statistics
    if all_results:
        combined_stats = pd.concat(all_results, ignore_index=True)
        combined_stats.to_csv(output_path / 'all_nuclei_statistics.csv', index=False)
        
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
        
        # Overall summary
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total images processed: {len(image_files)}")
        print(f"Total nuclei detected: {total_nuclei}")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Average time per image: {total_time/len(image_files):.2f}s")
        print(f"GPU used: {use_gpu and torch.cuda.is_available()}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*60}\n")
        
        return combined_stats
    else:
        print("No results to save!")
        return None

# Run the batch processing
if __name__ == "__main__":
    input_directory = "crop_1/crop"
    output_directory = "crop_1/cellpose_gpu_results"
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        use_gpu = True
    else:
        print("✗ No GPU detected, will use CPU")
        use_gpu = False
    
    # Process all images
    results = batch_process_cellpose_gpu(
        input_directory, 
        output_directory, 
        model_type='nuclei',  # or 'cyto', 'cyto2' for cell segmentation
        use_gpu=use_gpu,
        diameter=None  # Auto-detect, or specify like 30 for ~30 pixel diameter
    )