#!/usr/bin/env python3
"""
Visualize image preprocessing for VLM fitness evaluation.

This script generates and saves processed images to see what the VLM will receive:
- Zoomed/cropped individual images
- Timelapse collages
- Before/after comparisons

Usage:
    python scripts/visualize_preprocessing.py --batch batch-000070 --output-dir test_images
"""

import json
import argparse
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import our archive functions
from parse_interventions import load_batch_with_interventions

def save_image_comparison(pre_image: Image.Image, post_image: Image.Image, output_path: Path, title: str):
    """Save side-by-side comparison of pre/post images."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(pre_image)
    ax1.set_title("Pre-intervention")
    ax1.axis('off')

    ax2.imshow(post_image)
    ax2.set_title("Post-intervention")
    ax2.axis('off')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def preprocess_image_for_display(image_path: Path, zoom_factor: float = 2.0, target_size: Tuple[int, int] = (512, 512)) -> Image.Image:
    """Preprocess image for display: load, zoom/crop, resize."""
    try:
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        # Convert to PIL for easier processing
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Zoom in (crop center and resize)
        if zoom_factor > 1.0:
            width, height = img_pil.size
            crop_size = min(width, height) // zoom_factor
            left = (width - crop_size) // 2
            top = (height - crop_size) // 2
            right = left + crop_size
            bottom = top + crop_size
            img_pil = img_pil.crop((left, top, right, bottom))

        # Resize to target size
        img_pil = img_pil.resize(target_size, Image.Resampling.LANCZOS)

        return img_pil

    except Exception as e:
        print(f"Warning: Failed to preprocess {image_path}: {e}")
        return None

def create_timelapse_collage_display(image_paths: List[Path], grid_size: Tuple[int, int] = (3, 3), target_size: Tuple[int, int] = (512, 512)) -> Image.Image:
    """Create a collage of multiple images for display."""
    try:
        if not image_paths:
            return None

        # Sample images if we have more than grid slots
        max_images = grid_size[0] * grid_size[1]
        if len(image_paths) > max_images:
            image_paths = image_paths[:max_images]  # Take first N instead of random

        # Load and resize images
        images = []
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_pil = img_pil.resize((target_size[0] // grid_size[1], target_size[1] // grid_size[0]), Image.Resampling.LANCZOS)
                images.append(img_pil)

        if not images:
            return None

        # Create collage grid
        collage_width = target_size[0]
        collage_height = target_size[1]

        collage = Image.new('RGB', (collage_width, collage_height))

        thumb_width = collage_width // grid_size[1]
        thumb_height = collage_height // grid_size[0]

        for i, img in enumerate(images):
            if i >= grid_size[0] * grid_size[1]:
                break
            row = i // grid_size[1]
            col = i % grid_size[1]
            x = col * thumb_width
            y = row * thumb_height
            collage.paste(img, (x, y))

        return collage

    except Exception as e:
        print(f"Warning: Failed to create collage: {e}")
        return None

def visualize_batch_preprocessing(batch_id: str, output_dir: Path, num_samples: int = 3):
    """Generate and save visualization of preprocessing approaches for a batch."""
    print(f"Visualizing preprocessing for batch: {batch_id}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load batch data
    records = load_batch_with_interventions(batch_id)
    if not records:
        print(f"Error: No data found for batch {batch_id}")
        return

    record = records[0]

    if record['type'] == 'control':
        print(f"Batch {batch_id} is control (no intervention)")
        return

    # Get image paths
    pre_paths = [Path(img) for img in record['pre_images']]
    post_paths = [Path(img) for img in record['post_images']]

    if len(pre_paths) < num_samples or len(post_paths) < num_samples:
        num_samples = min(len(pre_paths), len(post_paths), num_samples)

    if num_samples == 0:
        print("No image pairs available")
        return

    print(f"Processing {num_samples} pre/post pairs from {len(pre_paths)}/{len(post_paths)} images")

    # Test different zoom factors
    zoom_factors = [1.0, 2.0, 3.0, 4.0]

    for zoom in zoom_factors:
        print(f"\n--- Testing zoom factor: {zoom}x ---")

        for i in range(num_samples):
            pre_sample = pre_paths[i]
            post_sample = post_paths[i]

            # Method 1: Individual zoomed images
            pre_zoomed = preprocess_image_for_display(pre_sample, zoom_factor=zoom)
            post_zoomed = preprocess_image_for_display(post_sample, zoom_factor=zoom)

            if pre_zoomed and post_zoomed:
                comparison_path = output_dir / f"zoom_{zoom}x_comparison_{i+1}.png"
                save_image_comparison(pre_zoomed, post_zoomed, comparison_path,
                                    f"Zoom {zoom}x - Sample {i+1} (Intervention: {record['intervention'][0]}ms)")
                print(f"  Saved: {comparison_path}")

    # Method 2: Timelapse collages
    print("\n--- Testing timelapse collages ---")

    collage_sizes = [(2, 2), (3, 3), (4, 4)]

    for rows, cols in collage_sizes:
        print(f"  Creating {rows}x{cols} collage...")

        # Use first N images for collage
        n_images = min(rows * cols, len(pre_paths), len(post_paths))

        pre_collage = create_timelapse_collage_display(pre_paths[:n_images], grid_size=(rows, cols))
        post_collage = create_timelapse_collage_display(post_paths[:n_images], grid_size=(rows, cols))

        if pre_collage and post_collage:
            collage_comparison_path = output_dir / f"collage_{rows}x{cols}_comparison.png"
            save_image_comparison(pre_collage, post_collage, collage_comparison_path,
                                f"{rows}x{cols} Timelapse Collage (Intervention: {record['intervention'][0]}ms)")
            print(f"  Saved: {collage_comparison_path}")

    # Save metadata
    metadata = {
        "batch_id": batch_id,
        "intervention_ms": record['intervention'][0],
        "pre_images_count": len(pre_paths),
        "post_images_count": len(post_paths),
        "samples_processed": num_samples,
        "zoom_factors_tested": zoom_factors,
        "collage_sizes_tested": collage_sizes,
        "output_directory": str(output_dir)
    }

    metadata_path = output_dir / "preprocessing_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Visualization complete! Check: {output_dir}")
    print(f"📊 Metadata saved: {metadata_path}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True, help='Batch ID to visualize')
    parser.add_argument('--output-dir', type=Path, default=Path('preprocessing_visualization'),
                       help='Output directory for images')
    parser.add_argument('--samples', type=int, default=3, help='Number of sample pairs to process')

    args = parser.parse_args()

    visualize_batch_preprocessing(args.batch, args.output_dir, args.samples)

if __name__ == '__main__':
    main()