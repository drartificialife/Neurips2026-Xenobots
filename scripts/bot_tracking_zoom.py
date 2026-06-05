#!/usr/bin/env python3
"""
Bot tracking and region-based zoom for VLM fitness evaluation.

This script detects bots in pre-intervention images and tracks their positions
to create accurate before/after comparisons for VLM evaluation.

Usage:
    python scripts/bot_tracking_zoom.py --batch batch-000070 --output-dir bot_tracking_test
"""

import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure
from scipy.spatial.distance import cdist

# Import our archive functions
from parse_interventions import load_batch_with_interventions

def detect_bots_in_image(image_path: Path, min_area: int = 10, max_area: int = 200) -> List[Dict[str, Any]]:
    """Detect dark circular bots in microscopy image using blob detection."""
    try:
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            return []

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold to find dark objects (bots)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find connected components
        labels = measure.label(cleaned)
        regions = measure.regionprops(labels)

        bots = []
        for region in regions:
            # Filter by area and circularity
            area = region.area
            if min_area <= area <= max_area:
                # Calculate circularity
                perimeter = region.perimeter
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                if circularity > 0.6:  # Somewhat circular
                    centroid = region.centroid
                    bbox = region.bbox

                    bots.append({
                        'centroid': (int(centroid[1]), int(centroid[0])),  # (x, y)
                        'bbox': bbox,  # (min_row, min_col, max_row, max_col)
                        'area': area,
                        'circularity': circularity,
                        'region': region
                    })

        return bots

    except Exception as e:
        print(f"Warning: Failed to detect bots in {image_path}: {e}")
        return []

def match_bots_across_frames(pre_bots: List[Dict], post_bots: List[Dict], max_distance: int = 50) -> List[Tuple[Dict, Dict]]:
    """Match bots between pre and post frames based on proximity."""
    if not pre_bots or not post_bots:
        return []

    # Extract centroids
    pre_centroids = np.array([bot['centroid'] for bot in pre_bots])
    post_centroids = np.array([bot['centroid'] for bot in post_bots])

    # Calculate distance matrix
    distances = cdist(pre_centroids, post_centroids)

    # Find matches within max_distance
    matches = []
    used_post = set()

    for i, pre_bot in enumerate(pre_bots):
        # Find closest post bot
        min_dist_idx = np.argmin(distances[i])
        min_dist = distances[i, min_dist_idx]

        if min_dist <= max_distance and min_dist_idx not in used_post:
            matches.append((pre_bot, post_bots[min_dist_idx]))
            used_post.add(min_dist_idx)

    return matches

def create_bot_focused_zoom(image_path: Path, bot_centroid: Tuple[int, int],
                           zoom_size: int = 128, target_size: Tuple[int, int] = (256, 256)) -> Image.Image:
    """Create a zoomed region centered on a specific bot."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Calculate zoom region around bot centroid
        x, y = bot_centroid
        half_zoom = zoom_size // 2

        left = max(0, x - half_zoom)
        top = max(0, y - half_zoom)
        right = min(img_pil.width, x + half_zoom)
        bottom = min(img_pil.height, y + half_zoom)

        # Crop the region
        cropped = img_pil.crop((left, top, right, bottom))

        # Resize to target size
        cropped = cropped.resize(target_size, Image.Resampling.LANCZOS)

        return cropped

    except Exception as e:
        print(f"Warning: Failed to create bot-focused zoom: {e}")
        return None

def visualize_bot_tracking(batch_id: str, output_dir: Path, num_samples: int = 3):
    """Visualize bot detection and tracking across pre/post frames."""
    print(f"Visualizing bot tracking for batch: {batch_id}")

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

    pre_paths = [Path(img) for img in record['pre_images']]
    post_paths = [Path(img) for img in record['post_images']]

    if len(pre_paths) < num_samples or len(post_paths) < num_samples:
        num_samples = min(len(pre_paths), len(post_paths), num_samples)

    print(f"Processing {num_samples} frame pairs...")

    all_matches = []

    for i in range(num_samples):
        pre_path = pre_paths[i]
        post_path = post_paths[i]

        print(f"\nFrame pair {i+1}: {pre_path.name} -> {post_path.name}")

        # Detect bots in both frames
        pre_bots = detect_bots_in_image(pre_path)
        post_bots = detect_bots_in_image(post_path)

        print(f"  Pre: {len(pre_bots)} bots detected")
        print(f"  Post: {len(post_bots)} bots detected")

        # Match bots across frames
        matches = match_bots_across_frames(pre_bots, post_bots)
        print(f"  Matched: {len(matches)} bot pairs")

        all_matches.extend(matches)

        # Create visualizations for this frame pair
        if matches:
            fig, axes = plt.subplots(2, len(matches), figsize=(4*len(matches), 8))
            if len(matches) == 1:
                axes = axes.reshape(2, -1)

            fig.suptitle(f'Bot Tracking - Frame {i+1} (Intervention: {record["intervention"][0]}ms)')

            for j, (pre_bot, post_bot) in enumerate(matches):
                # Pre-intervention zoom
                pre_zoom = create_bot_focused_zoom(pre_path, pre_bot['centroid'])
                if pre_zoom:
                    axes[0, j].imshow(pre_zoom)
                    axes[0, j].set_title(f'Bot {j+1} Pre\nArea: {pre_bot["area"]}')
                    axes[0, j].axis('off')

                # Post-intervention zoom
                post_zoom = create_bot_focused_zoom(post_path, post_bot['centroid'])
                if post_zoom:
                    axes[1, j].imshow(post_zoom)
                    axes[1, j].set_title(f'Bot {j+1} Post\nArea: {post_bot["area"]}')
                    axes[1, j].axis('off')

            plt.tight_layout()
            pair_output = output_dir / f"bot_tracking_frame_{i+1}.png"
            plt.savefig(pair_output, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  Saved: {pair_output}")

    # Summary statistics
    total_bots_pre = len(set(match[0]['centroid'] for match in all_matches))
    total_bots_post = len(set(match[1]['centroid'] for match in all_matches))

    print("\n📊 Summary:")
    print(f"  Total tracked bots: {len(all_matches)}")
    print(f"  Unique pre bots: {total_bots_pre}")
    print(f"  Unique post bots: {total_bots_post}")

    # Save metadata
    metadata = {
        "batch_id": batch_id,
        "intervention_ms": record['intervention'][0],
        "frames_analyzed": num_samples,
        "total_bot_matches": len(all_matches),
        "unique_pre_bots": total_bots_pre,
        "unique_post_bots": total_bots_post,
        "output_directory": str(output_dir)
    }

    metadata_path = output_dir / "bot_tracking_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Bot tracking complete! Check: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True, help='Batch ID to analyze')
    parser.add_argument('--output-dir', type=Path, default=Path('bot_tracking_visualization'),
                       help='Output directory for visualizations')
    parser.add_argument('--samples', type=int, default=3, help='Number of frame pairs to analyze')

    args = parser.parse_args()

    visualize_bot_tracking(args.batch, args.output_dir, args.samples)

if __name__ == '__main__':
    main()