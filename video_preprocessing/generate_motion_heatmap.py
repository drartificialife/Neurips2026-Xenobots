#!/usr/bin/env python3
"""
Generate motion heatmap from archive frames.

Creates a visual comparison of motion between pre and post-intervention periods
using motion intensity heatmaps with different colors.

Usage:
    python generate_motion_heatmap.py --batch batch-000148 --output output.png
"""

import argparse
import json
import re
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from typing import Tuple, List

ARCHIVE_ROOT = Path(r'D:\xenobot_videos')
OUTPUT_DIR = Path('video_preprocessing')


def get_intervention_times(batch_id: str) -> Tuple[str, str]:
    """Get intervention start and end times from batch commands."""
    batch_dir = ARCHIVE_ROOT / batch_id
    commands_dir = batch_dir / 'commands'
    
    electrical_commands = list(commands_dir.glob('electrical_*.json'))
    if not electrical_commands:
        raise FileNotFoundError(f"No electrical commands found in {commands_dir}")
    
    with open(electrical_commands[0]) as f:
        command = json.load(f)
    
    return command.get('start_time'), command.get('end_time')


def get_batch_frames_split(batch_id: str, start_time: str, end_time: str) -> Tuple[List[Path], List[Path]]:
    """Get frame images split by intervention timing."""
    batch_dir = ARCHIVE_ROOT / batch_id
    images_dir = batch_dir / 'images'
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    all_frames = sorted(images_dir.glob('*.jpg'))
    if not all_frames:
        raise FileNotFoundError(f"No frame images found in {images_dir}")
    
    # Parse intervention times
    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00')).replace(tzinfo=None)
    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00')).replace(tzinfo=None)
    
    pre_frames = []
    post_frames = []
    
    timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}T\d{2}\.\d{2}\.\d{2})')
    
    for frame_path in all_frames:
        filename = frame_path.stem
        match = timestamp_pattern.search(filename)
        if not match:
            continue
        
        timestamp_str = match.group(1)
        parts = timestamp_str.split('T')
        time_fixed = parts[1].replace('.', ':')
        frame_time_str = f"{parts[0]}T{time_fixed}"
        
        try:
            frame_dt = datetime.fromisoformat(frame_time_str)
            
            if frame_dt < start_dt:
                pre_frames.append(frame_path)
            elif frame_dt > end_dt:
                post_frames.append(frame_path)
        except ValueError:
            continue
    
    return pre_frames, post_frames


def create_motion_heatmap(batch_id: str, output_path: Path = None) -> Path:
    """
    Create motion heatmap from pre and post-intervention frames.
    
    Args:
        batch_id: Batch directory name
        output_path: Where to save heatmap (default: video_preprocessing/{batch_id}_heatmap.png)
    
    Returns:
        Path to saved heatmap
    """
    if output_path is None:
        output_path = OUTPUT_DIR / f"{batch_id}_heatmap.png"
    
    print(f"\nGenerating motion heatmap for {batch_id}...")
    
    # Get intervention timing
    print("  Reading intervention timing...")
    start_time, end_time = get_intervention_times(batch_id)
    
    # Split frames
    print("  Splitting frames by intervention timing...")
    pre_frames, post_frames = get_batch_frames_split(batch_id, start_time, end_time)
    
    print(f"  Pre-intervention: {len(pre_frames)} frames")
    print(f"  Post-intervention: {len(post_frames)} frames")
    
    if len(pre_frames) == 0 or len(post_frames) == 0:
        raise ValueError(f"Not enough frames: pre={len(pre_frames)}, post={len(post_frames)}")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(pre_frames[0]))
    if first_frame is None:
        raise RuntimeError(f"Could not read frame: {pre_frames[0]}")
    
    height, width = first_frame.shape[:2]
    
    # Resize if too large
    max_width = 1920
    if width > max_width:
        scale = max_width / width
        width = int(width * scale)
        height = int(height * scale)
    
    print(f"  Frame size: {width}x{height}")
    
    # Create pre-intervention heatmap
    print("  Creating pre-intervention heatmap...")
    pre_heatmap = np.zeros((height, width), dtype=np.float32)
    
    for i, frame_path in enumerate(pre_frames):
        if (i + 1) % max(1, len(pre_frames) // 10) == 0:
            print(f"    Processing pre-frame {i+1}/{len(pre_frames)}")
        
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        
        # Resize if needed
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale for intensity
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        pre_heatmap += gray
    
    # Normalize
    pre_heatmap = pre_heatmap / len(pre_frames)
    pre_heatmap = (pre_heatmap * 255).astype(np.uint8)
    
    # Create post-intervention heatmap
    print("  Creating post-intervention heatmap...")
    post_heatmap = np.zeros((height, width), dtype=np.float32)
    
    for i, frame_path in enumerate(post_frames):
        if (i + 1) % max(1, len(post_frames) // 10) == 0:
            print(f"    Processing post-frame {i+1}/{len(post_frames)}")
        
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        
        # Resize if needed
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        post_heatmap += gray
    
    # Normalize
    post_heatmap = post_heatmap / len(post_frames)
    post_heatmap = (post_heatmap * 255).astype(np.uint8)
    
    # Apply colormaps
    print("  Applying colormaps...")
    pre_colored = cv2.applyColorMap(pre_heatmap, cv2.COLORMAP_OCEAN)    # Blue-ish
    post_colored = cv2.applyColorMap(post_heatmap, cv2.COLORMAP_HOT)    # Red-ish
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    color = (255, 255, 255)  # White text
    
    cv2.putText(pre_colored, "PRE-INTERVENTION", (20, 50), font, font_scale, color, thickness)
    cv2.putText(post_colored, "POST-INTERVENTION", (20, 50), font, font_scale, color, thickness)
    
    # Create side-by-side comparison
    combined = np.hstack([pre_colored, post_colored])
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), combined)
    
    print(f"  Saved to: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate motion heatmap from archive')
    parser.add_argument('--batch', type=str, required=True, help='Batch ID (batch-xxxxx)')
    parser.add_argument('--output', type=str, help='Output image path (default: video_preprocessing/{batch}_heatmap.png)')
    
    args = parser.parse_args()
    
    output_path = Path(args.output) if args.output else None
    create_motion_heatmap(args.batch, output_path)
    
    print(f"\n[SUCCESS] Heatmap created\n")


if __name__ == '__main__':
    main()
