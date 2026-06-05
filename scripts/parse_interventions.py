#!/usr/bin/env python3
"""
Parse electrical interventions and classify images pre/post intervention.

Usage:
    python scripts/parse_interventions.py [--batch BATCH_ID] [--all]

Examples:
    # Parse specific batch
    python scripts/parse_interventions.py --batch batch-000070
    
    # Parse all downloaded batches
    python scripts/parse_interventions.py --all
"""

import json
import pickle
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple
import argparse

ARCHIVE_ROOT = Path('D:/xenobot_videos')
CACHE_ROOT = Path('D:/xenobot_videos/cache')
INTERVENTION_CACHE_FILE = CACHE_ROOT / 'intervention_metadata.json'


def parse_electrical_intervention(json_path: Path) -> Dict[str, Any]:
    """Parse electrical intervention JSON."""
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract key fields
    duration_ms = data['instructions'][0]['duration_ms']
    start_time = datetime.fromisoformat(data['start_time'].replace('Z', '+00:00'))
    end_time = datetime.fromisoformat(data['end_time'].replace('Z', '+00:00'))
    
    return {
        'command_id': data['command_id'],
        'duration_ms': duration_ms,
        'start_time': start_time,
        'end_time': end_time,
        'intervention_vector': [duration_ms / 1000.0],  # Convert to seconds, single value
        'current_ma': data['instructions'][0]['current_ma'],
        'angle_degrees': data['instructions'][0]['angle_degrees'],
        'frequency_hz': data['instructions'][0]['frequency_hz']
    }


def parse_image_timestamp(filename: str) -> datetime:
    """Extract timestamp from image filename.
    
    Format: UUID-YYYY-MM-DDTHH.MM.SS.jpg
    Example: 167b37b8-3d4e-4eb8-b2c7-c01d23306274-2025-10-07T17.35.13.jpg
    """
    import re
    
    # Use regex to find the timestamp pattern: 2025-10-07T17.35.13
    match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}\.\d{2}\.\d{2})', filename)
    
    if not match:
        raise ValueError(f"Could not find timestamp pattern in filename: {filename}")
    
    timestamp_str = match.group(1)
    # Convert dots to colons for proper ISO format: 17.35.13 -> 17:35:13
    timestamp_str = timestamp_str.replace('.', ':')
    # Parse as naive datetime and make it UTC aware
    dt = datetime.fromisoformat(timestamp_str)
    return dt.replace(tzinfo=timezone.utc)


def classify_images_pre_post(batch_dir: Path) -> Dict[str, Any]:
    """Classify images in batch as pre/post intervention."""
    
    images_dir = batch_dir / 'images'
    commands_dir = batch_dir / 'commands'
    
    if not images_dir.exists():
        return {'pre': [], 'post': [], 'control': True, 'images': []}
    
    # Get all images with timestamps
    images = []
    for img_path in images_dir.glob('*.jpg'):
        try:
            timestamp = parse_image_timestamp(img_path.name)
            images.append((timestamp, img_path))
        except ValueError as e:
            print(f"Warning: Could not parse timestamp from {img_path.name}: {e}")
    
    images.sort(key=lambda x: x[0])  # Sort by timestamp
    
    # Check for electrical interventions
    electrical_files = list(commands_dir.glob('electrical_*.json')) if commands_dir.exists() else []
    
    if not electrical_files:
        # Control batch: all images are natural dynamics
        return {
            'pre': [],
            'post': [],
            'control': True,
            'images': [str(img) for _, img in images],
            'num_images': len(images)
        }
    
    # Parse interventions
    interventions = []
    for json_file in electrical_files:
        try:
            intervention = parse_electrical_intervention(json_file)
            interventions.append(intervention)
        except Exception as e:
            print(f"Warning: Failed to parse {json_file}: {e}")
    
    if not interventions:
        return {
            'pre': [],
            'post': [],
            'control': True,
            'images': [str(img) for _, img in images],
            'num_images': len(images)
        }
    
    # For simplicity, use first intervention (can extend for multiple)
    intervention = interventions[0]
    start_time = intervention['start_time']
    end_time = intervention['end_time']
    
    pre_images = []
    post_images = []
    
    for timestamp, img_path in images:
        if timestamp < start_time:
            pre_images.append(str(img_path))
        elif timestamp > end_time:
            post_images.append(str(img_path))
    
    return {
        'pre': pre_images,
        'post': post_images,
        'control': False,
        'intervention': intervention,
        'num_pre': len(pre_images),
        'num_post': len(post_images)
    }


def load_batch_with_interventions(batch_id: str) -> List[Dict[str, Any]]:
    """Load batch metadata including intervention classification."""
    batch_dir = ARCHIVE_ROOT / batch_id
    
    if not batch_dir.exists():
        print(f"Warning: Batch directory {batch_dir} not found")
        return []
    
    # Classify images
    classification = classify_images_pre_post(batch_dir)
    
    # Create records for archive
    records = []
    
    if classification['control']:
        # Control batch: single record with all images
        records.append({
            'batch': batch_id,
            'type': 'control',
            'intervention': [0.0],  # No intervention
            'pre_images': classification['images'],
            'post_images': [],
            'prompt': 'natural_dynamics',
            'num_images': classification['num_images']
        })
    else:
        # Intervention batch
        intervention = classification['intervention']
        records.append({
            'batch': batch_id,
            'type': 'intervention',
            'intervention': intervention['intervention_vector'],
            'pre_images': classification['pre'],
            'post_images': classification['post'],
            'command_id': intervention['command_id'],
            'duration_ms': intervention['duration_ms'],
            'start_time': intervention['start_time'].isoformat(),
            'end_time': intervention['end_time'].isoformat(),
            'current_ma': intervention['current_ma'],
            'angle_degrees': intervention['angle_degrees'],
            'frequency_hz': intervention['frequency_hz'],
            'num_pre': classification['num_pre'],
            'num_post': classification['num_post']
        })
    
    return records


def load_all_intervention_metadata(force_reload: bool = False) -> List[Dict[str, Any]]:
    """Load intervention metadata from all downloaded batches."""
    CACHE_ROOT.mkdir(exist_ok=True, parents=True)
    
    # Try to load from cache first
    if not force_reload and INTERVENTION_CACHE_FILE.exists():
        with open(INTERVENTION_CACHE_FILE) as f:
            return json.load(f)
    
    all_metadata = []
    
    for batch_dir in ARCHIVE_ROOT.iterdir():
        if batch_dir.is_dir() and batch_dir.name.startswith('batch-'):
            print(f"Processing batch: {batch_dir.name}")
            batch_metadata = load_batch_with_interventions(batch_dir.name)
            all_metadata.extend(batch_metadata)
    
    # Cache metadata
    with open(INTERVENTION_CACHE_FILE, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    return all_metadata


def print_batch_summary(batch_id: str):
    """Print summary of batch classification."""
    records = load_batch_with_interventions(batch_id)
    
    if not records:
        print(f"No data found for batch {batch_id}")
        return
    
    record = records[0]
    
    print(f"\n{'='*60}")
    print(f"Batch: {batch_id}")
    print(f"Type: {record['type']}")
    
    if record['type'] == 'control':
        print(f"Images: {record['num_images']}")
        print("No electrical interventions (control batch)")
    else:
        print(f"Intervention: {record['intervention']} seconds")
        print(f"Duration: {record['duration_ms']} ms")
        print(f"Pre-intervention images: {record['num_pre']}")
        print(f"Post-intervention images: {record['num_post']}")
        print(f"Command ID: {record['command_id']}")
        print(f"Current: {record['current_ma']} mA")
        print(f"Angle: {record['angle_degrees']}°")
        print(f"Frequency: {record['frequency_hz']} Hz")
    
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--batch',
        type=str,
        help='Process specific batch ID (e.g., batch-000070)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all downloaded batches'
    )
    parser.add_argument(
        '--force-reload',
        action='store_true',
        help='Force reload metadata from disk (ignore cache)'
    )
    
    args = parser.parse_args()
    
    if args.batch:
        print_batch_summary(args.batch)
    elif args.all:
        print("Loading intervention metadata from all batches...")
        metadata = load_all_intervention_metadata(force_reload=args.force_reload)
        
        print(f"\n{'='*80}")
        print(f"Loaded {len(metadata)} intervention records")
        
        control_count = sum(1 for r in metadata if r['type'] == 'control')
        intervention_count = len(metadata) - control_count
        
        print(f"Control batches: {control_count}")
        print(f"Intervention batches: {intervention_count}")
        
        if intervention_count > 0:
            durations = [r['intervention'][0] for r in metadata if r['type'] == 'intervention']
            print(".1f")
            print(f"Duration range: {min(durations):.1f} - {max(durations):.1f} seconds")
        
        print(f"Metadata cached to: {INTERVENTION_CACHE_FILE}")
        print(f"{'='*80}")
    else:
        parser.print_help()


if __name__ == '__main__':
    import sys
    sys.exit(main())
