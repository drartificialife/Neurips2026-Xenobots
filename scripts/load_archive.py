#!/usr/bin/env python3
"""
Load local archive metadata and build nearest-neighbor index for images with electrical interventions.

Usage:
    python scripts/load_archive.py [--test]

Examples:
    # Load archive and build index
    python scripts/load_archive.py
    
    # Load and test with sample interventions
    python scripts/load_archive.py --test
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Import from our scripts
from parse_interventions import load_batch_with_interventions

ARCHIVE_ROOT = Path('D:/xenobot_videos')
CACHE_ROOT = Path('D:/xenobot_videos/cache')
INDEX_CACHE_FILE = CACHE_ROOT / 'nn_index.pkl'
METADATA_CACHE_FILE = CACHE_ROOT / 'archive_metadata.json'


def load_batch_metadata(batch_id: str) -> List[Dict[str, Any]]:
    """Load metadata từ một batch với interventions."""
    try:
        return load_batch_with_interventions(batch_id)
    except Exception as e:
        print(f"Warning: Could not load batch {batch_id}: {e}")
        return []


def load_all_archive_metadata(force_reload: bool = False) -> List[Dict[str, Any]]:
    """Load metadata từ tất cả batches đã download."""
    CACHE_ROOT.mkdir(exist_ok=True, parents=True)
    
    # Try to load from cache first
    if not force_reload and METADATA_CACHE_FILE.exists():
        with open(METADATA_CACHE_FILE) as f:
            return json.load(f)
    
    all_metadata = []
    
    for batch_dir in ARCHIVE_ROOT.iterdir():
        if batch_dir.is_dir() and batch_dir.name.startswith('batch-'):
            print(f"Loading batch: {batch_dir.name}")
            batch_metadata = load_batch_metadata(batch_dir.name)
            all_metadata.extend(batch_metadata)
    
    # Cache metadata
    with open(METADATA_CACHE_FILE, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    return all_metadata


def build_nn_index(archive: List[Dict[str, Any]], force_rebuild: bool = False) -> NearestNeighbors:
    """Build NN index từ intervention scalars (duration_ms)."""
    CACHE_ROOT.mkdir(exist_ok=True, parents=True)
    
    # Try to load from cache first
    if not force_rebuild and INDEX_CACHE_FILE.exists():
        with open(INDEX_CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    
    # Extract intervention scalars (duration_ms)
    X = np.array([rec['intervention'][0] for rec in archive]).reshape(-1, 1)
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(X)
    
    # Cache index
    with open(INDEX_CACHE_FILE, 'wb') as f:
        pickle.dump(nn, f)
    
    return nn


def find_nearest_archive(v_m: float, archive: List[Dict[str, Any]], nn: NearestNeighbors) -> Dict[str, Any]:
    """Tìm image gần nhất với intervention v_m (duration_ms)."""
    v_m = np.array([v_m]).reshape(1, -1)
    _, i = nn.kneighbors(v_m)
    return archive[i[0][0]]


def test_index(archive: List[Dict[str, Any]], nn: NearestNeighbors):
    """Test NN index với vài intervention mẫu (duration_ms)."""
    test_interventions = [100.0, 500.0, 1000.0, 2000.0, 0.0]  # duration_ms values
    
    print("\n" + "="*60)
    print("Testing NN Index with sample interventions (duration_ms)")
    print("="*60)
    
    for i, v_m in enumerate(test_interventions):
        nearest = find_nearest_archive(v_m, archive, nn)
        distance = abs(v_m - nearest['intervention'][0])
        
        print(f"\nTest {i+1}: v_m = {v_m} ms")
        print(f"  Nearest: {nearest['type']} batch {nearest['batch']}")
        print(f"  Archive intervention: {nearest['intervention'][0]} ms")
        print(".1f")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test NN index with sample interventions'
    )
    parser.add_argument(
        '--force-reload',
        action='store_true',
        help='Force reload metadata from disk (ignore cache)'
    )
    parser.add_argument(
        '--force-rebuild-index',
        action='store_true',
        help='Force rebuild NN index (ignore cache)'
    )
    
    args = parser.parse_args()
    
    print("Loading archive metadata...")
    archive = load_all_archive_metadata(force_reload=args.force_reload)
    
    if not archive:
        print("Error: No archive data found!")
        print(f"Check if batches are downloaded to {ARCHIVE_ROOT}")
        return 1
    
    print(f"✓ Loaded {len(archive)} images from {len(set(r['batch'] for r in archive))} batches")
    
    print("Building NN index...")
    nn_index = build_nn_index(archive, force_rebuild=args.force_rebuild_index)
    print("✓ NN index ready")
    
    if args.test:
        test_index(archive, nn_index)
    
    print(f"\nArchive ready!")
    print(f"  Metadata cached: {METADATA_CACHE_FILE}")
    print(f"  NN index cached: {INDEX_CACHE_FILE}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
