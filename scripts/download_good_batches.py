#!/usr/bin/env python3
"""
Download specified good batches from Dropbox using rclone.

Usage:
    python scripts/download_good_batches.py [--dry-run] [--batch BATCH_ID]
    
Examples:
    # Download all good batches listed in good_batch.txt
    python scripts/download_good_batches.py
    
    # Dry-run (show what would be downloaded without downloading)
    python scripts/download_good_batches.py --dry-run
    
    # Download specific batch
    python scripts/download_good_batches.py --batch batch_001
"""

import subprocess
import sys
from pathlib import Path
import argparse
import time

ARCHIVE_ROOT = Path('D:/xenobot_videos')
BATCH_LIST_FILE = Path('good_batch.txt')
RCLONE_REMOTE = 'dropbox_xenobot'
DROPBOX_ARCHIVE_BASE = '/BIOSENSE Team Folder/biosense_communication_interface/prod/archive'

# Rclone optimization settings
RCLONE_OPTS = {
    'transfers': 8,      # Parallel transfers
    'checkers': 16,      # Parallel checkers
    'fast_list': True,   # Fast listing
    'retries': 3,        # Auto-retry
    'progress': True,    # Show progress
    'verbose': False,    # Detailed logging (turn on for debug)
}


def load_good_batches():
    """Read good batch IDs from good_batch.txt."""
    if not BATCH_LIST_FILE.exists():
        print(f"Error: {BATCH_LIST_FILE} not found!")
        print("Please create this file with one batch ID per line.")
        sys.exit(1)
    
    with open(BATCH_LIST_FILE) as f:
        batches = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    return batches


def build_rclone_command(dropbox_path, local_path, dry_run=False):
    """Build rclone sync command with options."""
    cmd = ['rclone', 'sync', dropbox_path, str(local_path)]
    
    if dry_run:
        cmd.append('--dry-run')
    
    if RCLONE_OPTS['transfers']:
        cmd.extend(['--transfers', str(RCLONE_OPTS['transfers'])])
    
    if RCLONE_OPTS['checkers']:
        cmd.extend(['--checkers', str(RCLONE_OPTS['checkers'])])
    
    if RCLONE_OPTS['fast_list']:
        cmd.append('--fast-list')
    
    if RCLONE_OPTS['retries']:
        cmd.extend(['--retries', str(RCLONE_OPTS['retries'])])
    
    if RCLONE_OPTS['progress']:
        cmd.append('--progress')
    
    if RCLONE_OPTS['verbose']:
        cmd.append('--verbose')
    
    return cmd


def download_batch(batch_id, dry_run=False):
    """Download single batch using rclone."""
    dropbox_path = f'{RCLONE_REMOTE}:{DROPBOX_ARCHIVE_BASE}/{batch_id}'
    local_path = ARCHIVE_ROOT / batch_id
    
    print(f"\n{'='*60}")
    print(f"{'[DRY-RUN]' if dry_run else '[SYNC]'} {batch_id}")
    print(f"From: {dropbox_path}")
    print(f"To:   {local_path}")
    print(f"{'='*60}")
    
    # Create local directory
    local_path.mkdir(parents=True, exist_ok=True)
    
    # Build and run rclone command
    cmd = build_rclone_command(dropbox_path, local_path, dry_run=dry_run)
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✓ {batch_id} {'(dry-run simulation)' if dry_run else 'downloaded'}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {batch_id} failed with exit code {e.returncode}")
        return False


def check_rclone_installed():
    """Check if rclone is installed and accessible."""
    try:
        subprocess.run(['rclone', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_rclone_config(remote_name):
    """Check if rclone remote is configured."""
    try:
        result = subprocess.run(
            ['rclone', 'lsd', f'{remote_name}:{DROPBOX_ARCHIVE_BASE}'],
            capture_output=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be downloaded without actually downloading'
    )
    parser.add_argument(
        '--batch',
        type=str,
        help='Download specific batch ID instead of all good batches'
    )
    parser.add_argument(
        '--remote',
        type=str,
        default=RCLONE_REMOTE,
        help=f'Rclone remote name (default: {RCLONE_REMOTE})'
    )
    
    args = parser.parse_args()
    
    # Pre-flight checks
    if not check_rclone_installed():
        print("Error: rclone not found. Please install rclone:")
        print("  Windows: choco install rclone")
        print("  Linux:   curl https://rclone.org/install.sh | sudo bash")
        sys.exit(1)
    
    if not check_rclone_config(args.remote):
        print(f"Error: rclone remote '{args.remote}' not configured.")
        print("Run: rclone config")
        sys.exit(1)
    
    # Load batches
    if args.batch:
        batches = [args.batch]
    else:
        batches = load_good_batches()
    
    print(f"\nPrepared to download {len(batches)} batch(es)")
    if args.dry_run:
        print("[DRY-RUN MODE - No files will be downloaded]\n")
    
    # Download
    results = []
    start_time = time.time()
    
    for batch_id in batches:
        success = download_batch(batch_id, dry_run=args.dry_run)
        results.append((batch_id, success))
    
    # Summary
    elapsed = time.time() - start_time
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful
    
    print(f"\n{'='*60}")
    print(f"Summary: {successful}/{len(results)} successful")
    if failed > 0:
        print(f"Failed batches:")
        for batch_id, success in results:
            if not success:
                print(f"  - {batch_id}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'='*60}\n")
    
    if args.dry_run:
        print("Tip: Remove --dry-run to actually download.")
    else:
        print(f"✓ Archive ready at: {ARCHIVE_ROOT}")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
