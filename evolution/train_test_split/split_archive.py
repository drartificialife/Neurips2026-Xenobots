#!/usr/bin/env python3
"""
Split archive into train (80%) and test (20%) sets,
stratified by duration_ms to ensure both sets cover the full duration range.

Strategy: sort by duration_ms, take every 5th batch as test.

Output:
  train_batches.json  — 111 batches for evolution fitness evaluation
  test_batches.json   —  28 batches for held-out video generalization test
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
ARCHIVE_FILE = PROJECT_ROOT / 'cache' / 'archive_vlm_scores_v2.json'
OUT_DIR      = Path(__file__).parent

with open(ARCHIVE_FILE) as f:
    archive = json.load(f)

# Sort batch IDs by duration_ms
batches_sorted = sorted(archive.keys(), key=lambda b: archive[b]['duration_ms'])

print(f"Total batches: {len(batches_sorted)}")
print(f"Duration range: {archive[batches_sorted[0]]['duration_ms']} ms "
      f"— {archive[batches_sorted[-1]]['duration_ms']} ms")

# Stratified split: every 5th batch → test
test_ids  = [b for i, b in enumerate(batches_sorted) if i % 5 == 2]
train_ids = [b for i, b in enumerate(batches_sorted) if i % 5 != 2]

train_archive = {b: archive[b] for b in train_ids}
test_archive  = {b: archive[b] for b in test_ids}

print(f"\nTrain: {len(train_archive)} batches")
print(f"Test:  {len(test_archive)}  batches")

# Verify duration coverage
train_durations = sorted(archive[b]['duration_ms'] for b in train_ids)
test_durations  = sorted(archive[b]['duration_ms'] for b in test_ids)

print(f"\nTrain duration range: {train_durations[0]} — {train_durations[-1]} ms")
print(f"Test  duration range: {test_durations[0]}  — {test_durations[-1]}  ms")

# Save
with open(OUT_DIR / 'train_archive.json', 'w') as f:
    json.dump(train_archive, f)

with open(OUT_DIR / 'test_archive.json', 'w') as f:
    json.dump(test_archive, f)

print(f"\n[OK] Saved: train_archive.json ({len(train_archive)} batches)")
print(f"[OK] Saved: test_archive.json  ({len(test_archive)} batches)")
