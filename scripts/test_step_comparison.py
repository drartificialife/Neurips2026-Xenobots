#!/usr/bin/env python3
"""
Generate trajectories for one batch at multiple step sizes and compare VLM interpretation.
Tests: step=1, 5, 10, 20, 40 for same batch
"""

import json
import subprocess
from pathlib import Path

# Configuration
BATCH_ID = "batch-000070"
STEPS = [1, 5, 10, 20, 40]
TEST_DIR = Path('cache/trajectory_step_comparison')
TEST_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print(f"TRAJECTORY STEP SIZE COMPARISON: {BATCH_ID}")
print("="*80)

# 1. Generate trajectories at different step sizes
print("\n[1] Generating trajectories...")
trajectories = {}

for step in STEPS:
    output_file = TEST_DIR / f"{BATCH_ID}_step{step}.png"
    print(f"\nGenerating step={step}...")

    cmd = [
        'python', 'video_preprocessing/generate_trajectory_heatmap.py',
        '--batch', BATCH_ID,
        '--step', str(step),
        '--output', str(output_file)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  [OK] Saved to {output_file}")
        trajectories[step] = str(output_file)
    else:
        print(f"  [FAIL] {result.stderr}")

# 2. Run VLM interpretation on each trajectory
print("\n" + "="*80)
print("[2] Running VLM interpretation...")
print("="*80)

vlm_results = {}

for step in STEPS:
    if step not in trajectories:
        print(f"\nStep {step}: SKIPPED (generation failed)")
        continue

    traj_path = trajectories[step]
    print(f"\nStep {step}: {traj_path}")

    cmd = [
        'python', 'scripts/vlm_interpret_trajectory.py',
        '--image', traj_path,
        '--batch', BATCH_ID
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        # Try to parse VLM output
        print(result.stdout)
        vlm_results[step] = result.stdout
    else:
        print(f"  Error: {result.stderr}")
        vlm_results[step] = None

# 3. Summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

# Extract scores from outputs if possible
print(f"\n{'Step':>6} | {'Pts/Phase':>11} | VLM Interpretation")
print("-"*80)

for step in STEPS:
    if step not in trajectories:
        print(f"{step:>6} | {'FAILED':>11} |")
        continue

    # Try to estimate points per phase
    pts_pre = 541 // step
    pts_post = 531 // step
    avg_pts = (pts_pre + pts_post) / 2

    print(f"{step:>6} | {avg_pts:>11.0f} | (see above)")

print("\n" + "="*80)
print("Files saved to: " + str(TEST_DIR))
print("="*80)
