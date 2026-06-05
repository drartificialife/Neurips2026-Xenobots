#!/usr/bin/env python3
"""
Run VLM interpretation on trajectories at different step sizes and compare results.
Usage:
    python scripts/compare_vlm_steps.py --batch batch-000070
"""

import json
import subprocess
import sys
from pathlib import Path
from argparse import ArgumentParser

def run_vlm_on_image(image_path, batch_id):
    """Run VLM on a single image, return scores."""
    cmd = [
        sys.executable, 'scripts/vlm_interpret_trajectory.py',
        '--image', str(image_path),
        '--batch', batch_id
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.returncode

def main():
    parser = ArgumentParser()
    parser.add_argument('--batch', default='batch-000070', help='Batch ID')
    parser.add_argument('--steps', default='1,5,10,20,40', help='Step sizes to test (comma-separated)')
    args = parser.parse_args()

    batch_id = args.batch
    steps = [int(s.strip()) for s in args.steps.split(',')]
    test_dir = Path('cache/trajectory_step_comparison')

    print("="*80)
    print(f"VLM STEP SIZE COMPARISON: {batch_id}")
    print("="*80)

    # Check that images exist
    print("\n[1] Checking trajectory images...")
    images = {}
    for step in steps:
        img_path = test_dir / f"{batch_id}_step{step}.png"
        if img_path.exists():
            images[step] = img_path
            print(f"  step={step:2d}: FOUND ({img_path.stat().st_size / 1024:.0f} KB)")
        else:
            print(f"  step={step:2d}: MISSING - generate first with test_step_sizes.sh")

    if not images:
        print("\nNo trajectory images found. Run: bash test_step_sizes.sh")
        return

    # Run VLM on each
    print("\n[2] Running VLM interpretation on each step size...")
    print("-"*80)

    results = {}
    for step in sorted(images.keys()):
        img_path = images[step]
        pts_estimate = max(541, 531) // step

        print(f"\nStep {step:2d} (~{pts_estimate} pts/phase):")
        print(f"  Image: {img_path.name}")

        stdout, returncode = run_vlm_on_image(img_path, batch_id)

        if returncode == 0:
            print(stdout)
            results[step] = stdout
        else:
            print(f"  ERROR in VLM execution")
            results[step] = None

    # Save results
    print("\n" + "="*80)
    print("[3] Saving comparison results...")

    output_file = test_dir / f"{batch_id}_vlm_comparison.json"
    with open(output_file, 'w') as f:
        json.dump({
            'batch_id': batch_id,
            'steps': steps,
            'results_saved': True,
            'comparison_dir': str(test_dir),
        }, f, indent=2)

    print(f"\nComparison data saved to: {output_file}")
    print(f"Trajectories in: {test_dir}")
    print(f"  - {batch_id}_step1.png")
    print(f"  - {batch_id}_step5.png")
    print(f"  - {batch_id}_step10.png")
    print(f"  - {batch_id}_step20.png")
    print(f"  - {batch_id}_step40.png")
    print("\nView the images to see how step size affects visual clarity.")
    print("="*80)

if __name__ == '__main__':
    main()
