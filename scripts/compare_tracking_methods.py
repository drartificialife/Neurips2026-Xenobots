#!/usr/bin/env python3
"""
Compare template matching vs optical flow for same batch.
Shows which method gives more realistic motion metrics.

Usage:
    python scripts/compare_tracking_methods.py --batch batch-000070 --step 10
"""

import json
import sys
from pathlib import Path
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('--batch', default='batch-000070')
    parser.add_argument('--step', type=int, default=10)
    args = parser.parse_args()

    batch_id = args.batch
    step = args.step

    print("="*80)
    print(f"TRACKING METHOD COMPARISON: {batch_id}")
    print("="*80)

    # Load template matching results (from vision_scores)
    tm_file = Path(f'cache/vision_scores_step_{step}.json')
    of_file = Path(f'cache/optical_flow_metrics_step_{step}.json')

    tm_data = None
    of_data = None

    # Load template matching
    if tm_file.exists():
        with open(tm_file) as f:
            data = json.load(f)
            if batch_id in data and 'kinematics' in data[batch_id]:
                tm_data = data[batch_id]
                print(f"\n[1] TEMPLATE MATCHING (from {tm_file.name}):")
                k = tm_data['kinematics']
                print(f"    Pre distance:  {k['pre_distance_px']:8.1f} px")
                print(f"    Post distance: {k['post_distance_px']:8.1f} px")
                print(f"    Pre velocity:  {k['pre_velocity_px_per_frame']:8.3f} px/frame")
                print(f"    Post velocity: {k['post_velocity_px_per_frame']:8.3f} px/frame")
    else:
        print(f"\n[1] TEMPLATE MATCHING: FILE NOT FOUND ({tm_file})")
        print("    Run: python video_preprocessing/compute_vision_scores.py --batch batch-000070 --step 10")

    # Load optical flow
    if of_file.exists():
        with open(of_file) as f:
            data = json.load(f)
            if batch_id in data and 'kinematics' in data[batch_id]:
                of_data = data[batch_id]
                print(f"\n[2] OPTICAL FLOW (from {of_file.name}):")
                k = of_data['kinematics']
                print(f"    Pre distance:  {k['pre_distance_px']:8.1f} px")
                print(f"    Post distance: {k['post_distance_px']:8.1f} px")
                print(f"    Pre velocity:  {k['pre_velocity_px_per_frame']:8.3f} px/frame")
                print(f"    Post velocity: {k['post_velocity_px_per_frame']:8.3f} px/frame")
    else:
        print(f"\n[2] OPTICAL FLOW: FILE NOT FOUND ({of_file})")
        print("    Run: python video_preprocessing/compute_optical_flow_metrics.py --batch batch-000070 --step 10")

    # Comparison
    if tm_data and of_data:
        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)

        tm_k = tm_data['kinematics']
        of_k = of_data['kinematics']

        metrics = ['pre_distance_px', 'post_distance_px', 'pre_velocity_px_per_frame', 'post_velocity_px_per_frame']

        print(f"\n{'Metric':30} | {'Template Match':>15} | {'Optical Flow':>15} | {'Ratio (OF/TM)':>12}")
        print("-"*80)

        for metric in metrics:
            tm_val = tm_k[metric]
            of_val = of_k[metric]
            ratio = of_val / tm_val if tm_val > 0 else 0

            print(f"{metric:30} | {tm_val:15.1f} | {of_val:15.1f} | {ratio:12.2f}x")

        # Visual interpretation
        print("\n[INTERPRETATION]")
        pre_ratio = of_k['pre_distance_px'] / max(tm_k['pre_distance_px'], 0.1)
        post_ratio = of_k['post_distance_px'] / max(tm_k['post_distance_px'], 0.1)

        if pre_ratio > 3:
            print(f"  ⚠️  Optical flow shows {pre_ratio:.1f}x MORE motion in pre-phase")
            print(f"      → Template matching might be missing rotational motion")
            print(f"      → Bot likely rotated significantly while moving")
        elif pre_ratio < 1.2:
            print(f"  ✓ Pre-phase: Methods agree (ratio={pre_ratio:.2f})")
        else:
            print(f"  ~ Pre-phase: Optical flow slightly higher (ratio={pre_ratio:.2f})")
            print(f"    → Some rotational component detected")

        if post_ratio > 3:
            print(f"  ⚠️  Optical flow shows {post_ratio:.1f}x MORE motion in post-phase")
            print(f"      → Intervention caused rotation, not just translation")
        elif post_ratio < 1.2:
            print(f"  ✓ Post-phase: Methods agree (ratio={post_ratio:.2f})")
        else:
            print(f"  ~ Post-phase: Optical flow slightly higher (ratio={post_ratio:.2f})")

        # Score comparison
        print("\n[BEHAVIOR SCORES]")
        print(f"\n{'Behavior':20} | {'Template Match':>15} | {'Optical Flow':>15}")
        print("-"*55)

        for behavior in ['stop moving', 'move slow', 'move fast', 'go slower', 'go faster']:
            tm_score = tm_data['scores'].get(behavior, 0)
            of_score = of_data['scores'].get(behavior, 0)
            diff = of_score - tm_score

            marker = "→" if abs(diff) > 0.1 else " "
            print(f"{behavior:20} | {tm_score:15.3f} | {of_score:15.3f} {marker}")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
