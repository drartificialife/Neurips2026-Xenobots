#!/usr/bin/env python3
"""
Convert VLM captions (pre vs post) to behavior intervention scores.

Maps VLM descriptions to:
- stop moving
- move slow / move fast
- go slower / go faster

Usage:
    python scripts/vlm_to_behavior_scores.py --batch batch-000070 --step 10
"""

import json
import re
from pathlib import Path
from argparse import ArgumentParser


def extract_vlm_metrics(response: str) -> dict:
    """Extract quantitative metrics from VLM response."""
    metrics = {
        'distance': None,
        'speed': None,
        'coverage': None,
        'movement': None,
        'activity': None,
    }

    lines = response.split('\n')
    for line in lines:
        if 'DISTANCE:' in line:
            # Extract number from "DISTANCE: 100 pixels"
            match = re.search(r'DISTANCE:\s*([\d.]+)', line)
            if match:
                metrics['distance'] = float(match.group(1))

        elif 'SPEED:' in line:
            # Extract number from "SPEED: 0.5 px/frame"
            match = re.search(r'SPEED:\s*([\d.]+)', line)
            if match:
                metrics['speed'] = float(match.group(1))

        elif 'COVERAGE:' in line:
            # Extract percentage from "COVERAGE: 75%"
            match = re.search(r'COVERAGE:\s*([\d.]+)', line)
            if match:
                metrics['coverage'] = float(match.group(1))

        elif 'MOVEMENT:' in line:
            metrics['movement'] = line.replace('MOVEMENT:', '').strip().lower()

        elif 'ACTIVITY:' in line:
            metrics['activity'] = line.replace('ACTIVITY:', '').strip().lower()

    return metrics


def map_vlm_to_scores(pre_metrics: dict, post_metrics: dict) -> dict:
    """
    Map VLM estimated metrics to behavior scores (0-1).
    Uses quantitative metrics (distance, speed, coverage) for comparison.
    """
    scores = {
        'stop moving': 0.0,
        'move slow': 0.0,
        'move fast': 0.0,
        'go slower': 0.0,
        'go faster': 0.0,
    }

    pre_dist = pre_metrics.get('distance')
    post_dist = post_metrics.get('distance')
    pre_speed = pre_metrics.get('speed')
    post_speed = post_metrics.get('speed')
    pre_coverage = pre_metrics.get('coverage')
    post_coverage = post_metrics.get('coverage')

    # 1. stop moving: distance decreases significantly
    if pre_dist and post_dist:
        dist_reduction = (pre_dist - post_dist) / max(pre_dist, 0.1)
        scores['stop moving'] = min(1.0, max(0.0, dist_reduction))

    # 2. move slow: speed very low (< 0.5 px/frame) in post
    if post_speed is not None:
        scores['move slow'] = min(1.0, max(0.0, 1.0 - (post_speed / 0.5)))

    # 3. move fast: speed high (> 2 px/frame) in post
    if post_speed is not None:
        scores['move fast'] = min(1.0, max(0.0, post_speed / 2.0))

    # 4. go slower: speed decreases from pre to post
    if pre_speed and post_speed:
        speed_reduction = (pre_speed - post_speed) / max(pre_speed, 0.1)
        scores['go slower'] = min(1.0, max(0.0, speed_reduction))

    # 5. go faster: speed increases from pre to post
    if pre_speed and post_speed:
        speed_increase = (post_speed - pre_speed) / max(pre_speed, 0.1)
        scores['go faster'] = min(1.0, max(0.0, speed_increase))

    return scores


def analyze_batch(batch_id: str, step: int):
    """Analyze one batch's VLM captions and extract behavior scores."""
    caption_file = Path('cache/vlm_captions') / f'{batch_id}_step{step}_pre_intervention.json'

    if not caption_file.exists():
        print(f"[ERROR] Caption file not found: {caption_file}")
        print(f"        Run: python video_preprocessing/caption_video_vlm.py --batch {batch_id} --step {step}")
        return None

    print(f"\n{batch_id} (step={step}):")
    print(f"  Reading: {caption_file.name}")

    with open(caption_file) as f:
        captions = json.load(f)

    if len(captions) < 2:
        print(f"  [ERROR] Expected 2 captions (pre + post), got {len(captions)}")
        return None

    pre_caption = next((c for c in captions if c['phase'] == 'pre-intervention'), None)
    post_caption = next((c for c in captions if c['phase'] == 'post-intervention'), None)

    if not pre_caption or not post_caption:
        print(f"  [ERROR] Missing pre or post caption")
        return None

    # Extract metrics
    print(f"\n  PRE-INTERVENTION:")
    pre_metrics = extract_vlm_metrics(pre_caption['response'])
    for key, val in pre_metrics.items():
        print(f"    {key:10}: {val}")

    print(f"\n  POST-INTERVENTION:")
    post_metrics = extract_vlm_metrics(post_caption['response'])
    for key, val in post_metrics.items():
        print(f"    {key:10}: {val}")

    # Map to behavior scores
    scores = map_vlm_to_scores(pre_metrics, post_metrics)

    print(f"\n  BEHAVIOR SCORES:")
    for behavior, score in sorted(scores.items()):
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        print(f"    {behavior:15}: {score:.2f} [{bar}]")

    # Result
    result = {
        'batch_id': batch_id,
        'step': step,
        'pre_metrics': pre_metrics,
        'post_metrics': post_metrics,
        'behavior_scores': scores,
    }

    # Save
    out_file = Path('cache') / f'vlm_behavior_scores_{batch_id}_step{step}.json'
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to: {out_file}")

    return result


def main():
    parser = ArgumentParser(description='Convert VLM captions to behavior scores')
    parser.add_argument('--batch', required=True, help='Batch ID')
    parser.add_argument('--step', type=int, default=10, help='Step size (default: 10)')
    args = parser.parse_args()

    result = analyze_batch(args.batch, args.step)

    if result:
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nTop behavior scores:")
        sorted_scores = sorted(result['behavior_scores'].items(), key=lambda x: x[1], reverse=True)
        for behavior, score in sorted_scores[:3]:
            print(f"  {behavior:20}: {score:.2f}")


if __name__ == '__main__':
    main()
