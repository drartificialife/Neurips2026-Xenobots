#!/usr/bin/env python3
"""
Analyze behavior distribution from vision ground truth.
Shows which behaviors are dominant in batches and identifies data imbalance.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

output_dir = Path('cache/vlm_vision_validation')
output_dir.mkdir(parents=True, exist_ok=True)

# Load vision scores
print("Loading vision scores...")
with open(Path('cache/vision_scores_step_1.json')) as f:
    vision_data = json.load(f)

print(f"Total batches: {len(vision_data)}")

# Collect scores per behavior
behavior_scores = defaultdict(list)
batch_dominant = []

for batch_id, batch_data in vision_data.items():
    if not isinstance(batch_data, dict) or 'scores' not in batch_data:
        continue

    scores = batch_data['scores']

    # Collect scores
    for behavior, score in scores.items():
        behavior_scores[behavior].append(score)

    # Determine dominant behavior (highest score)
    if scores:
        dominant_behavior = max(scores.items(), key=lambda x: x[1])
        batch_dominant.append({
            'batch_id': batch_id,
            'dominant_behavior': dominant_behavior[0],
            'dominant_score': dominant_behavior[1],
            'all_scores': scores
        })

print("\n" + "="*80)
print("BEHAVIOR DISTRIBUTION ANALYSIS")
print("="*80)

# Statistics per behavior
print("\n[SCORE STATISTICS PER BEHAVIOR]")
print("-"*80)
print(f"{'Behavior':20} | {'Count':8} | {'Mean':8} | {'Std':8} | {'Min':8} | {'Max':8} | {'Median':8}")
print("-"*80)

behavior_stats = {}
for behavior in sorted(behavior_scores.keys()):
    scores = np.array(behavior_scores[behavior])
    behavior_stats[behavior] = {
        'count': len(scores),
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'min': float(np.min(scores)),
        'max': float(np.max(scores)),
        'median': float(np.median(scores)),
    }
    stat = behavior_stats[behavior]
    print(f"{behavior:20} | {stat['count']:8} | {stat['mean']:8.3f} | {stat['std']:8.3f} | "
          f"{stat['min']:8.3f} | {stat['max']:8.3f} | {stat['median']:8.3f}")

# Dominant behavior distribution
print("\n[DOMINANT BEHAVIOR DISTRIBUTION (by batch)]")
print("-"*80)
dominant_counts = defaultdict(int)
for item in batch_dominant:
    dominant_counts[item['dominant_behavior']] += 1

total_batches = len(batch_dominant)
for behavior in sorted(dominant_counts.keys()):
    count = dominant_counts[behavior]
    pct = 100 * count / total_batches
    print(f"{behavior:20}: {count:3} batches ({pct:5.1f}%)")

# Score range distribution
print("\n[SCORE RANGE DISTRIBUTION PER BEHAVIOR]")
print("-"*80)
ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

for behavior in sorted(behavior_scores.keys()):
    scores = np.array(behavior_scores[behavior])
    print(f"\n{behavior}:")
    for low, high in ranges:
        count = np.sum((scores >= low) & (scores < high))
        pct = 100 * count / len(scores)
        print(f"  [{low:.1f}, {high:.1f}): {count:3} ({pct:5.1f}%)")

# High vs low performance behaviors
print("\n[DATA IMBALANCE ANALYSIS]")
print("-"*80)
overall_mean = np.mean([s for scores in behavior_scores.values() for s in scores])
print(f"Overall average score: {overall_mean:.3f}\n")

imbalance = []
for behavior in sorted(behavior_stats.keys()):
    stat = behavior_stats[behavior]
    diff = stat['mean'] - overall_mean
    imbalance.append((behavior, stat['mean'], diff))

imbalance.sort(key=lambda x: x[1], reverse=True)

print("Behaviors by average score (sorted):")
for behavior, mean, diff in imbalance:
    marker = "[HIGH]" if diff > 0.05 else "[LOW ]" if diff < -0.05 else "[MID ]"
    print(f"{marker} {behavior:20}: {mean:.3f} ({diff:+.3f})")

# Identify problem behaviors
print("\n[FINDINGS]")
print("-"*80)
high_behaviors = [b for b, _, d in imbalance if d > 0.05]
low_behaviors = [b for b, _, d in imbalance if d < -0.05]

if high_behaviors:
    print(f"[OVERREPRESENTED] High average score: {', '.join(high_behaviors)}")
    print("  -> These behaviors may be easier to trigger or more common in data")
else:
    print("[OK] No significantly overrepresented behaviors")

if low_behaviors:
    print(f"[UNDERREPRESENTED] Low average score: {', '.join(low_behaviors)}")
    print("  -> These behaviors may be harder or rarer in data")
else:
    print("[OK] No significantly underrepresented behaviors")

# Check if distribution matches VLM performance
print("\n[RELATIONSHIP TO VLM PERFORMANCE]")
print("-"*80)
vlm_performance = {
    'stop moving': {'r': 0.673, 'category': 'GOOD'},
    'go slower': {'r': 0.622, 'category': 'GOOD'},
    'move fast': {'r': 0.548, 'category': 'WEAK'},
    'move slow': {'r': 0.543, 'category': 'WEAK'},
    'go faster': {'r': 0.344, 'category': 'POOR'},
}

print("Behavior               | Mean Score | VLM r    | VLM Category | Data Balance")
print("-"*80)
for behavior, mean_score, _ in imbalance:
    vlm_info = vlm_performance.get(behavior, {})
    vlm_r = vlm_info.get('r', 0)
    vlm_cat = vlm_info.get('category', 'N/A')
    data_balance = "[OK]" if -0.05 <= (mean_score - overall_mean) <= 0.05 else "[IMBALANCED]"
    print(f"{behavior:20} | {mean_score:10.3f} | {vlm_r:8.3f} | {vlm_cat:12} | {data_balance}")

# Save results
results = {
    'total_batches': total_batches,
    'behavior_stats': behavior_stats,
    'dominant_counts': dict(dominant_counts),
    'overall_mean': float(overall_mean),
    'imbalanced_behaviors': {
        'high': high_behaviors,
        'low': low_behaviors,
    }
}

results_file = output_dir / 'behavior_distribution_stats.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nDetailed results saved to: {results_file}")

print("\n" + "="*80)
print("DONE")
print("="*80)
