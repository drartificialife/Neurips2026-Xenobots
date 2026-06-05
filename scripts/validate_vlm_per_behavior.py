#!/usr/bin/env python3
"""
Detailed per-behavior validation: VLM vs Vision.
Shows correlation, agreement, recommendations FOR EACH BEHAVIOR.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats
import matplotlib.pyplot as plt
import seaborn as sns

output_dir = Path('cache/vlm_vision_validation')
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading data...")
with open(Path('cache/archive_vlm_scores_all_prompts.json')) as f:
    vlm_data = json.load(f)

with open(Path('cache/vision_scores_step_1.json')) as f:
    vision_data = json.load(f)

with open(Path('evolution/train_test_split/prompts.json')) as f:
    prompts_dict = json.load(f)

# Mapping
variant_to_behavior = {}
for behavior, variants in prompts_dict.items():
    for variant in variants:
        variant_to_behavior[variant] = behavior

# Collect per-behavior data
behavior_data = {b: [] for b in prompts_dict.keys()}

for batch_id in vlm_data:
    if batch_id not in vision_data:
        continue

    vlm_batch = vlm_data[batch_id]
    vision_batch = vision_data[batch_id]

    if not isinstance(vlm_batch, dict) or not isinstance(vision_batch, dict):
        continue
    if 'duration_ms' in vlm_batch:
        vlm_batch = {k: v for k, v in vlm_batch.items() if k != 'duration_ms'}
    if 'step' not in vision_batch or 'scores' not in vision_batch:
        continue

    vision_scores = vision_batch['scores']

    # For each behavior, collect all VLM variants
    vlm_by_behavior = {}
    for prompt, score_data in vlm_batch.items():
        if not isinstance(score_data, dict) or 'score' not in score_data:
            continue
        vlm_score = score_data['score']
        if vlm_score < 0:
            continue

        behavior = variant_to_behavior.get(prompt)
        if not behavior:
            continue

        if behavior not in vlm_by_behavior:
            vlm_by_behavior[behavior] = []
        vlm_by_behavior[behavior].append(vlm_score)

    # Store per behavior
    for behavior, vision_score in vision_scores.items():
        if behavior not in vlm_by_behavior:
            continue

        vlm_scores = vlm_by_behavior[behavior]
        vlm_avg = np.mean(vlm_scores)

        behavior_data[behavior].append({
            'batch_id': batch_id,
            'vlm_avg': vlm_avg,
            'vlm_all': vlm_scores,
            'vlm_std': np.std(vlm_scores),
            'vision_score': vision_score,
            'difference': vlm_avg - vision_score,
            'abs_difference': abs(vlm_avg - vision_score),
        })

print(f"\nTotal comparisons per behavior:")
for b, items in behavior_data.items():
    print(f"  {b:15}: {len(items)} pairs")

# ============================================================================
# COMPUTE PER-BEHAVIOR METRICS
# ============================================================================

print("\n" + "="*80)
print("PER-BEHAVIOR DETAILED VALIDATION")
print("="*80)

per_behavior_results = {}

for behavior in sorted(behavior_data.keys()):
    items = behavior_data[behavior]

    vlm_scores = np.array([item['vlm_avg'] for item in items])
    vision_scores = np.array([item['vision_score'] for item in items])
    diffs = np.array([item['abs_difference'] for item in items])
    signed_diffs = np.array([item['difference'] for item in items])

    # Correlation
    if len(vlm_scores) > 2:
        pearson_r, pearson_p = sp_stats.pearsonr(vlm_scores, vision_scores)
        spearman_rho, spearman_p = sp_stats.spearmanr(vlm_scores, vision_scores)
    else:
        pearson_r = pearson_p = spearman_rho = spearman_p = np.nan

    # Agreement
    mae = np.mean(diffs)
    rmse = np.sqrt(np.mean(signed_diffs**2))
    agree_05 = np.sum(diffs <= 0.05) / len(diffs) * 100
    agree_10 = np.sum(diffs <= 0.10) / len(diffs) * 100
    agree_15 = np.sum(diffs <= 0.15) / len(diffs) * 100

    # T-test
    if len(vlm_scores) > 1:
        t_stat, t_pval = sp_stats.ttest_rel(vlm_scores, vision_scores)
    else:
        t_stat = t_pval = np.nan

    # Determine recommendation
    if pearson_r >= 0.65:
        recommendation = "GOOD - Can use VLM"
        rec_score = 3
    elif pearson_r >= 0.55:
        recommendation = "MODERATE - Use with caution"
        rec_score = 2
    elif pearson_r >= 0.45:
        recommendation = "WEAK - Prefer Vision, but VLM acceptable"
        rec_score = 1
    else:
        recommendation = "POOR - Use Vision ONLY"
        rec_score = 0

    per_behavior_results[behavior] = {
        'n_pairs': len(items),
        'vlm_mean': float(np.mean(vlm_scores)),
        'vlm_std': float(np.std(vlm_scores)),
        'vision_mean': float(np.mean(vision_scores)),
        'vision_std': float(np.std(vision_scores)),
        'pearson_r': float(pearson_r) if not np.isnan(pearson_r) else None,
        'pearson_p': float(pearson_p) if not np.isnan(pearson_p) else None,
        'spearman_rho': float(spearman_rho) if not np.isnan(spearman_rho) else None,
        'spearman_p': float(spearman_p) if not np.isnan(spearman_p) else None,
        'mae': float(mae),
        'rmse': float(rmse),
        'mean_signed_diff': float(np.mean(signed_diffs)),
        'std_signed_diff': float(np.std(signed_diffs)),
        'agreement_05': float(agree_05),
        'agreement_10': float(agree_10),
        'agreement_15': float(agree_15),
        't_statistic': float(t_stat) if not np.isnan(t_stat) else None,
        't_pvalue': float(t_pval) if not np.isnan(t_pval) else None,
        'recommendation': recommendation,
        'rec_score': rec_score,
    }

# Print detailed table
print("\n[DETAILED PER-BEHAVIOR METRICS]")
print("-"*80)
print("Behavior       | n_pairs | VLM_mean | Vision_mean | Pearson_r | MAE   | Agree_10% | Recommendation")
print("-"*80)

for behavior in sorted(per_behavior_results.keys()):
    r = per_behavior_results[behavior]
    print(f"{behavior:14} | {r['n_pairs']:7} | {r['vlm_mean']:8.3f} | {r['vision_mean']:11.3f} | "
          f"{r['pearson_r']:9.3f} | {r['mae']:5.3f} | {r['agreement_10']:8.1f}% | {r['recommendation']}")

# Print detailed stats per behavior
print("\n" + "="*80)
print("DETAILED STATISTICS PER BEHAVIOR")
print("="*80)

for behavior in sorted(behavior_data.keys()):
    r = per_behavior_results[behavior]
    items = behavior_data[behavior]

    print(f"\n{behavior.upper()}")
    print("-"*80)
    print(f"Sample size: {r['n_pairs']} pairs")
    print(f"\nScore distributions:")
    print(f"  VLM:    mean={r['vlm_mean']:.3f}, std={r['vlm_std']:.3f}")
    print(f"  Vision: mean={r['vision_mean']:.3f}, std={r['vision_std']:.3f}")

    print(f"\nCorrelation:")
    print(f"  Pearson r: {r['pearson_r']:.3f} (p={r['pearson_p']:.2e})")
    print(f"  Spearman rho: {r['spearman_rho']:.3f} (p={r['spearman_p']:.2e})")

    print(f"\nAgreement:")
    print(f"  MAE: {r['mae']:.3f}")
    print(f"  RMSE: {r['rmse']:.3f}")
    print(f"  Within 0.05: {r['agreement_05']:.1f}%")
    print(f"  Within 0.10: {r['agreement_10']:.1f}%")
    print(f"  Within 0.15: {r['agreement_15']:.1f}%")

    print(f"\nBias:")
    print(f"  Mean difference (VLM-Vision): {r['mean_signed_diff']:.3f} +/- {r['std_signed_diff']:.3f}")

    print(f"\nStatistical test:")
    print(f"  Paired t-test: t={r['t_statistic']:.3f}, p={r['t_pvalue']:.2e}")
    print(f"  Significant at 0.05: {r['t_pvalue'] < 0.05}")

    print(f"\nRECOMMENDATION: {r['recommendation']}")

    # Show outliers for this behavior
    diffs = np.array([item['abs_difference'] for item in items])
    outlier_threshold = np.percentile(diffs, 90)
    outliers = [item for item in items if item['abs_difference'] > outlier_threshold]

    if outliers:
        print(f"\nTop outliers (top 10% disagreement):")
        for outlier in sorted(outliers, key=lambda x: x['abs_difference'], reverse=True)[:5]:
            print(f"  {outlier['batch_id']:17} | VLM={outlier['vlm_avg']:.3f}, Vision={outlier['vision_score']:.3f}, Diff={outlier['abs_difference']:.3f}")

# ============================================================================
# SUMMARY & DECISION TABLE
# ============================================================================

print("\n" + "="*80)
print("TRAINING STRATEGY RECOMMENDATION")
print("="*80)

# Categorize behaviors
use_vlm = []
use_hybrid = []
use_vision_only = []

for behavior, r in per_behavior_results.items():
    if r['rec_score'] >= 2:
        use_vlm.append((behavior, r['pearson_r']))
    elif r['rec_score'] == 1:
        use_hybrid.append((behavior, r['pearson_r']))
    else:
        use_vision_only.append((behavior, r['pearson_r']))

print("\n[CATEGORY 1] Can use VLM for training:")
if use_vlm:
    for b, r in sorted(use_vlm, key=lambda x: x[1], reverse=True):
        print(f"  [OK] {b:15} (r={r:.3f})")
else:
    print("  (None - VLM too unreliable)")

print("\n[CATEGORY 2] Use Hybrid (Vision primary, VLM secondary):")
if use_hybrid:
    for b, r in sorted(use_hybrid, key=lambda x: x[1], reverse=True):
        print(f"  [WARN] {b:15} (r={r:.3f})")
else:
    print("  (None)")

print("\n[CATEGORY 3] Use Vision metric ONLY:")
if use_vision_only:
    for b, r in sorted(use_vision_only, key=lambda x: x[1], reverse=True):
        print(f"  [BAD] {b:15} (r={r:.3f})")
else:
    print("  (None)")

# Final recommendation
print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)

if use_vlm:
    print(f"\nStrategy: Selective VLM for {len(use_vlm)} behaviors, Vision for others")
    print("\nTraining code:")
    print("```python")
    for b, _ in sorted(use_vlm, key=lambda x: x[1], reverse=True):
        print(f"if behavior == '{b}':")
        print(f"    fitness = vlm_score  # VLM reliable enough")
    for b, _ in sorted(use_hybrid, key=lambda x: x[1], reverse=True):
        print(f"elif behavior == '{b}':")
        print(f"    fitness = 0.7 * vision_score + 0.3 * vlm_score  # Hybrid")
    for b, _ in sorted(use_vision_only, key=lambda x: x[1], reverse=True):
        print(f"elif behavior == '{b}':")
        print(f"    fitness = vision_score  # Vision only - VLM unreliable")
    print("```")
elif use_hybrid:
    print(f"\nStrategy: Hybrid (weighted average) for all behaviors")
    print("fitness = 0.7 * vision_score + 0.3 * vlm_score")
else:
    print(f"\nStrategy: Use Vision metric for ALL behaviors - VLM unreliable")
    print("fitness = vision_score")

# Save results
results_file = output_dir / 'per_behavior_validation.json'
with open(results_file, 'w') as f:
    json.dump(per_behavior_results, f, indent=2)

print(f"\nDetailed results saved to: {results_file}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\nGenerating visualization...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Per-Behavior VLM vs Vision Correlation', fontsize=14, fontweight='bold')

behaviors = sorted(behavior_data.keys())

for idx, behavior in enumerate(behaviors):
    ax = axes[idx // 3, idx % 3]
    items = behavior_data[behavior]

    vlm_scores = np.array([item['vlm_avg'] for item in items])
    vision_scores = np.array([item['vision_score'] for item in items])

    # Scatter
    ax.scatter(vision_scores, vlm_scores, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)

    # Diagonal (perfect agreement)
    ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect agreement', alpha=0.7)

    # Fitted line
    if len(vlm_scores) > 2:
        z = np.polyfit(vision_scores, vlm_scores, 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, 1, 100)
        ax.plot(x_line, p(x_line), 'b-', lw=2, alpha=0.7, label='Fitted line')

    r = per_behavior_results[behavior]

    # Color background based on quality
    if r['rec_score'] >= 2:
        bg_color = 'lightgreen'
        status = 'GOOD'
    elif r['rec_score'] >= 1:
        bg_color = 'lightyellow'
        status = 'WEAK'
    else:
        bg_color = 'lightcoral'
        status = 'POOR'

    ax.set_facecolor(bg_color)

    ax.text(0.05, 0.95, f'r={r["pearson_r"]:.3f}\nMAE={r["mae"]:.3f}\nn={r["n_pairs"]}\n{status}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Vision Score', fontsize=10)
    ax.set_ylabel('VLM Score', fontsize=10)
    ax.set_title(f'{behavior}', fontsize=11, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')

plt.tight_layout()
fig_file = output_dir / 'per_behavior_scatter.png'
plt.savefig(fig_file, dpi=150, bbox_inches='tight')
print(f"Saved: {fig_file}")
plt.close()

print("\n" + "="*80)
print("DONE")
print("="*80)

