#!/usr/bin/env python3
"""
Validate VLM scores against Vision (objective) metrics BEFORE training.

Compares 30 VLM prompts with 5 vision metrics.
Computes: correlation, agreement, statistical tests, outliers.

Output: cache/vlm_vision_validation/
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats
import matplotlib.pyplot as plt
import seaborn as sns

# Create output directory
output_dir = Path('cache/vlm_vision_validation')
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading VLM scores...")
with open(Path('cache/archive_vlm_scores_all_prompts.json')) as f:
    vlm_data = json.load(f)

print("Loading Vision scores...")
with open(Path('cache/vision_scores_step_1.json')) as f:
    vision_data = json.load(f)

# Load prompts to map variants → base behavior
with open(Path('evolution/train_test_split/prompts.json')) as f:
    prompts_dict = json.load(f)

# Create mapping: variant → base_behavior
variant_to_behavior = {}
for behavior, variants in prompts_dict.items():
    for variant in variants:
        variant_to_behavior[variant] = behavior

print(f"Loaded {len(vlm_data)} batches from VLM")
print(f"Loaded {len(vision_data)} batches from Vision")

# Prepare comparison data
comparison_data = []

for batch_id in vlm_data:
    if batch_id not in vision_data:
        continue
    if batch_id == 'batch-000070':  # Skip if metadata
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

    # Map each VLM prompt to base behavior, aggregate
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

    # Compare: average VLM per behavior vs vision per behavior
    for behavior, vision_score in vision_scores.items():
        if behavior not in vlm_by_behavior:
            continue

        vlm_scores_for_behavior = vlm_by_behavior[behavior]
        vlm_avg = np.mean(vlm_scores_for_behavior)

        comparison_data.append({
            'batch_id': batch_id,
            'behavior': behavior,
            'vlm_score': vlm_avg,
            'vlm_n_variants': len(vlm_scores_for_behavior),
            'vlm_all_scores': vlm_scores_for_behavior,
            'vision_score': vision_score,
            'difference': vlm_avg - vision_score,
            'abs_difference': abs(vlm_avg - vision_score),
        })

print(f"\nTotal comparisons: {len(comparison_data)}")

# Convert to dataframe-like structure
comparison_dict = {}
for item in comparison_data:
    behavior = item['behavior']
    if behavior not in comparison_dict:
        comparison_dict[behavior] = []
    comparison_dict[behavior].append(item)

# ============================================================================
# COMPUTE METRICS
# ============================================================================

print("\n" + "="*80)
print("VALIDATION RESULTS: VLM vs Vision Metrics")
print("="*80)

results = {
    'total_comparisons': len(comparison_data),
    'n_batches': len(set(c['batch_id'] for c in comparison_data)),
    'n_behaviors': len(comparison_dict),
    'overall_metrics': {},
    'per_behavior_metrics': {},
    'statistical_tests': {},
    'outlier_analysis': {},
    'scatter_data': {},
}

# Overall metrics
all_vlm = np.array([c['vlm_score'] for c in comparison_data])
all_vision = np.array([c['vision_score'] for c in comparison_data])
all_diff = np.array([c['difference'] for c in comparison_data])
all_abs_diff = np.array([c['abs_difference'] for c in comparison_data])

pearson_r, pearson_p = sp_stats.pearsonr(all_vlm, all_vision)
spearman_rho, spearman_p = sp_stats.spearmanr(all_vlm, all_vision)
mae = np.mean(all_abs_diff)
rmse = np.sqrt(np.mean(all_diff**2))

results['overall_metrics'] = {
    'pearson_r': float(pearson_r),
    'pearson_p_value': float(pearson_p),
    'spearman_rho': float(spearman_rho),
    'spearman_p_value': float(spearman_p),
    'mean_absolute_error': float(mae),
    'rmse': float(rmse),
    'mean_difference': float(np.mean(all_diff)),
    'std_difference': float(np.std(all_diff)),
    'vlm_mean': float(np.mean(all_vlm)),
    'vlm_std': float(np.std(all_vlm)),
    'vision_mean': float(np.mean(all_vision)),
    'vision_std': float(np.std(all_vision)),
}

print("\n[OVERALL METRICS]")
print("-"*80)
print(f"Pearson correlation (r):     {pearson_r:7.3f} (p={pearson_p:.2e})")
print(f"Spearman rank correlation:   {spearman_rho:7.3f} (p={spearman_p:.2e})")
print(f"Mean Absolute Error (MAE):   {mae:7.3f}")
print(f"Root Mean Square Error:      {rmse:7.3f}")
print(f"Mean difference (VLM-Vision):{np.mean(all_diff):7.3f} +/- {np.std(all_diff):.3f}")

# Per-behavior metrics
print(f"\n[PER-BEHAVIOR METRICS]")
print("-"*80)
print("Behavior        | n_pairs | VLM_mean | Vision_mean | Corr_r | MAE")
print("-"*80)

for behavior in sorted(comparison_dict.keys()):
    items = comparison_dict[behavior]
    vlm_scores = np.array([item['vlm_score'] for item in items])
    vision_scores = np.array([item['vision_score'] for item in items])
    diffs = np.array([item['abs_difference'] for item in items])

    if len(vlm_scores) > 1:
        corr_r, _ = sp_stats.pearsonr(vlm_scores, vision_scores)
    else:
        corr_r = np.nan

    mae_behavior = np.mean(diffs)

    results['per_behavior_metrics'][behavior] = {
        'n_pairs': len(items),
        'vlm_mean': float(np.mean(vlm_scores)),
        'vlm_std': float(np.std(vlm_scores)),
        'vision_mean': float(np.mean(vision_scores)),
        'vision_std': float(np.std(vision_scores)),
        'correlation_r': float(corr_r) if not np.isnan(corr_r) else None,
        'mae': float(mae_behavior),
    }

    print(f"{behavior:15} | {len(items):7} | {np.mean(vlm_scores):8.3f} | "
          f"{np.mean(vision_scores):11.3f} | {corr_r:6.3f} | {mae_behavior:6.3f}")

# Statistical tests
print(f"\n[STATISTICAL TESTS]")
print("-"*80)

# Paired t-test
t_stat, t_pval = sp_stats.ttest_rel(all_vlm, all_vision)
print(f"Paired t-test (VLM vs Vision):")
print(f"  t-statistic: {t_stat:7.3f}")
print(f"  p-value:     {t_pval:.2e}")
print(f"  Interpretation: VLM {'SIGNIFICANTLY' if t_pval < 0.05 else 'NOT significantly'} "
      f"different from Vision")

results['statistical_tests']['paired_ttest'] = {
    't_statistic': float(t_stat),
    'p_value': float(t_pval),
    'significant_at_0.05': bool(t_pval < 0.05),
}

# Wilcoxon signed-rank (non-parametric)
w_stat, w_pval = sp_stats.wilcoxon(all_vlm, all_vision)
print(f"\nWilcoxon signed-rank test:")
print(f"  W-statistic: {w_stat:7.3f}")
print(f"  p-value:     {w_pval:.2e}")

results['statistical_tests']['wilcoxon'] = {
    'w_statistic': float(w_stat),
    'p_value': float(w_pval),
}

# Agreement analysis
print(f"\n[AGREEMENT ANALYSIS]")
print("-"*80)

thresholds = [0.05, 0.10, 0.15, 0.20]
for threshold in thresholds:
    agree = np.sum(all_abs_diff <= threshold)
    pct = 100 * agree / len(all_abs_diff)
    print(f"Batches with <{threshold} difference: {agree}/{len(all_abs_diff)} ({pct:.1f}%)")

results['agreement_analysis'] = {
    'thresholds': thresholds,
    'agreement_percentages': {
        f'threshold_{t}': float(100 * np.sum(all_abs_diff <= t) / len(all_abs_diff))
        for t in thresholds
    }
}

# Outliers (batches with large disagreement)
print(f"\n[OUTLIER ANALYSIS]")
print("-"*80)

outlier_threshold = np.percentile(all_abs_diff, 90)
outliers = [c for c in comparison_data if c['abs_difference'] > outlier_threshold]

print(f"Outliers (top 10% disagreement, >={outlier_threshold:.3f}):")
print(f"Batch ID          | Behavior       | VLM    | Vision | Diff")
print("-"*80)

for item in sorted(outliers, key=lambda x: x['abs_difference'], reverse=True)[:10]:
    print(f"{item['batch_id']:17} | {item['behavior']:14} | "
          f"{item['vlm_score']:6.3f} | {item['vision_score']:6.3f} | {item['abs_difference']:6.3f}")

results['outlier_analysis'] = {
    'outlier_threshold': float(outlier_threshold),
    'n_outliers': len(outliers),
    'top_outliers': [
        {
            'batch_id': o['batch_id'],
            'behavior': o['behavior'],
            'vlm_score': float(o['vlm_score']),
            'vision_score': float(o['vision_score']),
            'difference': float(o['abs_difference']),
        }
        for o in sorted(outliers, key=lambda x: x['abs_difference'], reverse=True)[:10]
    ]
}

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print(f"\n[GENERATING VISUALIZATIONS]")

# 1. Scatter plot: VLM vs Vision
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('VLM vs Vision Scores by Behavior', fontsize=14, fontweight='bold')

behaviors = sorted(comparison_dict.keys())
for idx, behavior in enumerate(behaviors):
    ax = axes[idx // 3, idx % 3]
    items = comparison_dict[behavior]

    vlm_scores = np.array([item['vlm_score'] for item in items])
    vision_scores = np.array([item['vision_score'] for item in items])

    ax.scatter(vision_scores, vlm_scores, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)

    # Diagonal line
    ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect agreement')

    # Correlation
    if len(vlm_scores) > 1:
        corr_r, _ = sp_stats.pearsonr(vlm_scores, vision_scores)
        ax.text(0.05, 0.95, f'r={corr_r:.3f}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Vision Score', fontsize=10)
    ax.set_ylabel('VLM Score', fontsize=10)
    ax.set_title(f'{behavior} (n={len(items)})', fontsize=11, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

# Hide unused subplot
axes[1, 2].axis('off')

plt.tight_layout()
scatter_file = output_dir / 'vlm_vs_vision_scatter.png'
plt.savefig(scatter_file, dpi=150, bbox_inches='tight')
print(f"  Saved: {scatter_file}")
plt.close()

# 2. Bland-Altman plot
fig, ax = plt.subplots(figsize=(10, 7))

means = (all_vlm + all_vision) / 2
diffs = all_vlm - all_vision

ax.scatter(means, diffs, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
ax.axhline(y=0, color='r', linestyle='-', lw=2, label='No difference')

mean_diff = np.mean(diffs)
std_diff = np.std(diffs)

ax.axhline(y=mean_diff, color='g', linestyle='--', lw=2, label=f'Mean diff: {mean_diff:.3f}')
ax.axhline(y=mean_diff + 1.96*std_diff, color='orange', linestyle='--', lw=1.5,
           label=f'+1.96 SD: {mean_diff + 1.96*std_diff:.3f}')
ax.axhline(y=mean_diff - 1.96*std_diff, color='orange', linestyle='--', lw=1.5,
           label=f'-1.96 SD: {mean_diff - 1.96*std_diff:.3f}')

ax.set_xlabel('Average of VLM and Vision Scores', fontsize=11)
ax.set_ylabel('Difference (VLM - Vision)', fontsize=11)
ax.set_title('Bland-Altman Plot: Agreement between VLM and Vision', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

bland_altman_file = output_dir / 'bland_altman_plot.png'
plt.savefig(bland_altman_file, dpi=150, bbox_inches='tight')
print(f"  Saved: {bland_altman_file}")
plt.close()

# 3. Distribution comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(all_vlm, bins=20, alpha=0.6, label='VLM', color='blue', edgecolor='black')
axes[0].hist(all_vision, bins=20, alpha=0.6, label='Vision', color='green', edgecolor='black')
axes[0].set_xlabel('Score', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Distribution of Scores', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(all_abs_diff, bins=20, color='red', alpha=0.7, edgecolor='black')
axes[1].axvline(mae, color='blue', linestyle='--', lw=2, label=f'MAE: {mae:.3f}')
axes[1].set_xlabel('Absolute Difference |VLM - Vision|', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title('Distribution of Disagreement', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

dist_file = output_dir / 'score_distributions.png'
plt.savefig(dist_file, dpi=150, bbox_inches='tight')
print(f"  Saved: {dist_file}")
plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================

results_file = output_dir / 'validation_results.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"  Saved: {results_file}")

# Save detailed comparison
comparison_file = output_dir / 'detailed_comparisons.json'
with open(comparison_file, 'w') as f:
    json.dump({
        'comparisons': comparison_data,
    }, f, indent=2)

print(f"  Saved: {comparison_file}")

# ============================================================================
# SUMMARY & RECOMMENDATION
# ============================================================================

print("\n" + "="*80)
print("RECOMMENDATION FOR TRAINING")
print("="*80)

# Score VLM reliability
reliability_score = 0

if pearson_r > 0.7:
    reliability_score += 3
    vlm_status = "EXCELLENT"
elif pearson_r > 0.6:
    reliability_score += 2
    vlm_status = "GOOD"
elif pearson_r > 0.5:
    reliability_score += 1
    vlm_status = "MODERATE"
else:
    vlm_status = "POOR"

if mae < 0.15:
    reliability_score += 2
elif mae < 0.25:
    reliability_score += 1

recommendation = ""
if reliability_score >= 5:
    recommendation = "USE VLM for training (with caution on per-behavior level)"
elif reliability_score >= 3:
    recommendation = "USE HYBRID: Vision primary, VLM secondary validation"
else:
    recommendation = "USE VISION METRIC ONLY - VLM unreliable for training"

print(f"\nVLM Correlation Quality: {vlm_status} (r={pearson_r:.3f})")
print(f"Agreement Quality: MAE={mae:.3f}")
print(f"Statistical Significance: p={pearson_p:.2e}")
print(f"\nRecommendation: {recommendation}")

print(f"\nFiles saved to: {output_dir}/")
print("="*80 + "\n")

