#!/usr/bin/env python3
"""
P2I Generalization Results - Visualization and Summary.

Shows:
1. Box plot (distribution) by base prompt
2. Summary statistics
3. Text table summary

Usage:
    python scripts/plot_p2i_generalization.py
"""

import json
from pathlib import Path
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt

# Paths
RESULTS_DIR = Path('results')
RESULTS_FILE = RESULTS_DIR / 'p2i_generalization_test.json'
OUTPUT_IMAGE = RESULTS_DIR / 'p2i_generalization_results.png'

def load_results() -> Dict:
    """Load test results."""
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Results not found: {RESULTS_FILE}")
    with open(RESULTS_FILE) as f:
        return json.load(f)


def create_visualization(data: Dict):
    """Create and save visualization of generalization results."""
    results = data['results']

    # Group by base prompt
    by_base = {}
    for r in results:
        base = r['base_prompt']
        if base not in by_base:
            by_base[base] = []
        by_base[base].append(r)

    base_prompts = sorted(by_base.keys())

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('P2I Generalization Test Results', fontsize=18, fontweight='bold', y=0.98)

    # 1. Box plot by base prompt (distribution)
    ax1 = plt.subplot(1, 2, 1)
    data_by_base = [by_base[base] for base in base_prompts]
    scores_by_base = [[t['vlm_score'] for t in tests] for tests in data_by_base]

    color_map = {
        'slow down': '#FF6B6B',
        'stop moving': '#FFA500',
        'go fast': '#4ECDC4',
        'move faster': '#45B7D1'
    }

    bp = ax1.boxplot(scores_by_base, tick_labels=base_prompts, patch_artist=True)
    for patch, base in zip(bp['boxes'], base_prompts):
        patch.set_facecolor(color_map[base])
        patch.set_alpha(0.7)

    ax1.set_ylabel('VLM Score', fontsize=12, fontweight='bold')
    ax1.set_title('Score Distribution by Base Prompt', fontsize=13, fontweight='bold')
    ax1.set_ylim([-0.05, 1.1])
    ax1.grid(axis='y', alpha=0.3)

    # 2. Summary table
    ax2 = plt.subplot(1, 2, 2)
    ax2.axis('off')

    all_scores_list = [r['vlm_score'] for r in results]
    all_pass = sum(1 for s in all_scores_list if s > 0)
    all_total = len(all_scores_list)

    # Build summary text
    summary_lines = [
        "OVERALL SUMMARY",
        "=" * 40,
        "",
        f"Total Tests: {all_total}",
        f"Passed (>0.0): {all_pass}/{all_total} ({100*all_pass/all_total:.1f}%)",
        "",
        "Statistics:",
        f"  Mean: {data['vlm_score_mean']:.4f}",
        f"  Median: {data['vlm_score_median']:.4f}",
        f"  Std Dev: {data['vlm_score_std']:.4f}",
        f"  Range: {data['min_score']:.2f} - {data['max_score']:.2f}",
        "",
        "BY BASE PROMPT:",
        "-" * 40,
    ]

    for base in base_prompts:
        tests = by_base[base]
        scores = [t['vlm_score'] for t in tests]
        mean = np.mean(scores)
        pass_count = sum(1 for s in scores if s > 0)
        pass_rate = f"{100*pass_count/len(tests):.0f}%"
        summary_lines.append(f"{base:<18} {len(tests)} tests, {mean:.3f} mean, {pass_rate} pass")

    summary_text = "\n".join(summary_lines)

    ax2.text(0.05, 0.95, summary_text, fontsize=11, family='monospace',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches='tight')
    print(f"\n[SUCCESS] Visualization saved to: {OUTPUT_IMAGE}\n")


def print_text_summary(data: Dict):
    """Print text summary to console."""
    results = data['results']

    # Group by base prompt
    by_base = {}
    for r in results:
        base = r['base_prompt']
        if base not in by_base:
            by_base[base] = []
        by_base[base].append(r)

    base_prompts = sorted(by_base.keys())

    # Print header
    print("=" * 80)
    print("P2I GENERALIZATION TEST RESULTS")
    print("=" * 80)

    # Print results by base prompt
    for base in base_prompts:
        tests = by_base[base]
        scores = [t['vlm_score'] for t in tests]

        print(f"\n{base.upper()}")
        print("-" * 80)
        print(f"{'Prompt':<28} | {'Score':>6} | {'Status':<15}")
        print("-" * 80)

        # Table rows
        for t in tests:
            if t['vlm_score'] > 0.5:
                status = f"[PASS] ({t['vlm_score']:.2f})"
            elif t['vlm_score'] > 0:
                status = f"[WEAK] ({t['vlm_score']:.2f})"
            else:
                status = f"[FAIL]"
            print(f"{t['prompt']:<28} | {t['vlm_score']:>6.2f} | {status:<15}")

        # Base summary
        mean = np.mean(scores)
        min_s = min(scores)
        max_s = max(scores)
        pass_count = sum(1 for s in scores if s > 0)
        print("-" * 80)
        print(f"{'Summary':<28} | {mean:>6.2f} | Passed: {pass_count}/{len(tests)}, Min: {min_s:.2f}, Max: {max_s:.2f}")

    # Overall stats
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    all_scores = [r['vlm_score'] for r in results]
    all_pass = sum(1 for s in all_scores if s > 0)
    all_total = len(all_scores)

    print(f"\nTotal Tests:          {all_total}")
    print(f"Tests Passed (>0.0):  {all_pass}/{all_total} ({100*all_pass/all_total:.1f}%)")
    print(f"\nStatistics:")
    print(f"  Mean Score:         {data['vlm_score_mean']:.4f}")
    print(f"  Median Score:       {data['vlm_score_median']:.4f}")
    print(f"  Std Dev:            {data['vlm_score_std']:.4f}")
    print(f"  Range:              {data['min_score']:.2f} - {data['max_score']:.2f}")

    # Score by base prompt table
    print(f"\n{'-'*80}")
    print(f"{'Base Prompt':<20} | {'# Tests':>8} | {'Mean':>6} | {'Range':<15}")
    print(f"{'-'*80}")

    for base in base_prompts:
        tests = by_base[base]
        scores = [t['vlm_score'] for t in tests]
        mean = np.mean(scores)
        min_s = min(scores)
        max_s = max(scores)
        print(f"{base:<20} | {len(tests):>8} | {mean:>6.3f} | {min_s:.2f} - {max_s:.2f}")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    data = load_results()

    # Create visualization
    create_visualization(data)

    # Print text summary
    print_text_summary(data)
