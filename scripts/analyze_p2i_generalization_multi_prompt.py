#!/usr/bin/env python3
"""
Analyze Multi-Prompt P2I Generalization Results.

Compares single-prompt vs multi-prompt performance.

Usage:
    python scripts/analyze_p2i_generalization_multi_prompt.py
"""

import json
from pathlib import Path
import numpy as np

SINGLE_RESULTS = Path('results/p2i_generalization_test.json')
MULTI_RESULTS = Path('results/p2i_generalization_test_multi_prompt.json')


def load_results(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def print_header(title):
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}\n")


def analyze():
    single_data = load_results(SINGLE_RESULTS)
    multi_data = load_results(MULTI_RESULTS)

    if not single_data:
        print("Single-prompt results not found!")
        return
    if not multi_data:
        print("Multi-prompt results not found!")
        return

    print_header("COMPARISON: Single-Prompt vs Multi-Prompt P2I Networks")

    # Overall stats
    print("OVERALL STATISTICS")
    print("-" * 80)
    print(f"{'Metric':<30} {'Single-Prompt':>20} {'Multi-Prompt':>20}")
    print("-" * 80)

    metrics = [
        ('Total Tests', lambda d: d['total_tests']),
        ('Mean Score', lambda d: f"{d['vlm_score_mean']:.4f}"),
        ('Median Score', lambda d: f"{d['vlm_score_median']:.4f}"),
        ('Std Dev', lambda d: f"{d['vlm_score_std']:.4f}"),
        ('CV (σ/μ)', lambda d: f"{d['vlm_score_cv']:.4f}"),
        ('Score Range', lambda d: f"{d['min_score']:.2f} - {d['max_score']:.2f}"),
    ]

    for metric_name, metric_fn in metrics:
        single_val = metric_fn(single_data)
        multi_val = metric_fn(multi_data)
        print(f"{metric_name:<30} {str(single_val):>20} {str(multi_val):>20}")

    # Pass rates
    print_header("PASS RATES")

    def count_passes(data, threshold=0.0):
        scores = [r['vlm_score'] for r in data['results']]
        return sum(1 for s in scores if s > threshold), len(scores)

    excellent_single, total = count_passes(single_data, 0.9)
    excellent_multi, _ = count_passes(multi_data, 0.9)

    good_single, _ = count_passes(single_data, 0.7)
    good_multi, _ = count_passes(multi_data, 0.7)

    passed_single, _ = count_passes(single_data, 0.0)
    passed_multi, _ = count_passes(multi_data, 0.0)

    print(f"{'Metric':<30} {'Single-Prompt':>20} {'Multi-Prompt':>20}")
    print("-" * 80)
    print(f"{'Excellent (>0.9)':<30} {excellent_single}/{total:<19} {excellent_multi}/{total:<19}")
    print(f"{'Good (>0.7)':<30} {good_single}/{total:<19} {good_multi}/{total:<19}")
    print(f"{'Passed (>0.0)':<30} {passed_single}/{total:<19} {passed_multi}/{total:<19}")

    # Per base prompt breakdown
    print_header("PER BASE-PROMPT BREAKDOWN")

    def group_by_base(data):
        by_base = {}
        for r in data['results']:
            base = r['base_prompt']
            if base not in by_base:
                by_base[base] = []
            by_base[base].append(r)
        return by_base

    single_by_base = group_by_base(single_data)
    multi_by_base = group_by_base(multi_data)

    for base in sorted(single_by_base.keys()):
        single_tests = single_by_base[base]
        multi_tests = multi_by_base[base]

        single_scores = [t['vlm_score'] for t in single_tests]
        multi_scores = [t['vlm_score'] for t in multi_tests]

        single_pass = sum(1 for s in single_scores if s > 0)
        multi_pass = sum(1 for s in multi_scores if s > 0)

        print(f"\n{base.upper()}")
        print("-" * 80)
        print(f"{'Metric':<30} {'Single':>20} {'Multi':>20}")
        print("-" * 80)
        print(f"{'Mean':<30} {np.mean(single_scores):>20.4f} {np.mean(multi_scores):>20.4f}")
        print(f"{'Median':<30} {np.median(single_scores):>20.4f} {np.median(multi_scores):>20.4f}")
        print(f"{'Std Dev':<30} {np.std(single_scores):>20.4f} {np.std(multi_scores):>20.4f}")
        print(f"{'Passed':<30} {single_pass}/{len(single_scores):<19} {multi_pass}/{len(multi_scores):<19}")

    # Summary
    print_header("SUMMARY")

    single_mean = single_data['vlm_score_mean']
    multi_mean = multi_data['vlm_score_mean']
    improvement = ((multi_mean - single_mean) / single_mean) * 100

    print(f"Mean Score Improvement: {improvement:+.2f}%")

    if multi_mean > single_mean:
        print(f"Status: Multi-prompt performs BETTER")
    elif multi_mean < single_mean:
        print(f"Status: Multi-prompt performs WORSE")
    else:
        print(f"Status: Multi-prompt performs SAME")

    print()


if __name__ == '__main__':
    analyze()
