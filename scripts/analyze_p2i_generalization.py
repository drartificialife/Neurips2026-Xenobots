#!/usr/bin/env python3
"""
Analyze P2I Generalization Test Results - Detailed Statistics.

Provides:
- Overall statistics
- Per-base-prompt breakdown
- Pass rates and distributions
- Performance summary

Usage:
    python scripts/analyze_p2i_generalization.py
"""

import json
from pathlib import Path
import numpy as np

RESULTS_FILE = Path('results/p2i_generalization_test.json')

def load_results():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Results not found: {RESULTS_FILE}")
    with open(RESULTS_FILE) as f:
        return json.load(f)

def print_header(title):
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}\n")

def analyze_results():
    data = load_results()
    results = data['results']

    # Overall stats
    print_header("OVERALL STATISTICS")
    print(f"Total Tests:           {data['total_tests']}")
    print(f"Mean Score:            {data['vlm_score_mean']:.4f}")
    print(f"Median Score:          {data['vlm_score_median']:.4f}")
    print(f"Std Dev:               {data['vlm_score_std']:.4f}")
    print(f"CV (Coeff. Variation): {data['vlm_score_cv']:.4f}")
    print(f"Score Range:           {data['min_score']:.2f} - {data['max_score']:.2f}")

    all_scores = [r['vlm_score'] for r in results]

    # Pass rates
    print_header("PASS RATES")
    excellent = sum(1 for s in all_scores if s > 0.9)
    good = sum(1 for s in all_scores if 0.7 < s <= 0.9)
    weak = sum(1 for s in all_scores if 0 < s <= 0.7)
    failed = sum(1 for s in all_scores if s == 0.0)

    print(f"Excellent (>0.9):      {excellent:2d}/{len(all_scores)}  ({100*excellent/len(all_scores):5.1f}%)")
    print(f"Good (0.7-0.9):        {good:2d}/{len(all_scores)}  ({100*good/len(all_scores):5.1f}%)")
    print(f"Weak (0-0.7):          {weak:2d}/{len(all_scores)}  ({100*weak/len(all_scores):5.1f}%)")
    print(f"Failed (0.0):          {failed:2d}/{len(all_scores)}  ({100*failed/len(all_scores):5.1f}%)")
    print(f"Passed (>0.0):         {len(all_scores)-failed:2d}/{len(all_scores)}  ({100*(len(all_scores)-failed)/len(all_scores):5.1f}%)")

    # Group by base prompt
    by_base = {}
    for r in results:
        base = r['base_prompt']
        if base not in by_base:
            by_base[base] = []
        by_base[base].append(r)

    print_header("BREAKDOWN BY BASE PROMPT")

    for base in sorted(by_base.keys()):
        tests = by_base[base]
        scores = [t['vlm_score'] for t in tests]

        passed = sum(1 for s in scores if s > 0)
        excellent = sum(1 for s in scores if s > 0.9)

        print(f"\n{base.upper()}")
        print(f"{'-'*80}")
        print(f"  Count:               {len(tests)}")
        print(f"  Mean:                {np.mean(scores):.4f}")
        print(f"  Median:              {np.median(scores):.4f}")
        print(f"  Std Dev:             {np.std(scores):.4f}")
        print(f"  Min-Max:             {np.min(scores):.2f} - {np.max(scores):.2f}")
        print(f"  Passed (>0):         {passed}/{len(tests)}  ({100*passed/len(tests):5.1f}%)")
        print(f"  Excellent (>0.9):    {excellent}/{len(tests)}  ({100*excellent/len(tests):5.1f}%)")
        print()
        print(f"  {'Prompt':<25} {'Score':>8}")
        print(f"  {'-'*35}")
        for t in sorted(tests, key=lambda x: -x['vlm_score']):
            status = "[EXCELLENT]" if t['vlm_score'] > 0.9 else "[GOOD]" if t['vlm_score'] > 0.7 else "[WEAK]" if t['vlm_score'] > 0 else "[FAIL]"
            print(f"  {t['prompt']:<25} {t['vlm_score']:>7.3f}  {status}")

    # Summary interpretation
    print_header("INTERPRETATION")
    mean = data['vlm_score_mean']
    std = data['vlm_score_std']

    if mean > 0.85 and std < 0.15:
        print("Status: [EXCELLENT]")
        print("The P2I networks show excellent generalization to unseen prompts.")
        print("High mean score and low variance indicate consistent, reliable performance.")
    elif mean > 0.75 and std < 0.25:
        print("Status: [GOOD]")
        print("The P2I networks show good generalization with some variance.")
        print("Most tests pass, but some unseen prompts challenge the networks.")
    elif mean > 0.60:
        print("Status: [ACCEPTABLE]")
        print("The P2I networks show moderate generalization.")
        print("Significant variance suggests some prompts don't transfer well.")
    else:
        print("Status: [NEEDS IMPROVEMENT]")
        print("The P2I networks show limited generalization.")
        print("Consider improving training data diversity or network architecture.")

    print()

if __name__ == '__main__':
    analyze_results()
