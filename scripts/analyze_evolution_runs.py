#!/usr/bin/env python3
"""
Analyze 30 evolution runs: mean fitness curve + Wilcoxon test.

Usage:
    python scripts/analyze_evolution_runs.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wilcoxon

RESULTS_DIR = Path('results')

with open(Path('scripts/train_prompts.json')) as _f:
    PROMPTS = json.load(_f)


def load_all_histories():
    files = sorted(RESULTS_DIR.glob('p2i_evolution_history_seed*.json'))
    histories = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
            histories.append(data['history'])
    return histories


def main():
    histories = load_all_histories()
    n_runs = len(histories)
    n_gens = len(histories[0])
    print(f"Loaded {n_runs} runs, {n_gens} generations each\n")

    # Extract best fitness per generation per run: shape (n_runs, n_gens)
    best_matrix = np.array([[g['best'] for g in h] for h in histories])
    mean_matrix = np.array([[g['mean'] for g in h] for h in histories])

    # Per-prompt best: shape (n_runs, n_gens, n_prompts)
    prompt_matrix = {}
    for prompt in PROMPTS:
        prompt_matrix[prompt] = np.array([
            [g['best_breakdown'][prompt] for g in h] for h in histories
        ])

    gens = np.arange(1, n_gens + 1)

    # --- Wilcoxon test: gen 1 vs gen 100 ---
    first_gen = best_matrix[:, 0]
    last_gen = best_matrix[:, -1]

    print("=" * 60)
    print("Wilcoxon Signed-Rank Test: Gen 1 vs Gen 100 (best fitness)")
    print("=" * 60)
    print(f"  Gen  1: mean={first_gen.mean():.4f} ± {first_gen.std():.4f}")
    print(f"  Gen {n_gens:3d}: mean={last_gen.mean():.4f} ± {last_gen.std():.4f}")

    stat, p_value = wilcoxon(first_gen, last_gen, alternative='less')
    print(f"  Wilcoxon stat={stat:.1f}, p={p_value:.6f}")
    if p_value < 0.05:
        print(f"  => Significant (p < 0.05): evolution DOES drive fitness forward")
    else:
        print(f"  => NOT significant (p >= 0.05)")

    # Per-prompt Wilcoxon
    print(f"\nPer-prompt Wilcoxon (gen 1 vs gen {n_gens}):")
    for prompt in PROMPTS:
        first_p = prompt_matrix[prompt][:, 0]
        last_p = prompt_matrix[prompt][:, -1]
        try:
            stat_p, p_val_p = wilcoxon(first_p, last_p, alternative='less')
        except ValueError:
            stat_p, p_val_p = 0, 1.0
        sig = "*" if p_val_p < 0.05 else ""
        print(f"  {prompt:14s}: {first_p.mean():.3f} → {last_p.mean():.3f}  "
              f"p={p_val_p:.4f} {sig}")

    # --- Plot 1: Overall fitness (mean ± std across runs) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    best_mean = best_matrix.mean(axis=0)
    best_std = best_matrix.std(axis=0)
    mean_mean = mean_matrix.mean(axis=0)
    mean_std = mean_matrix.std(axis=0)

    ax1.plot(gens, best_mean, 'b-', linewidth=2, label='Best (across runs)')
    ax1.fill_between(gens, best_mean - best_std, best_mean + best_std,
                      alpha=0.2, color='blue')
    ax1.plot(gens, mean_mean, 'g--', linewidth=1, alpha=0.7, label='Pop mean (across runs)')
    ax1.fill_between(gens, mean_mean - mean_std, mean_mean + mean_std,
                      alpha=0.15, color='green')

    ax1.set_ylabel('Fitness', fontsize=12)
    ax1.set_title(f'P2I Evolution — {n_runs} Runs (mean ± std)\n'
                  f'Wilcoxon p={p_value:.4f}', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Per-prompt fitness ---
    colors = plt.cm.tab10(np.linspace(0, 1, len(PROMPTS)))
    for prompt, color in zip(PROMPTS, colors):
        pm = prompt_matrix[prompt]
        pm_mean = pm.mean(axis=0)
        pm_std = pm.std(axis=0)
        ax2.plot(gens, pm_mean, '-', color=color, linewidth=1.5, label=prompt)
        ax2.fill_between(gens, pm_mean - pm_std, pm_mean + pm_std,
                          alpha=0.1, color=color)

    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Fitness', fontsize=12)
    ax2.set_title('Per-Prompt Best Fitness (mean ± std)', fontsize=14)
    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, borderaxespad=0)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = RESULTS_DIR / 'p2i_evolution_30runs_summary.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nPlot saved to: {out_path}")

    # Save summary JSON
    summary = {
        'n_runs': n_runs,
        'n_generations': n_gens,
        'overall': {
            'gen1_mean': float(first_gen.mean()),
            'gen1_std': float(first_gen.std()),
            f'gen{n_gens}_mean': float(last_gen.mean()),
            f'gen{n_gens}_std': float(last_gen.std()),
            'wilcoxon_stat': float(stat),
            'wilcoxon_p': float(p_value),
        },
        'per_prompt': {
            prompt: {
                'gen1_mean': float(prompt_matrix[prompt][:, 0].mean()),
                f'gen{n_gens}_mean': float(prompt_matrix[prompt][:, -1].mean()),
            }
            for prompt in PROMPTS
        }
    }
    summary_path = RESULTS_DIR / 'p2i_evolution_30runs_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
