#!/usr/bin/env python3
"""
Per-behavior training fitness: best breakdown at final generation,
averaged across 30 seeds, for all 3 architectures.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
RESULTS_DIR = Path(__file__).parent / 'results'
OUT_DIR     = Path(__file__).parent

BEHAVIORS = ['stop moving', 'move slow', 'move fast', 'go slower', 'go faster']

# ── Load per-behavior final fitness ──────────────────────────────
def load_per_behavior(pattern, key_fn, n=30):
    """Returns dict: behavior -> list of scores across seeds."""
    scores = {b: [] for b in BEHAVIORS}
    for seed in range(n):
        p = RESULTS_DIR / pattern.format(seed=seed)
        if not p.exists():
            continue
        d = json.load(open(p))
        breakdown = key_fn(d)
        for b in BEHAVIORS:
            scores[b].append(breakdown.get(b, 0.0))
    return scores

sh_scores = load_per_behavior(
    'evo_single_head_seed{seed:03d}.json',
    lambda d: d['history'][-1]['best_breakdown']
)

mh_scores = load_per_behavior(
    'evo_cnn_multi_seed{seed:03d}.json',
    lambda d: d['history'][-1]['best_breakdown']
)

cl_scores = load_per_behavior(
    'evo_continual_seed{seed:03d}.json',
    lambda d: list(d['phases'].values())[-1]['per_task_final']
)

# ── Print summary ─────────────────────────────────────────────────
print(f"{'Behavior':<15} {'Single':>10} {'Multi':>10} {'Continual':>10}")
print("-" * 50)
for b in BEHAVIORS:
    s = np.mean(sh_scores[b])
    m = np.mean(mh_scores[b])
    c = np.mean(cl_scores[b])
    print(f"{b:<15} {s:>10.3f} {m:>10.3f} {c:>10.3f}")

# ── Plot ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

x      = np.arange(len(BEHAVIORS))
width  = 0.25
colors = ['orange', 'steelblue', 'lightgreen']
archs  = [('Single-Head', sh_scores), ('Multi-Head', mh_scores), ('Continual', cl_scores)]

for i, (label, scores) in enumerate(archs):
    means = [np.mean(scores[b]) for b in BEHAVIORS]
    stds  = [np.std(scores[b])  for b in BEHAVIORS]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, means, width,
                  yerr=stds, capsize=5,
                  label=label, color=colors[i], alpha=0.8,
                  edgecolor='black', linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels(BEHAVIORS, fontsize=11)
ax.set_ylabel('Best Training Score (avg 30 seeds)', fontsize=12)
ax.set_title('Per-Behavior Training Fitness: All 3 Architectures', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0, 1.15)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'plot_D_per_behavior_training.png', dpi=150, bbox_inches='tight')
print("\n[OK] plot_D_per_behavior_training.png")
plt.close()
