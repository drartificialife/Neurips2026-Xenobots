#!/usr/bin/env python3
"""
Compare training dynamics across 3 architectures (30 seeds each).

Plot A: Training curves — best fitness per generation (single-head vs multi-head)
Plot B: Final training fitness distribution — all 3 architectures (violin)
Plot C: Continual learning phase progression — fitness as tasks are added
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
RESULTS_DIR = Path(__file__).parent / 'results'
OUT_DIR     = Path(__file__).parent

# ── Load single-head ──────────────────────────────────────────────
sh_histories, sh_finals = [], []
for seed in range(30):
    p = RESULTS_DIR / f'evo_single_head_seed{seed:03d}.json'
    if p.exists():
        d = json.load(open(p))
        sh_histories.append([h['best'] for h in d['history']])
        sh_finals.append(d['best_fitness'])

# ── Load multi-head ───────────────────────────────────────────────
mh_histories, mh_finals = [], []
for seed in range(30):
    p = RESULTS_DIR / f'evo_cnn_multi_seed{seed:03d}.json'
    if p.exists():
        d = json.load(open(p))
        mh_histories.append([h['best'] for h in d['history']])
        mh_finals.append(max(d['history'], key=lambda x: x['best'])['best'])

# ── Load continual ────────────────────────────────────────────────
cl_finals, cl_phases = [], []
for seed in range(30):
    p = RESULTS_DIR / f'evo_continual_seed{seed:03d}.json'
    if p.exists():
        d = json.load(open(p))
        cl_finals.append(d['global_best_fitness'])
        cl_phases.append({k: v['final_fitness'] for k, v in d['phases'].items()})

print(f"Loaded: single={len(sh_finals)}, multi={len(mh_finals)}, continual={len(cl_finals)} seeds")
print(f"Single-Head final:  {np.mean(sh_finals):.4f} +/- {np.std(sh_finals):.4f}")
print(f"Multi-Head final:   {np.mean(mh_finals):.4f} +/- {np.std(mh_finals):.4f}")
print(f"Continual final:    {np.mean(cl_finals):.4f} +/- {np.std(cl_finals):.4f}")

# ── Align history lengths ─────────────────────────────────────────
min_gens_sh = min(len(h) for h in sh_histories)
min_gens_mh = min(len(h) for h in mh_histories)
sh_arr = np.array([h[:min_gens_sh] for h in sh_histories])
mh_arr = np.array([h[:min_gens_mh] for h in mh_histories])

# ════════════════════════════════════════════════════════════════════
# Plot A: Training curves
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 5))

def plot_curve(arr, color, label):
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    gens = np.arange(1, len(mean) + 1)
    ax.plot(gens, mean, color=color, linewidth=2, label=label)
    ax.fill_between(gens, mean - std, mean + std, color=color, alpha=0.15)

plot_curve(sh_arr, 'orange',  f'Single-Head (mean±std, n=30)')
plot_curve(mh_arr, 'steelblue', f'Multi-Head (mean±std, n=30)')

ax.set_xlabel('Generation', fontsize=12)
ax.set_ylabel('Best Fitness (training)', fontsize=12)
ax.set_title('Training Curves: Single-Head vs Multi-Head (30 seeds)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(OUT_DIR / 'plot_A_training_curves.png', dpi=150, bbox_inches='tight')
print("[OK] plot_A_training_curves.png")
plt.close()

# ════════════════════════════════════════════════════════════════════
# Plot B: Final training fitness — violin plot
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))

data   = [sh_finals, mh_finals, cl_finals]
labels = ['Single-Head', 'Multi-Head', 'Continual']
colors = ['orange', 'steelblue', 'lightgreen']

parts = ax.violinplot(data, positions=[1, 2, 3], showmedians=True, showextrema=True)
for i, (pc, color) in enumerate(zip(parts['bodies'], colors)):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)
parts['cmedians'].set_color('black')
parts['cmedians'].set_linewidth(2)

# Overlay individual points
for i, (vals, color) in enumerate(zip(data, colors)):
    x = np.random.normal(i + 1, 0.04, size=len(vals))
    ax.scatter(x, vals, color=color, s=20, alpha=0.5, zorder=3, edgecolors='black', linewidths=0.3)

# Mean labels
for i, vals in enumerate(data):
    ax.text(i + 1, max(vals) + 0.01, f'{np.mean(vals):.3f}',
            ha='center', fontsize=10, fontweight='bold')

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel('Best Training Fitness', fontsize=12)
ax.set_title('Final Training Fitness Distribution (30 seeds each)', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(OUT_DIR / 'plot_B_final_fitness.png', dpi=150, bbox_inches='tight')
print("[OK] plot_B_final_fitness.png")
plt.close()

# ════════════════════════════════════════════════════════════════════
# Plot C: Continual learning — fitness per phase across 30 seeds
# ════════════════════════════════════════════════════════════════════
phase_keys  = ['phase_1', 'phase_2', 'phase_3', 'phase_4', 'phase_5']
phase_labels = ['Phase 1\n(1 task)', 'Phase 2\n(2 tasks)', 'Phase 3\n(3 tasks)',
                'Phase 4\n(4 tasks)', 'Phase 5\n(5 tasks)']

phase_data = []
for pk in phase_keys:
    vals = [s[pk] for s in cl_phases if pk in s]
    phase_data.append(vals)

fig, ax = plt.subplots(figsize=(10, 5))

means = [np.mean(v) for v in phase_data]
stds  = [np.std(v)  for v in phase_data]
x     = np.arange(len(phase_keys))

ax.plot(x, means, color='lightgreen', linewidth=2.5, marker='o', markersize=8, zorder=3)
ax.fill_between(x, np.array(means) - np.array(stds),
                   np.array(means) + np.array(stds), color='lightgreen', alpha=0.25)

for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.01, f'{m:.3f}', ha='center', fontsize=10, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(phase_labels, fontsize=11)
ax.set_ylabel('Mean Fitness (across active tasks)', fontsize=12)
ax.set_title('Continual Learning: Fitness per Phase (30 seeds)', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(OUT_DIR / 'plot_C_continual_phases.png', dpi=150, bbox_inches='tight')
print("[OK] plot_C_continual_phases.png")
plt.close()
