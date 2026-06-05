#!/usr/bin/env python3
"""
Visualize the train/test split by duration_ms.
Shows stratified sampling: every 5th batch (by duration) goes to test.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT_DIR = Path(__file__).parent

with open(OUT_DIR / 'train_archive.json') as f:
    train = json.load(f)
with open(OUT_DIR / 'test_archive.json') as f:
    test = json.load(f)

train_durations = sorted(d['duration_ms'] / 1000 for d in train.values())
test_durations  = sorted(d['duration_ms'] / 1000 for d in test.values())

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6),
                                gridspec_kw={'height_ratios': [2, 1]})

# ── Top panel: strip plot showing all batches ──
ax1.scatter(train_durations, [1] * len(train_durations),
            color='steelblue', s=80, alpha=0.8, label=f'Train ({len(train_durations)} batches)', zorder=3)
ax1.scatter(test_durations,  [1] * len(test_durations),
            color='tomato', s=80, marker='D', alpha=0.9, label=f'Test ({len(test_durations)} batches)', zorder=4)

ax1.set_yticks([])
ax1.set_xlabel('')
ax1.set_xlim(0, max(train_durations + test_durations) * 1.05)
ax1.set_title('Train / Test Split — Stratified by Duration', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(axis='x', alpha=0.3)
ax1.set_ylabel('Batches', fontsize=10)

# ── Bottom panel: histogram ──
bins = np.linspace(0, max(train_durations + test_durations) * 1.05, 20)
ax2.hist(train_durations, bins=bins, color='steelblue', alpha=0.7, label='Train')
ax2.hist(test_durations,  bins=bins, color='tomato',    alpha=0.7, label='Test')
ax2.set_xlabel('Duration (seconds)', fontsize=11)
ax2.set_ylabel('Count', fontsize=10)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'plot_train_test_split.png', dpi=150, bbox_inches='tight')
print("[OK] Saved: plot_train_test_split.png")
plt.close()
