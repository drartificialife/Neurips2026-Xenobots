#!/usr/bin/env python3
"""
validate_vlm_vs_cv.py

Validate whether VLM scores agree with CV kinematics.

QUESTION: Is the VLM score a reliable proxy for actual behavior?

For each batch in the archive, we have:
  - VLM score per behavior   (archive_vlm_scores_v2.json)  — subjective AI judgment
  - CV kinematics            (kinematics_cache.json)        — objective measurement

We compare them to answer:
  1. Correlation: does a higher VLM score → CV says behavior occurred?
  2. ROC AUC: can VLM score predict CV binary outcome?
  3. At what VLM threshold does CV label flip from 0 → 1?
  4. Where do they DISAGREE? (most informative for paper)

WHY THIS MATTERS
----------------
If VLM and CV agree well → training on VLM scores is valid proxy for real behavior.
If they disagree         → VLM score is unreliable; CV is the correct metric.
Either way, this gives us a defensible answer to reviewers asking about circularity.

OUTPUT
------
  - Console: per-behavior statistics table
  - paper/figures/vlm_vs_cv_correlation.png   — scatter + ROC per behavior
  - paper/vlm_cv_agreement.json               — numerical results
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pointbiserialr
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

ARCHIVE_JSON    = PROJECT_ROOT / 'cache' / 'archive_vlm_scores_v2.json'
KINEMATICS_JSON = PROJECT_ROOT / 'cache' / 'kinematics_cache.json'
FIGURES_DIR     = PROJECT_ROOT / 'paper' / 'figures'
PAPER_DIR       = PROJECT_ROOT / 'paper'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# CV thresholds — must match characterize_archive.py
STOP_THRESHOLD = 0.02
SLOW_THRESHOLD = 0.08
FAST_THRESHOLD = 0.20
CHANGE_FRAC    = 0.15


def cv_label(k: dict, behavior: str) -> int:
    """Return CV binary label (1 = behavior occurred, 0 = did not)."""
    pre  = k['pre_speed']
    post = k['post_speed']

    if behavior == 'stop moving':
        return 1 if post < STOP_THRESHOLD else 0
    elif behavior == 'move slow':
        return 1 if post < SLOW_THRESHOLD else 0
    elif behavior == 'move fast':
        return 1 if post > FAST_THRESHOLD else 0
    elif behavior == 'go slower':
        if pre == 0.0:
            return 1 if post == 0.0 else 0
        return 1 if post < pre * (1.0 - CHANGE_FRAC) else 0
    elif behavior == 'go faster':
        if pre == 0.0:
            return 1 if post > STOP_THRESHOLD else 0
        return 1 if post > pre * (1.0 + CHANGE_FRAC) else 0
    return -1


def main():
    # Load data
    with open(ARCHIVE_JSON) as f:
        archive = json.load(f)
    with open(KINEMATICS_JSON) as f:
        kinematics = json.load(f)

    behaviors = ['stop moving', 'move slow', 'move fast', 'go slower', 'go faster']

    # Build per-behavior paired (vlm_score, cv_label) lists
    paired = {b: {'vlm': [], 'cv': [], 'batch_ids': []} for b in behaviors}

    for batch_id, vlm_data in archive.items():
        if vlm_data.get('duration_ms') is None:
            continue
        k = kinematics.get(batch_id)
        if k is None or 'error' in k:
            continue  # no CV data for this batch

        for behavior in behaviors:
            bdata = vlm_data.get(behavior)
            if bdata is None:
                continue
            vlm_score = bdata.get('score')
            if vlm_score is None:
                continue

            cv = cv_label(k, behavior)
            paired[behavior]['vlm'].append(float(vlm_score))
            paired[behavior]['cv'].append(cv)
            paired[behavior]['batch_ids'].append(batch_id)

    # ── Statistics per behavior ───────────────────────────────────
    print(f"\n{'='*75}")
    print("VLM Score vs CV Kinematics — Correlation Analysis")
    print(f"{'='*75}")
    print(f"\n{'Behavior':<14} {'N':>4} {'CV=1':>6} {'Spearman r':>11} {'p-val':>8} "
          f"{'ROC AUC':>8} {'Mean VLM|CV=1':>14} {'Mean VLM|CV=0':>14}")
    print(f"{'-'*75}")

    results = {}
    for behavior in behaviors:
        vlm = np.array(paired[behavior]['vlm'])
        cv  = np.array(paired[behavior]['cv'])
        n   = len(vlm)

        if n < 5:
            print(f"{behavior:<14}  N too small ({n})")
            continue

        n_pos = int(cv.sum())
        n_neg = n - n_pos

        # Spearman correlation: VLM score vs CV binary label
        rho, pval = spearmanr(vlm, cv)

        # ROC AUC: does VLM score discriminate CV=1 from CV=0?
        if n_pos > 0 and n_neg > 0:
            auc = roc_auc_score(cv, vlm)
        else:
            auc = float('nan')

        # Mean VLM score conditioned on CV label
        mean_vlm_cv1 = float(vlm[cv == 1].mean()) if n_pos > 0 else float('nan')
        mean_vlm_cv0 = float(vlm[cv == 0].mean()) if n_neg > 0 else float('nan')

        sig = '*' if pval < 0.05 else (' ' if pval < 0.1 else ' ')
        print(f"{behavior:<14} {n:>4} {n_pos:>6}  {rho:>+9.3f}{sig}  {pval:>8.4f}  "
              f"{auc:>7.3f}   {mean_vlm_cv1:>13.3f}   {mean_vlm_cv0:>13.3f}")

        results[behavior] = {
            'n_pairs':        n,
            'n_cv_positive':  n_pos,
            'spearman_r':     float(rho),
            'spearman_p':     float(pval),
            'roc_auc':        float(auc) if not np.isnan(auc) else None,
            'mean_vlm_cv1':   float(mean_vlm_cv1) if not np.isnan(mean_vlm_cv1) else None,
            'mean_vlm_cv0':   float(mean_vlm_cv0) if not np.isnan(mean_vlm_cv0) else None,
        }

    # ── Disagreement analysis ─────────────────────────────────────
    print(f"\n{'='*75}")
    print("Disagreement cases (VLM high but CV=0, or VLM low but CV=1)")
    print(f"{'='*75}")
    for behavior in behaviors:
        vlm = np.array(paired[behavior]['vlm'])
        cv  = np.array(paired[behavior]['cv'])
        ids = paired[behavior]['batch_ids']
        if len(vlm) == 0:
            continue

        # VLM says yes (>0.5) but CV says no
        false_pos = [(ids[i], vlm[i]) for i in range(len(vlm))
                     if vlm[i] >= 0.5 and cv[i] == 0]
        # VLM says no (<0.5) but CV says yes
        false_neg = [(ids[i], vlm[i]) for i in range(len(vlm))
                     if vlm[i] < 0.5  and cv[i] == 1]

        total = len(vlm)
        print(f"\n{behavior} (N={total}):")
        print(f"  VLM>=0.5 but CV=0 (VLM overestimates): {len(false_pos):>3} "
              f"({100*len(false_pos)/total:.0f}%)")
        print(f"  VLM<0.5  but CV=1 (VLM underestimates): {len(false_neg):>3} "
              f"({100*len(false_neg)/total:.0f}%)")

    # ── Plot: scatter + ROC ───────────────────────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes_scatter = axes[0]
    axes_roc     = axes[1]

    for col, behavior in enumerate(behaviors):
        vlm = np.array(paired[behavior]['vlm'])
        cv  = np.array(paired[behavior]['cv'])
        n   = len(vlm)
        if n == 0:
            continue

        # Row 0: VLM score distribution, split by CV label
        ax = axes_scatter[col]
        cv0 = vlm[cv == 0]
        cv1 = vlm[cv == 1]
        ax.hist(cv0, bins=10, range=(0, 1), alpha=0.6, color='tomato',
                label=f'CV=0 (n={len(cv0)})', density=True)
        ax.hist(cv1, bins=10, range=(0, 1), alpha=0.6, color='steelblue',
                label=f'CV=1 (n={len(cv1)})', density=True)
        ax.axvline(0.5, color='black', linestyle='--', linewidth=1, label='VLM=0.5')
        r = results.get(behavior, {})
        ax.set_title(f"{behavior}\nρ={r.get('spearman_r', 0):.2f}  "
                     f"AUC={r.get('roc_auc', 0):.2f}", fontsize=9)
        ax.set_xlabel('VLM score', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

        # Row 1: ROC curve
        ax2 = axes_roc[col]
        n_pos = int(cv.sum())
        n_neg = n - n_pos
        if n_pos > 0 and n_neg > 0:
            fpr, tpr, _ = roc_curve(cv, vlm)
            auc_val = results.get(behavior, {}).get('roc_auc', 0)
            ax2.plot(fpr, tpr, color='steelblue', linewidth=1.5,
                     label=f'AUC={auc_val:.2f}')
            ax2.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random')
            ax2.set_xlabel('False Positive Rate', fontsize=8)
            ax2.set_ylabel('True Positive Rate', fontsize=8)
            ax2.legend(fontsize=7)
        else:
            ax2.text(0.5, 0.5, 'Not enough\nCV positives',
                     ha='center', va='center', fontsize=9)
        ax2.set_title(f'ROC — {behavior}', fontsize=9)
        ax2.tick_params(labelsize=7)

    fig.suptitle('VLM Score vs CV Kinematics: Agreement Analysis\n'
                 'Top: VLM score distribution split by CV label | '
                 'Bottom: ROC curve (VLM as predictor of CV)',
                 fontsize=11)
    fig.tight_layout()
    out = FIGURES_DIR / 'vlm_vs_cv_correlation.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out}")

    # Save JSON results
    out_json = PAPER_DIR / 'vlm_cv_agreement.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_json}")
    print("\n[OK] validate_vlm_vs_cv.py complete")


if __name__ == '__main__':
    main()
