#!/usr/bin/env python3
"""
spot_check_cv.py

Visual spot-check: show trajectory heatmaps alongside CV labels and VLM scores.

PURPOSE
-------
Before trusting CV kinematics as ground truth, we need to verify that:
  1. The tracking correctly found the bot (not a bubble)
  2. The behavioral labels match what a human would see in the image

This script creates a grid of trajectory images annotated with:
  - CV label (computed from kinematics)
  - VLM score (from archive)
  - Pre/post speed values

Cases are sorted to highlight DISAGREEMENTS first (most useful for spotting errors).

OUTPUT
------
  paper/figures/spot_check_<behavior>.png  — grid of 20 cases per behavior
  Console: summary of how many cases to visually inspect

USAGE
-----
  python scripts/spot_check_cv.py                   # all behaviors
  python scripts/spot_check_cv.py --behavior "go slower"
  python scripts/spot_check_cv.py --n 20            # show 20 cases per behavior
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PROJECT_ROOT    = Path(__file__).parent.parent
ARCHIVE_JSON    = PROJECT_ROOT / 'cache' / 'archive_vlm_scores_v2.json'
KINEMATICS_JSON = PROJECT_ROOT / 'cache' / 'kinematics_cache.json'
HEATMAP_DIR     = PROJECT_ROOT / 'bot_trajectory'
FIGURES_DIR     = PROJECT_ROOT / 'paper' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Must match characterize_archive.py
STOP_THRESHOLD = 0.02
SLOW_THRESHOLD = 0.08
FAST_THRESHOLD = 0.20
CHANGE_FRAC    = 0.15


def cv_label(k: dict, behavior: str) -> int:
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--behavior', type=str, default=None,
                        help='Single behavior to check (default: all 5)')
    parser.add_argument('--n', type=int, default=20,
                        help='Cases per behavior to show (default 20)')
    parser.add_argument('--sort', choices=['disagree', 'cv1', 'cv0', 'random'],
                        default='disagree',
                        help='Sort order: disagree=VLM/CV conflict first (default)')
    args = parser.parse_args()

    with open(ARCHIVE_JSON) as f:
        archive = json.load(f)
    with open(KINEMATICS_JSON) as f:
        kinematics = json.load(f)

    behaviors = ([args.behavior] if args.behavior
                 else ['stop moving', 'move slow', 'move fast', 'go slower', 'go faster'])

    for behavior in behaviors:
        print(f"\n--- {behavior} ---")

        # Collect all cases with both CV and VLM data AND a heatmap image
        cases = []
        for batch_id, vlm_data in archive.items():
            if vlm_data.get('duration_ms') is None:
                continue
            k = kinematics.get(batch_id)
            if k is None or 'error' in k:
                continue

            bdata = vlm_data.get(behavior)
            if bdata is None:
                continue
            vlm_score = bdata.get('score')
            if vlm_score is None:
                continue

            # Check heatmap exists
            heatmap_path = HEATMAP_DIR / f"{batch_id}_trajectory_heatmap.png"
            if not heatmap_path.exists():
                # Try subdirectory
                heatmap_path = HEATMAP_DIR / 'New folder' / f"{batch_id}_trajectory_heatmap.png"
                if not heatmap_path.exists():
                    continue

            cv = cv_label(k, behavior)
            pre  = k['pre_speed']
            post = k['post_speed']

            # Disagreement score: high = VLM and CV strongly disagree
            # VLM says 1 but CV says 0, or VLM says 0 but CV says 1
            if cv == 1:
                disagree_score = 1.0 - vlm_score  # VLM underestimates
            else:
                disagree_score = vlm_score          # VLM overestimates

            cases.append({
                'batch_id':      batch_id,
                'vlm_score':     float(vlm_score),
                'cv_label':      cv,
                'pre_speed':     float(pre),
                'post_speed':    float(post),
                'disagree_score': disagree_score,
                'heatmap_path':  heatmap_path,
                'duration_ms':   vlm_data['duration_ms'],
            })

        if not cases:
            print(f"  No cases with heatmaps found.")
            continue

        # Sort cases
        if args.sort == 'disagree':
            cases.sort(key=lambda c: -c['disagree_score'])
            title_suffix = "sorted by VLM↔CV disagreement (worst first)"
        elif args.sort == 'cv1':
            cases = [c for c in cases if c['cv_label'] == 1]
            cases.sort(key=lambda c: c['vlm_score'])
            title_suffix = "CV=1 only, sorted by VLM score ascending"
        elif args.sort == 'cv0':
            cases = [c for c in cases if c['cv_label'] == 0]
            cases.sort(key=lambda c: -c['vlm_score'])
            title_suffix = "CV=0 only, sorted by VLM score descending"
        else:
            import random
            random.shuffle(cases)
            title_suffix = "random sample"

        selected = cases[:args.n]
        n_show   = len(selected)
        n_cols   = 5
        n_rows   = (n_show + n_cols - 1) // n_cols

        print(f"  Total cases: {len(cases)} | Showing: {n_show} ({args.sort})")

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 3.5, n_rows * 3.2))
        axes = np.array(axes).reshape(-1) if n_rows > 1 else np.array(axes).reshape(-1)

        for i, case in enumerate(selected):
            ax = axes[i]
            try:
                img = mpimg.imread(str(case['heatmap_path']))
                ax.imshow(img)
            except Exception:
                ax.set_facecolor('#eee')
                ax.text(0.5, 0.5, 'image\nnot found',
                        ha='center', va='center', transform=ax.transAxes)

            # Color border: green = agree, red = disagree
            vlm_says = 1 if case['vlm_score'] >= 0.5 else 0
            agree    = (vlm_says == case['cv_label'])
            border_color = '#2ca02c' if agree else '#d62728'
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)

            # Annotation
            cv_str  = 'CV=1 [YES]' if case['cv_label'] == 1 else 'CV=0 [NO]'
            vlm_str = f"VLM={case['vlm_score']:.2f}"
            speed_str = f"pre={case['pre_speed']:.3f} post={case['post_speed']:.3f}"
            agree_str = '[agree]' if agree else '[DISAGREE]'

            ax.set_title(
                f"{case['batch_id']}\n"
                f"{cv_str}  {vlm_str}\n"
                f"{speed_str}\n"
                f"{agree_str}",
                fontsize=6.5, pad=2,
                color='#2ca02c' if agree else '#d62728'
            )
            ax.axis('off')

        # Hide unused axes
        for i in range(n_show, len(axes)):
            axes[i].axis('off')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ca02c', label='VLM and CV agree'),
            Patch(facecolor='#d62728', label='VLM and CV DISAGREE'),
        ]
        fig.legend(handles=legend_elements, loc='lower center',
                   ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.0))

        fig.suptitle(
            f'Spot-Check: "{behavior}"\n{title_suffix}\n'
            f'Green border = agree | Red border = DISAGREE | '
            f'[Look at red borders to verify CV is correct]',
            fontsize=10, y=1.01
        )
        fig.tight_layout()

        bname = behavior.replace(' ', '_')
        out   = FIGURES_DIR / f'spot_check_{bname}.png'
        fig.savefig(out, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out}")

        # Print top disagreements to console for quick check
        print(f"\n  Top 5 disagreements ({behavior}):")
        print(f"  {'Batch':<16} {'CV':>4} {'VLM':>6} {'pre_spd':>9} {'post_spd':>9}")
        print(f"  {'-'*50}")
        for c in cases[:5]:
            print(f"  {c['batch_id']:<16} {c['cv_label']:>4}  {c['vlm_score']:>5.2f}  "
                  f"{c['pre_speed']:>8.4f}  {c['post_speed']:>8.4f}")

    print(f"\n[OK] Open paper/figures/spot_check_*.png and visually inspect RED borders.")
    print("     If CV label looks wrong for many red cases → adjust thresholds.")
    print("     If CV label looks RIGHT → VLM is the noisy one.")


if __name__ == '__main__':
    main()
