#!/usr/bin/env python3
"""
characterize_archive.py

Compute objective CV kinematics for every batch in the archive.
Uses template-matching bot tracking (same as generate_trajectory_heatmap.py)
to extract pre/post distances and speeds, then assigns behavioral labels.

Outputs:
  - cache/kinematics_cache.json  : per-batch kinematics + labels
  - paper/dataset_table.csv       : summary table for paper
  - paper/figures/dataset_speed_distribution.png
  - paper/figures/dataset_behavior_counts.png

Usage:
  python scripts/characterize_archive.py
  python scripts/characterize_archive.py --step 5   # sample every 5th frame (faster)
  python scripts/characterize_archive.py --recalc   # ignore cache, recompute all
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from parse_interventions import load_batch_with_interventions

CACHE_FILE   = PROJECT_ROOT / 'cache' / 'kinematics_cache.json'
ARCHIVE_JSON = PROJECT_ROOT / 'cache' / 'archive_vlm_scores_v2.json'
FIGURES_DIR  = PROJECT_ROOT / 'paper' / 'figures'
PAPER_DIR    = PROJECT_ROOT / 'paper'


# ── Thresholds (pixels/frame at scale=0.5) ───────────────────────
# Defined after first pass looking at distribution — tune if needed.
STOP_THRESHOLD   = 0.02   # post_speed < this → "stop moving"
SLOW_THRESHOLD   = 0.08   # post_speed in [STOP, SLOW) → "move slow"
FAST_THRESHOLD   = 0.20   # post_speed > this → "move fast"
CHANGE_FRACTION  = 0.15   # relative change for go_slower / go_faster


# ── CV tracking ───────────────────────────────────────────────────
def _load_gray(image_path: Path, scale: float) -> Optional[np.ndarray]:
    """Load image as grayscale, optionally downscaled."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        new_w = max(1, int(gray.shape[1] * scale))
        new_h = max(1, int(gray.shape[0] * scale))
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return gray


def _init_bot_template(first_image_path: Path, scale: float = 0.5) -> Optional[dict]:
    """Detect bot in first frame and return tracker dict (at given scale)."""
    gray = _load_gray(first_image_path, scale)
    if gray is None:
        return None

    h, w = gray.shape
    cx, cy = w // 2, h // 2
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 10
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    dish_radius = int(min(w, h) * 0.4)
    cv2.circle(mask, (cx, cy), dish_radius, 255, -1)
    adaptive = cv2.bitwise_and(adaptive, adaptive, mask=mask)

    contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if 2 < area < 2000:  # area scales with scale^2
            M = cv2.moments(c)
            if M['m00'] > 0:
                bx = M['m10'] / M['m00']
                by = M['m01'] / M['m00']
                dist = np.hypot(bx - cx, by - cy)
                candidates.append((dist, area, bx, by))

    if not candidates:
        return None

    candidates.sort()
    bot_x, bot_y = candidates[0][2], candidates[0][3]

    tmpl_half = max(20, int(80 * scale))
    tx1 = max(0, int(bot_x - tmpl_half))
    ty1 = max(0, int(bot_y - tmpl_half))
    tx2 = min(w, int(bot_x + tmpl_half))
    ty2 = min(h, int(bot_y + tmpl_half))
    template = gray[ty1:ty2, tx1:tx2]

    th, tw = template.shape
    if th == 0 or tw == 0:
        return None

    search_mask = np.zeros((max(1, h - th + 1), max(1, w - tw + 1)), dtype=np.uint8)
    cv2.circle(search_mask, (cx - tmpl_half, cy - tmpl_half), dish_radius, 255, -1)

    return {
        'template':    template,
        'search_mask': search_mask,
        'tmpl_half':   tmpl_half,
        'scale':       scale,
        'frame_shape': (h, w),
    }


def _detect_with_template(image_path: Path, tracker: dict) -> Optional[Tuple[float, float]]:
    gray = _load_gray(image_path, tracker['scale'])
    if gray is None:
        return None
    result = cv2.matchTemplate(gray, tracker['template'], cv2.TM_CCOEFF_NORMED)
    result_masked = np.where(tracker['search_mask'] > 0, result, -1.0)
    _, max_val, _, max_loc = cv2.minMaxLoc(result_masked)
    if max_val < 0.5:
        return None
    tmpl_half = tracker['tmpl_half']
    return (float(max_loc[0] + tmpl_half), float(max_loc[1] + tmpl_half))


def track_positions(image_paths: List[str], step: int, tracker: dict,
                    max_jump_px: float = 30.0) -> List[Tuple[float, float]]:
    """Track bot, filtering out detections that jump too far (bubble false positives).

    max_jump_px: max allowed displacement between consecutive sampled frames.
    At scale=0.5, 30 px ≈ 60 px on original image — well above real bot speed.
    """
    positions = []
    total = len(image_paths)
    indices = list(range(0, total, step))
    if total > 0 and (total - 1) not in indices:
        indices.append(total - 1)

    last_valid = None
    for i in indices:
        pos = _detect_with_template(Path(image_paths[i]), tracker)
        if pos is None:
            continue
        # Jump filter: reject if too far from last known position
        if last_valid is not None:
            jump = np.hypot(pos[0] - last_valid[0], pos[1] - last_valid[1])
            if jump > max_jump_px:
                continue  # likely a bubble match — skip
        positions.append(pos)
        last_valid = pos
    return positions


def total_distance(positions: List[Tuple[float, float]]) -> float:
    dist = 0.0
    for i in range(len(positions) - 1):
        dx = positions[i+1][0] - positions[i][0]
        dy = positions[i+1][1] - positions[i][1]
        dist += np.hypot(dx, dy)
    return dist


def compute_kinematics(batch_id: str, step: int, scale: float = 0.5) -> Optional[dict]:
    """Track bot and return kinematics dict for one batch."""
    try:
        records = load_batch_with_interventions(batch_id)
    except Exception as e:
        return {'error': str(e)}

    if not records:
        return {'error': 'no records'}

    record = records[0]
    if record.get('type') != 'intervention':
        return {'error': 'not intervention type'}

    pre_images  = record.get('pre_images', [])
    post_images = record.get('post_images', [])

    if not pre_images or not post_images:
        return {'error': 'missing pre/post images'}

    tracker = _init_bot_template(Path(pre_images[0]), scale=scale)
    if tracker is None:
        return {'error': 'bot template init failed'}

    pre_pos  = track_positions(pre_images,  step, tracker)
    post_pos = track_positions(post_images, step, tracker)

    if len(pre_pos) < 2 or len(post_pos) < 2:
        return {'error': f'too few positions: pre={len(pre_pos)}, post={len(post_pos)}'}

    pre_dist  = total_distance(pre_pos)
    post_dist = total_distance(post_pos)

    pre_n  = len(pre_pos)
    post_n = len(post_pos)

    pre_speed  = pre_dist  / pre_n   if pre_n  > 0 else 0.0
    post_speed = post_dist / post_n  if post_n > 0 else 0.0

    duration_ms = record.get('duration_ms', None)

    return {
        'pre_distance':  float(pre_dist),
        'post_distance': float(post_dist),
        'pre_n_frames':  pre_n,
        'post_n_frames': post_n,
        'pre_speed':     float(pre_speed),
        'post_speed':    float(post_speed),
        'duration_ms':   duration_ms,
        'step':          step,
        'scale':         scale,
    }


def _worker(args_tuple):
    """Top-level worker for ProcessPoolExecutor (must be picklable)."""
    batch_id, step, scale = args_tuple
    return batch_id, compute_kinematics(batch_id, step, scale)


def assign_labels(k: dict) -> dict:
    """Assign binary behavioral labels from kinematics."""
    ps  = k['post_speed']
    pre = k['pre_speed']
    labels = {
        'stop_moving': int(ps < STOP_THRESHOLD),
        'move_slow':   int(STOP_THRESHOLD <= ps < SLOW_THRESHOLD),
        'move_fast':   int(ps > FAST_THRESHOLD),
        'go_slower':   int(ps < pre * (1 - CHANGE_FRACTION)),
        'go_faster':   int(ps > pre * (1 + CHANGE_FRACTION)),
    }
    return labels


# ── Main pipeline ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step',    type=int,   default=10,  help='Frame sampling step (default 10)')
    parser.add_argument('--scale',   type=float, default=0.5, help='Image downscale factor (default 0.5, 4x speedup)')
    parser.add_argument('--workers', type=int,   default=4,   help='Parallel workers (default 4)')
    parser.add_argument('--recalc',  action='store_true',     help='Recompute even if cached')
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load archive batch IDs
    with open(ARCHIVE_JSON) as f:
        archive = json.load(f)
    batch_ids = sorted([k for k in archive if archive[k].get('duration_ms') is not None])
    print(f"Archive: {len(batch_ids)} batches")

    # Load or init cache
    cache = {}
    if CACHE_FILE.exists() and not args.recalc:
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        print(f"Cache: {len(cache)} entries loaded")

    # Batches that still need processing
    todo = [b for b in batch_ids if b not in cache or args.recalc]
    print(f"To process: {len(todo)} batches  (workers={args.workers}, step={args.step}, scale={args.scale})")

    if todo:
        tasks = [(b, args.step, args.scale) for b in todo]
        done_count = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, t): t[0] for t in tasks}
            for future in as_completed(futures):
                batch_id, k = future.result()
                cache[batch_id] = k
                done_count += 1
                if 'error' in k:
                    print(f"  [{done_count}/{len(todo)}] {batch_id}  ERROR: {k['error']}")
                else:
                    print(f"  [{done_count}/{len(todo)}] {batch_id}  pre={k['pre_speed']:.4f}  post={k['post_speed']:.4f}")
                # Save after each result (resume support)
                with open(CACHE_FILE, 'w') as f:
                    json.dump(cache, f, indent=2)

    print(f"\nDone: {len(cache)} total in cache")

    # Collect valid kinematics
    valid = {b: cache[b] for b in batch_ids if b in cache and 'error' not in cache[b]}
    errors = {b: cache[b]['error'] for b in batch_ids if b in cache and 'error' in cache[b]}
    print(f"Valid: {len(valid)} | Errors: {len(errors)}")
    if errors:
        for b, e in list(errors.items())[:5]:
            print(f"  {b}: {e}")

    if not valid:
        print("No valid kinematics to analyze.")
        return

    # Assign labels
    for b in valid:
        valid[b]['labels'] = assign_labels(valid[b])

    # ── Dataset table ─────────────────────────────────────────────
    behavior_map = {
        'stop_moving': 'stop moving',
        'move_slow':   'move slow',
        'move_fast':   'move fast',
        'go_slower':   'go slower',
        'go_faster':   'go faster',
    }
    counts = {k: sum(v['labels'][k] for v in valid.values()) for k in behavior_map}
    n_total = len(valid)

    print(f"\n{'='*50}")
    print(f"Dataset Characterization (N={n_total} batches)")
    print(f"{'='*50}")
    print(f"{'Behavior':<20} {'N':>5} {'%':>7}")
    print(f"{'-'*34}")
    for key, label in behavior_map.items():
        c = counts[key]
        print(f"{label:<20} {c:>5} {100*c/n_total:>6.1f}%")

    # Speed statistics
    pre_speeds  = [v['pre_speed']  for v in valid.values()]
    post_speeds = [v['post_speed'] for v in valid.values()]
    print(f"\nPre-intervention speed (px/sampled-frame):")
    print(f"  mean={np.mean(pre_speeds):.4f}  std={np.std(pre_speeds):.4f}")
    print(f"  min={np.min(pre_speeds):.4f}  p25={np.percentile(pre_speeds,25):.4f}")
    print(f"  p50={np.median(pre_speeds):.4f}  p75={np.percentile(pre_speeds,75):.4f}  max={np.max(pre_speeds):.4f}")
    print(f"\nPost-intervention speed (px/sampled-frame):")
    print(f"  mean={np.mean(post_speeds):.4f}  std={np.std(post_speeds):.4f}")
    print(f"  min={np.min(post_speeds):.4f}  p25={np.percentile(post_speeds,25):.4f}")
    print(f"  p50={np.median(post_speeds):.4f}  p75={np.percentile(post_speeds,75):.4f}  max={np.max(post_speeds):.4f}")
    print(f"\nThresholds used:")
    print(f"  stop_moving:  post_speed < {STOP_THRESHOLD}")
    print(f"  move_slow:    post_speed in [{STOP_THRESHOLD}, {SLOW_THRESHOLD})")
    print(f"  move_fast:    post_speed > {FAST_THRESHOLD}")
    print(f"  go_slower:    post_speed < pre_speed * {1 - CHANGE_FRACTION:.2f}")
    print(f"  go_faster:    post_speed > pre_speed * {1 + CHANGE_FRACTION:.2f}")

    # Save CSV table
    csv_path = PAPER_DIR / 'dataset_table.csv'
    with open(csv_path, 'w') as f:
        f.write("behavior,label_type,count,percent\n")
        for key, label in behavior_map.items():
            ltype = "absolute" if key in ('stop_moving', 'move_slow', 'move_fast') else "relative"
            c = counts[key]
            f.write(f"{label},{ltype},{c},{100*c/n_total:.1f}\n")
    print(f"\nSaved: {csv_path}")

    # Save full results
    results_path = PROJECT_ROOT / 'cache' / 'kinematics_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'n_total': n_total,
            'thresholds': {
                'stop_threshold':  STOP_THRESHOLD,
                'slow_threshold':  SLOW_THRESHOLD,
                'fast_threshold':  FAST_THRESHOLD,
                'change_fraction': CHANGE_FRACTION,
            },
            'behavior_counts': counts,
            'batches': {b: valid[b] for b in sorted(valid)},
        }, f, indent=2)
    print(f"Saved: {results_path}")

    # ── Plot 1: Speed distribution (pre vs post) ──────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(max(pre_speeds), max(post_speeds)) * 1.05, 40)
    ax.hist(pre_speeds,  bins=bins, alpha=0.6, color='steelblue', label=f'Pre-intervention (n={n_total})', edgecolor='white', linewidth=0.4)
    ax.hist(post_speeds, bins=bins, alpha=0.6, color='tomato',    label=f'Post-intervention (n={n_total})', edgecolor='white', linewidth=0.4)

    # Threshold lines
    ax.axvline(STOP_THRESHOLD, color='black',  linestyle='--', linewidth=1.2, label=f'Stop threshold ({STOP_THRESHOLD})')
    ax.axvline(SLOW_THRESHOLD, color='orange', linestyle='--', linewidth=1.2, label=f'Slow threshold ({SLOW_THRESHOLD})')
    ax.axvline(FAST_THRESHOLD, color='green',  linestyle='--', linewidth=1.2, label=f'Fast threshold ({FAST_THRESHOLD})')

    ax.set_xlabel('Speed (px / sampled frame)', fontsize=12)
    ax.set_ylabel('Number of batches', fontsize=12)
    ax.set_title('Pre vs Post Intervention Speed Distribution\n(139 batches, CV kinematics)', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = FIGURES_DIR / 'dataset_speed_distribution.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved: {p}")

    # ── Plot 2: Behavior counts (bar chart) ───────────────────────
    labels_display = ['stop\nmoving', 'move\nslow', 'move\nfast', 'go\nslower', 'go\nfaster']
    keys_list = ['stop_moving', 'move_slow', 'move_fast', 'go_slower', 'go_faster']
    colors = ['#444', '#6baed6', '#fd8d3c', '#74c476', '#fd8d3c']
    colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#2ca02c', '#d62728']

    fig, ax = plt.subplots(figsize=(7, 5))
    xpos = np.arange(len(keys_list))
    bars = ax.bar(xpos, [counts[k] for k in keys_list], color=colors, edgecolor='white', linewidth=0.5)

    for bar, key in zip(bars, keys_list):
        c = counts[key]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{c}\n({100*c/n_total:.0f}%)', ha='center', va='bottom', fontsize=10)

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels_display, fontsize=11)
    ax.set_ylabel('Number of batches', fontsize=12)
    ax.set_title(f'Behavioral Label Counts (N={n_total} batches)\n[absolute: left 3] [relative: right 2]', fontsize=12)
    ax.set_ylim(0, n_total * 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axvline(2.5, color='gray', linestyle=':', linewidth=1.5)
    ax.text(1.0, n_total * 1.08, 'Absolute speed', ha='center', fontsize=9, color='gray')
    ax.text(3.5, n_total * 1.08, 'Relative change', ha='center', fontsize=9, color='gray')
    fig.tight_layout()
    p = FIGURES_DIR / 'dataset_behavior_counts.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved: {p}")

    # ── Plot 3: Scatter pre vs post speed ─────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pre_speeds, post_speeds, alpha=0.5, s=20, color='steelblue', edgecolors='none')
    lim = max(max(pre_speeds), max(post_speeds)) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1.0, label='no change (y=x)')
    ax.set_xlabel('Pre-intervention speed (px/frame)', fontsize=11)
    ax.set_ylabel('Post-intervention speed (px/frame)', fontsize=11)
    ax.set_title('Pre vs Post Speed per Batch', fontsize=12)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = FIGURES_DIR / 'dataset_speed_scatter.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved: {p}")

    print("\n[OK] characterize_archive.py complete")


if __name__ == '__main__':
    main()
