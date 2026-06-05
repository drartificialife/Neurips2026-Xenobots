#!/usr/bin/env python3
"""
Trajectory visualization for pre/post intervention motion.

Creates one combined plot for pre/post bot movement with arrows + speed indicators.
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects

from parse_interventions import load_batch_with_interventions


def detect_bot_centroid(image_path: Path) -> Tuple[float, float]:
    """Detect bot centroid in single image (assuming single bot present)."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError(f"No contour detected in {image_path}")

    # Choose largest contour, assumed bot
    contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(contour)
    if M['m00'] == 0:
        raise RuntimeError(f"Zero-moment contour in {image_path}")

    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    return float(cx), float(cy)


def sample_positions(image_paths: List[str], macro_step: int) -> List[Tuple[float, float, int]]:
    """Sample bot positions with index and macro_step stride."""
    if not image_paths:
        return []

    positions = []
    for i in range(0, len(image_paths), macro_step):
        path = Path(image_paths[i])
        try:
            cx, cy = detect_bot_centroid(path)
            positions.append((cx, cy, i))
        except Exception as e:
            print(f"Warning: centroid failed for {path.name}: {e}")

    return positions


def plot_trajectory(pre_positions: List[Tuple[float, float, int]],
                    post_positions: List[Tuple[float, float, int]],
                    output_path: Path,
                    macro_step: int):
    """Plot pre/post trajectories with arrow and speed annotation."""
    if not pre_positions or not post_positions:
        raise ValueError("Need both pre and post positions")


    pre_x, pre_y, pre_t = zip(*pre_positions)
    post_x, post_y, post_t = zip(*post_positions)

    # Plot pre and post at their true positions (no alignment)
    post_x_shifted = post_x
    post_y_shifted = post_y

    fig, (ax_pre, ax_post) = plt.subplots(2, 1, figsize=(10, 12), sharex=True, sharey=True, gridspec_kw={'hspace': 0.18})

    # Colormap for steps (early to late)
    pre_cmap = plt.colormaps['Blues'].resampled(len(pre_x))
    post_cmap = plt.colormaps['Reds'].resampled(len(post_x))

    # --- PRE subplot ---
    ax_pre.plot(pre_x, pre_y, '-', color='blue', linewidth=0.8, alpha=0.25)
    for i, (x, y, t) in enumerate(zip(pre_x, pre_y, pre_t)):
        ax_pre.scatter(x, y, s=18, color=pre_cmap(i), marker='o', alpha=0.95, zorder=5)
        if i == 0 or i == len(pre_x)-1:
            ax_pre.text(x, y, f'{t}', color='black', fontsize=11, ha='center', va='center', fontweight='bold',
                       zorder=7, path_effects=[patheffects.withStroke(linewidth=2.5, foreground='white')], clip_on=True)
    def draw_segments(ax, xs, ys, cmap):
        for i in range(len(xs) - 1):
            x0, y0 = xs[i], ys[i]
            x1, y1 = xs[i+1], ys[i+1]
            dx, dy = x1 - x0, y1 - y0
            dist = np.hypot(dx, dy)
            width = 1.2 + min(0.5, dist / 150.0)
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle='->',
                    color=cmap(i),
                    lw=width,
                    mutation_scale=16,
                    shrinkA=0,
                    shrinkB=0,
                    alpha=0.85,
                ),
                zorder=3,
            )
    draw_segments(ax_pre, pre_x, pre_y, pre_cmap)
    ax_pre.scatter([pre_x[0]], [pre_y[0]], color='cyan', s=28, marker='o', label='Pre start (step 0)')
    ax_pre.scatter([pre_x[-1]], [pre_y[-1]], color='navy', s=32, marker='x', linewidths=1.8, label=f'Pre end (step {pre_t[-1]})')
    ax_pre.set_title('Pre-intervention')
    ax_pre.set_ylabel('Y (pixels)')
    ax_pre.invert_yaxis()
    ax_pre.grid(True, alpha=0.3)
    ax_pre.legend(loc='best', fontsize=8)

    # --- POST subplot ---
    ax_post.plot(post_x_shifted, post_y_shifted, '-', color='red', linewidth=0.8, alpha=0.25)
    for i, (x, y, t) in enumerate(zip(post_x_shifted, post_y_shifted, post_t)):
        ax_post.scatter(x, y, s=18, color=post_cmap(i), marker='o', alpha=0.95, zorder=5)
        if i == 0 or i == len(post_x_shifted)-1:
            ax_post.text(x, y, f'{t}', color='black', fontsize=11, ha='center', va='center', fontweight='bold',
                        zorder=7, path_effects=[patheffects.withStroke(linewidth=2.5, foreground='white')], clip_on=True)
    draw_segments(ax_post, post_x_shifted, post_y_shifted, post_cmap)
    ax_post.scatter([post_x_shifted[0]], [post_y_shifted[0]], color='orange', s=28, marker='o', label=f'Post start (step {post_t[0]})')
    ax_post.scatter([post_x_shifted[-1]], [post_y_shifted[-1]], color='maroon', s=32, marker='x', linewidths=1.8, label=f'Post end (step {post_t[-1]})')
    ax_post.set_title('Post-intervention')
    ax_post.set_xlabel('X (pixels)')
    ax_post.set_ylabel('Y (pixels)')
    ax_post.invert_yaxis()
    ax_post.grid(True, alpha=0.3)
    ax_post.legend(loc='best', fontsize=8)

    # Set same axis limits for both subplots

    all_x = list(pre_x) + list(post_x_shifted)
    all_y = list(pre_y) + list(post_y_shifted)
    # Add margin (5% of range) to avoid data flush against axis
    def add_margin(minv, maxv, frac=0.05):
        rng = maxv - minv
        if rng == 0:
            return minv - 1, maxv + 1
        margin = rng * frac
        return minv - margin, maxv + margin

    xlim = add_margin(min(all_x), max(all_x))
    ylim = add_margin(min(all_y), max(all_y))
    ax_pre.set_xlim(*xlim)
    ax_pre.set_ylim(*ylim)
    ax_post.set_xlim(*xlim)
    ax_post.set_ylim(*ylim)

    fig.suptitle(f"Bot Trajectory (macro_step={macro_step})\nStep number (start/end) and color = time order", fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved trajectory figure: {output_path}")

def save_all_trajectories(pre_positions, post_positions, batch_id, macro_step, all_dir):
    """Save plot in all_trajectories directory as well."""
    all_dir = Path(all_dir)
    all_dir.mkdir(parents=True, exist_ok=True)
    out_path = all_dir / f"{batch_id}_macro{macro_step}_prepost.png"
    plot_trajectory(pre_positions, post_positions, out_path, macro_step)


def main():
    parser = argparse.ArgumentParser(description='Trajectory pre/post intervention visualization')
    parser.add_argument('--batch', required=True, help='Batch ID (batch-xxxxx)')
    parser.add_argument('--macro-step', type=int, default=50, help='Frame interval for sampling')
    args = parser.parse_args()

    records = load_batch_with_interventions(args.batch)
    if not records:
        raise FileNotFoundError(f"No data for batch {args.batch}")

    record = records[0]
    if record.get('type') != 'intervention':
        raise ValueError('Batch is not intervention type')

    pre_images = record.get('pre_images', [])
    post_images = record.get('post_images', [])

    if not pre_images or not post_images:
        raise ValueError('Need both pre and post images')

    pre_positions = sample_positions(pre_images, args.macro_step)
    post_positions = sample_positions(post_images, args.macro_step)

    if not pre_positions or not post_positions:
        raise RuntimeError('Failed to get bot positions; check filter or image content')

    # Save output directly in the batch archive folder
    from parse_interventions import ARCHIVE_ROOT
    batch_dir = ARCHIVE_ROOT / args.batch
    output_path = batch_dir / f"trajectory_macro{args.macro_step}_prepost.png"
    plot_trajectory(pre_positions, post_positions, output_path, args.macro_step)
    # Also save in all_trajectories
    save_all_trajectories(pre_positions, post_positions, args.batch, args.macro_step, ARCHIVE_ROOT / "all_trajectories")


if __name__ == '__main__':
    main()
