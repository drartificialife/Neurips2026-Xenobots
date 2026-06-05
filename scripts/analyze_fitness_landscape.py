#!/usr/bin/env python3
"""
Analyze and visualize the fitness landscape: which intervention durations work best for each prompt.

Creates:
1. Scatter plots: Duration vs Fitness for each prompt
2. Heatmap: Duration bins vs Fitness scores
3. Histogram: Distribution of high-fitness interventions

Usage:
    python scripts/analyze_fitness_landscape.py
    python scripts/analyze_fitness_landscape.py --min-fitness 0.8
    python scripts/analyze_fitness_landscape.py --output plots/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
except ImportError:
    print("Installing matplotlib...")
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'matplotlib'])
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

CACHE_DIR = Path('cache')
SCORES_FILE = CACHE_DIR / 'archive_vlm_scores.json'
PROMPTS = ["slow down", "stop moving", "go fast", "move faster"]
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']


def load_precomputed_scores() -> Dict[str, Dict]:
    """Load pre-computed VLM scores."""
    if not SCORES_FILE.exists():
        raise FileNotFoundError(f"Pre-computed scores not found: {SCORES_FILE}")

    with open(SCORES_FILE, 'r') as f:
        return json.load(f)


def get_fitness_landscape(scores: Dict, prompt: str, min_fitness: float = 0.0) -> Tuple[List[float], List[float]]:
    """
    Extract durations and fitness scores for a prompt.

    Returns:
        Tuple of (durations, fitnesses) filtering out errors and low scores
    """
    durations = []
    fitnesses = []

    for batch_id, batch_data in scores.items():
        duration = batch_data.get('duration_ms')
        if duration is None:
            continue

        if prompt not in batch_data:
            continue

        score_obj = batch_data[prompt]
        score = score_obj.get('score', 0.0)
        desc = str(score_obj.get('desc', '')).strip()

        # Skip error entries
        if desc.startswith('Error:'):
            continue

        if score >= min_fitness:
            durations.append(duration)
            fitnesses.append(score)

    return durations, fitnesses


def plot_fitness_landscape_scatter():
    """Create scatter plots of duration vs fitness for each prompt."""
    scores = load_precomputed_scores()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Intervention Fitness Landscape: Duration vs Fitness Score',
                 fontsize=16, fontweight='bold')

    axes = axes.flatten()

    for idx, (prompt, color) in enumerate(zip(PROMPTS, COLORS)):
        durations, fitnesses = get_fitness_landscape(scores, prompt, min_fitness=0.0)

        ax = axes[idx]

        # Scatter plot: all points
        scatter = ax.scatter(durations, fitnesses, alpha=0.6, s=80, c=fitnesses,
                           cmap='RdYlGn', edgecolor='black', linewidth=0.5, vmin=0, vmax=1)

        # Highlight high-fitness region (>=0.9)
        high_fitness_durations = [d for d, f in zip(durations, fitnesses) if f >= 0.9]
        high_fitness_scores = [f for d, f in zip(durations, fitnesses) if f >= 0.9]
        if high_fitness_durations:
            ax.scatter(high_fitness_durations, high_fitness_scores, alpha=0.8, s=150,
                      marker='*', edgecolor='red', linewidth=1.5, color='red', label='High fitness (>=0.9)')

        ax.set_title(f'{prompt.upper()}\n({len(durations)} batches evaluated)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Intervention Duration (ms)', fontsize=11)
        ax.set_ylabel('Fitness Score', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.1])

        if high_fitness_durations:
            ax.legend(fontsize=10, loc='lower right')

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Fitness', fontsize=10)

    plt.tight_layout()
    return fig


def plot_fitness_distribution():
    """Create histograms showing distribution of high-fitness interventions."""
    scores = load_precomputed_scores()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Distribution of High-Fitness Interventions (Fitness >= 0.8)',
                 fontsize=16, fontweight='bold')

    axes = axes.flatten()

    for idx, (prompt, color) in enumerate(zip(PROMPTS, COLORS)):
        durations, fitnesses = get_fitness_landscape(scores, prompt, min_fitness=0.8)

        ax = axes[idx]

        if durations:
            # Histogram of durations (log scale)
            n_bins = 15
            ax.hist(durations, bins=n_bins, color=color, alpha=0.7, edgecolor='black', linewidth=1.2)

            # Add mean line
            mean_duration = np.mean(durations)
            ax.axvline(mean_duration, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_duration:.0f} ms')

            # Add median line
            median_duration = np.median(durations)
            ax.axvline(median_duration, color='blue', linestyle=':', linewidth=2, label=f'Median: {median_duration:.0f} ms')

            ax.set_title(f'{prompt.upper()}\n(n={len(durations)} high-fitness batches)',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Intervention Duration (ms)', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title(f'{prompt.upper()}\n(No high-fitness batches)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_fitness_heatmap():
    """Create heatmap of duration bins vs fitness score ranges."""
    scores = load_precomputed_scores()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Fitness Heatmap: Duration Bins vs Fitness Ranges',
                 fontsize=16, fontweight='bold')

    axes = axes.flatten()

    # Define bins
    duration_bins = np.logspace(3, 5.5, 12)  # Log scale: 1k to 300k ms
    fitness_bins = np.linspace(0, 1, 11)  # 0 to 1 fitness

    for idx, (prompt, color) in enumerate(zip(PROMPTS, COLORS)):
        durations, fitnesses = get_fitness_landscape(scores, prompt, min_fitness=0.0)

        ax = axes[idx]

        if durations:
            # Create 2D histogram
            heatmap, xedges, yedges = np.histogram2d(
                durations, fitnesses,
                bins=[duration_bins, fitness_bins]
            )

            # Plot heatmap
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = ax.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto',
                          cmap='YlOrRd', interpolation='nearest')

            ax.set_xscale('log')
            ax.set_title(f'{prompt.upper()}\n(Total: {len(durations)} batches)',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Intervention Duration (ms, log scale)', fontsize=11)
            ax.set_ylabel('Fitness Score', fontsize=11)

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Count', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14, transform=ax.transAxes)

    plt.tight_layout()
    return fig


def print_landscape_stats():
    """Print statistics about the fitness landscape."""
    scores = load_precomputed_scores()

    print("\n" + "="*70)
    print("FITNESS LANDSCAPE STATISTICS")
    print("="*70)

    for prompt in PROMPTS:
        durations, fitnesses = get_fitness_landscape(scores, prompt, min_fitness=0.0)

        if not durations:
            print(f"\n{prompt.upper()}: No data")
            continue

        durations = np.array(durations)
        fitnesses = np.array(fitnesses)

        # High fitness (>= 0.9)
        high_mask = fitnesses >= 0.9
        high_durations = durations[high_mask]

        # Medium fitness (0.5-0.9)
        med_mask = (fitnesses >= 0.5) & (fitnesses < 0.9)
        med_durations = durations[med_mask]

        print(f"\n{prompt.upper()}:")
        print(f"  Total batches: {len(fitnesses)}")
        print(f"  Fitness range: {np.min(fitnesses):.4f} - {np.max(fitnesses):.4f}")
        print(f"  Fitness mean: {np.mean(fitnesses):.4f}, std: {np.std(fitnesses):.4f}")

        if len(high_durations) > 0:
            print(f"\n  HIGH FITNESS (>= 0.9): {len(high_durations)} batches")
            print(f"    Duration range: {np.min(high_durations):.0f} - {np.max(high_durations):.0f} ms")
            print(f"    Duration mean: {np.mean(high_durations):.0f} ms")
            print(f"    Duration median: {np.median(high_durations):.0f} ms")

        if len(med_durations) > 0:
            print(f"\n  MEDIUM FITNESS (0.5-0.9): {len(med_durations)} batches")
            print(f"    Duration range: {np.min(med_durations):.0f} - {np.max(med_durations):.0f} ms")
            print(f"    Duration mean: {np.mean(med_durations):.0f} ms")

    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze fitness landscape')
    parser.add_argument('--min-fitness', type=float, default=0.0,
                       help='Minimum fitness to include (default: 0.0)')
    parser.add_argument('--output', type=str, default='results/',
                       help='Output directory for plots (default: results/)')
    parser.add_argument('--format', type=str, default='png',
                       help='Output format: png, pdf, svg (default: png)')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Analyzing fitness landscape...")
    print(f"Output directory: {output_dir}\n")

    # Print statistics
    print_landscape_stats()

    # Generate plots
    print("Generating visualizations...")

    print("  - Scatter plots (Duration vs Fitness)...")
    fig = plot_fitness_landscape_scatter()
    output_path = output_dir / f"fitness_landscape_scatter.{args.format}"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_path}")
    plt.close()

    print("  - Distribution histograms...")
    fig = plot_fitness_distribution()
    output_path = output_dir / f"fitness_distribution.{args.format}"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_path}")
    plt.close()

    print("  - Fitness heatmaps...")
    fig = plot_fitness_heatmap()
    output_path = output_dir / f"fitness_heatmap.{args.format}"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_path}")
    plt.close()

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
