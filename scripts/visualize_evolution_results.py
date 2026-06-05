#!/usr/bin/env python3
"""
Visualize evolution results: fitness curves, comparisons, and summaries.

Usage:
    python scripts/visualize_evolution_results.py
    python scripts/visualize_evolution_results.py --single-only
    python scripts/visualize_evolution_results.py --multi-only
    python scripts/visualize_evolution_results.py --output plots/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Try to import matplotlib, install if needed
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("Installing matplotlib...")
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'matplotlib'])
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

RESULTS_DIR = Path('results')
PROMPTS = ["slow down", "stop moving", "go fast", "move faster"]


def load_evolution_results(prompt: str = None) -> Dict[str, Any]:
    """Load evolution results JSON file."""
    if prompt:
        file_path = RESULTS_DIR / f"evolution_results_{prompt.replace(' ', '_')}.json"
    else:
        file_path = RESULTS_DIR / "evolution_results_multi_prompt.json"

    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")

    with open(file_path, 'r') as f:
        return json.load(f)


def plot_single_prompt_fitness_curves():
    """Plot fitness curves for all 4 single-prompt evolution runs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Single-Prompt Evolution: Fitness Over Generations', fontsize=16, fontweight='bold')

    axes = axes.flatten()

    for idx, prompt in enumerate(PROMPTS):
        try:
            result = load_evolution_results(prompt)
        except FileNotFoundError:
            print(f"Warning: Results not found for '{prompt}'")
            continue

        generations = [log['generation'] for log in result['generation_log']]
        mean_fitness = [log['mean_fitness'] for log in result['generation_log']]
        max_fitness = [log['max_fitness'] for log in result['generation_log']]

        ax = axes[idx]
        ax.plot(generations, mean_fitness, 'o-', label='Mean Fitness', linewidth=2, markersize=4, alpha=0.7)
        ax.plot(generations, max_fitness, 's-', label='Max Fitness', linewidth=2, markersize=4, alpha=0.7)
        ax.axhline(y=result['best_fitness'], color='r', linestyle='--', alpha=0.5, label=f"Best: {result['best_fitness']:.4f}")

        ax.set_title(f"{prompt}\nBest duration: {result['best_duration_ms']:.0f} ms (Gen {result['best_generation']})", fontsize=11)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness Score')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    return fig


def plot_multi_prompt_fitness_curve():
    """Plot fitness curve for multi-prompt evolution."""
    try:
        result = load_evolution_results()
    except FileNotFoundError:
        print("Warning: Multi-prompt results not found")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Multi-Prompt Evolution: Balanced Optimization', fontsize=16, fontweight='bold')

    # Fitness curve
    generations = [log['generation'] for log in result['generation_log']]
    mean_fitness = [log['mean_fitness'] for log in result['generation_log']]
    max_fitness = [log['max_fitness'] for log in result['generation_log']]

    ax1.plot(generations, mean_fitness, 'o-', label='Mean Fitness', linewidth=2, markersize=5, alpha=0.7, color='blue')
    ax1.plot(generations, max_fitness, 's-', label='Max Fitness', linewidth=2, markersize=5, alpha=0.7, color='green')
    ax1.axhline(y=result['best_fitness'], color='r', linestyle='--', alpha=0.5, linewidth=2, label=f"Best: {result['best_fitness']:.4f}")
    ax1.axvline(x=result['best_generation'], color='r', linestyle=':', alpha=0.5, linewidth=2)

    ax1.set_title(f'Fitness Over Generations\n(Best at Gen {result["best_generation"]})', fontsize=12)
    ax1.set_xlabel('Generation', fontsize=11)
    ax1.set_ylabel('Fitness Score', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 1.05])

    # Per-prompt scores (bar chart)
    scores = result['best_per_prompt_scores']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    bars = ax2.bar(PROMPTS, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_title('Per-Prompt Fitness Scores\n(Final Best Solution)', fontsize=12)
    ax2.set_ylabel('Fitness Score', fontsize=11)
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')

    plt.tight_layout()
    return fig


def plot_duration_comparison():
    """Compare optimal durations across single vs multi-prompt evolution."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Optimal Intervention Durations: Single vs Multi-Prompt', fontsize=14, fontweight='bold')

    # Load results
    single_prompts_results = {}
    for prompt in PROMPTS:
        try:
            result = load_evolution_results(prompt)
            single_prompts_results[prompt] = result['best_duration_ms']
        except FileNotFoundError:
            print(f"Warning: Single-prompt results not found for '{prompt}'")

    multi_result = load_evolution_results()
    multi_durations = {p: d for p, d in zip(PROMPTS, multi_result['best_genome'])}

    x = np.arange(len(PROMPTS))
    width = 0.35

    single_durations = [single_prompts_results.get(p, 0) for p in PROMPTS]
    bars1 = ax.bar(x - width/2, single_durations, width, label='Single-Prompt', alpha=0.8, color='skyblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, [multi_durations[p] for p in PROMPTS], width, label='Multi-Prompt', alpha=0.8, color='coral', edgecolor='black')

    ax.set_xlabel('Prompt', fontsize=12, fontweight='bold')
    ax.set_ylabel('Duration (ms)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(PROMPTS)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height/1000:.1f}k',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def plot_fitness_summary():
    """Summary table of all results."""
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    ax.axis('tight')
    ax.axis('off')

    # Collect data
    data = []
    headers = ['Evolution Type', 'Prompt', 'Duration (ms)', 'Fitness', 'Generation', 'Notes']

    # Single-prompt results
    for prompt in PROMPTS:
        try:
            result = load_evolution_results(prompt)
            data.append([
                'Single',
                prompt,
                f"{result['best_duration_ms']:.1f}",
                f"{result['best_fitness']:.4f}",
                result['best_generation'],
                f"Pop: {result['population_size']}"
            ])
        except FileNotFoundError:
            pass

    # Multi-prompt result
    try:
        result = load_evolution_results()
        for i, prompt in enumerate(PROMPTS):
            data.append([
                'Multi' if i == 0 else '',
                prompt,
                f"{result['best_genome'][i]:.1f}",
                f"{result['best_per_prompt_scores'][i]:.4f}",
                result['best_generation'] if i == 0 else '',
                f"Mean: {result['best_fitness']:.4f}" if i == 0 else ''
            ])
    except FileNotFoundError:
        pass

    table = ax.table(cellText=data, colLabels=headers, cellLoc='center', loc='center',
                     colWidths=[0.12, 0.15, 0.15, 0.12, 0.12, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
            else:
                table[(i, j)].set_facecolor('white')

    plt.title('Evolution Results Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize evolution results')
    parser.add_argument('--single-only', action='store_true', help='Plot single-prompt results only')
    parser.add_argument('--multi-only', action='store_true', help='Plot multi-prompt results only')
    parser.add_argument('--output', type=str, default='results/', help='Output directory for plots (default: results/)')
    parser.add_argument('--format', type=str, default='png', help='Output format: png, pdf, svg (default: png)')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Generating visualizations...")
    print(f"Output directory: {output_dir}\n")

    # Single-prompt fitness curves
    if not args.multi_only:
        print("  - Single-prompt fitness curves...")
        fig = plot_single_prompt_fitness_curves()
        if fig:
            output_path = output_dir / f"evolution_single_prompt_fitness.{args.format}"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"    Saved: {output_path}")
        plt.close()

    # Multi-prompt fitness curve
    if not args.single_only:
        print("  - Multi-prompt fitness curve...")
        fig = plot_multi_prompt_fitness_curve()
        if fig:
            output_path = output_dir / f"evolution_multi_prompt_fitness.{args.format}"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"    Saved: {output_path}")
        plt.close()

    # Duration comparison
    if not args.single_only:
        print("  - Duration comparison (Single vs Multi)...")
        fig = plot_duration_comparison()
        output_path = output_dir / f"duration_comparison.{args.format}"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"    Saved: {output_path}")
        plt.close()

    # Summary table
    print("  - Results summary table...")
    fig = plot_fitness_summary()
    output_path = output_dir / f"evolution_summary.{args.format}"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_path}")
    plt.close()

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
