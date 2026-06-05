#!/usr/bin/env python3
"""
Evolve Multi-Prompt P2I Network — TRAIN/TEST SPLIT VERSION.

Uses only train_archive.json (111 batches, 80% of full archive).
The remaining 28 batches (test_archive.json) are held out for video generalization test.

Backbone: 1D Conv layers → GAP → shared features
Heads: one FC output per base prompt → sigmoid → duration [0,1]
Fitness: average VLM score across all base prompts

Usage:
    python evolution/train_test_split/evolve_multi_prompt.py --generations 100 --population-size 100
    python evolution/train_test_split/evolve_multi_prompt.py --runs 30 --device cuda
"""

import argparse
import copy
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SPLIT_DIR   = Path(__file__).parent
SCORES_FILE = SPLIT_DIR / 'train_archive.json'   # 111 batches only (held out: test_archive.json)
MODELS_DIR  = SPLIT_DIR / 'models'
RESULTS_DIR = SPLIT_DIR / 'results'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / 'scripts' / 'train_prompts.json') as f:
    BASE_PROMPTS = json.load(f)


def _prompt_to_key(prompt: str) -> str:
    return prompt.replace(' ', '_')


# ---------------------------------------------------------------------------
# Network — 1D CNN with multi-head output
# ---------------------------------------------------------------------------
class MultiPromptP2ICNN(nn.Module):
    """1D CNN backbone + per-prompt output heads.

    Input: embedding (384D)
    Output: N durations (one per base prompt), normalized [0,1]
    """

    def __init__(self, prompts: List[str] = None, embedding_dim: int = 384):
        super().__init__()
        if prompts is None:
            prompts = BASE_PROMPTS
        self.prompts = prompts

        # 1D Conv backbone: (batch, 1, 384) → (batch, 16, 384) → GAP → (batch, 16)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Shared FC after GAP
        self.shared_fc = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Per-prompt heads
        self.heads = nn.ModuleDict({
            _prompt_to_key(p): nn.Linear(32, 1) for p in prompts
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (batch, 384)
        x = x.unsqueeze(1)              # (batch, 1, 384)
        x = self.conv(x)                # (batch, 16, 384)
        x = x.mean(dim=2)               # GAP → (batch, 16)
        x = self.shared_fc(x)           # (batch, 32)

        outputs = {}
        for prompt in self.prompts:
            key = _prompt_to_key(prompt)
            outputs[prompt] = torch.sigmoid(self.heads[key](x))
        return outputs


# ---------------------------------------------------------------------------
# Archive / fitness
# ---------------------------------------------------------------------------
def load_archive():
    with open(SCORES_FILE) as f:
        vlm_scores = json.load(f)

    batch_ids = sorted([
        k for k in vlm_scores
        if k.startswith('batch-') and vlm_scores[k].get('duration_ms') is not None
    ])
    durations = np.array([vlm_scores[b]['duration_ms'] for b in batch_ids]).reshape(-1, 1)
    nn_index = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(durations)
    dur_min, dur_max = float(durations.min()), float(durations.max())

    return vlm_scores, batch_ids, nn_index, dur_min, dur_max


def fitness_lookup(pred_norm, prompt, vlm_scores, batch_ids, nn_index, dur_min, dur_max):
    duration_ms = pred_norm * (dur_max - dur_min) + dur_min
    duration_ms = np.clip(duration_ms, dur_min, dur_max)

    arr = np.array([duration_ms]).reshape(1, -1)
    _, indices = nn_index.kneighbors(arr)
    batch_id = batch_ids[min(indices[0][0], len(batch_ids) - 1)]

    score_entry = vlm_scores[batch_id].get(prompt, {})
    if isinstance(score_entry, dict):
        if str(score_entry.get('desc', '')).startswith('Error'):
            return 0.0, batch_id, duration_ms
        return float(score_entry.get('score', 0.0)), batch_id, duration_ms
    return 0.0, batch_id, duration_ms


def evaluate_network(network, embedder, vlm_scores, batch_ids, nn_index, dur_min, dur_max, device='cpu'):
    """Evaluate on all base prompts. Returns (avg_fitness, per_prompt_fitness)."""
    network.eval()
    with torch.no_grad():
        emb = embedder.encode(BASE_PROMPTS[0], convert_to_tensor=False)
        tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
        outputs = network(tensor)

        fitness_by_prompt = {}
        for prompt in BASE_PROMPTS:
            pred_norm = outputs[prompt].squeeze().item()
            score, _, _ = fitness_lookup(pred_norm, prompt, vlm_scores, batch_ids, nn_index, dur_min, dur_max)
            fitness_by_prompt[prompt] = score

        avg = np.mean(list(fitness_by_prompt.values()))
        return avg, fitness_by_prompt


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
def mutate(network, strength=0.05):
    child = copy.deepcopy(network)
    for param in child.parameters():
        param.data.add_(torch.randn_like(param) * strength)
    return child


def crossover(p1, p2):
    alpha = np.random.uniform(0.2, 0.8)
    child = copy.deepcopy(p1)
    sd = child.state_dict()
    sd1, sd2 = p1.state_dict(), p2.state_dict()
    for key in sd:
        sd[key] = alpha * sd1[key] + (1 - alpha) * sd2[key]
    child.load_state_dict(sd)
    return child


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Evolution
# ---------------------------------------------------------------------------
def evolve_multi_prompt(
    generations: int = 200,
    population_size: int = 200,
    seed: int = 0,
    device: str = 'cpu',
) -> Dict:
    set_seed(seed)

    print(f"\n{'='*60}")
    print(f"Evolution — Multi-Prompt (CNN)")
    print(f"Prompts: {BASE_PROMPTS}")
    print(f"Seed: {seed} | Gens: {generations} | Pop: {population_size}")
    print(f"{'='*60}\n")

    vlm_scores, batch_ids, nn_index, dur_min, dur_max = load_archive()
    print(f"Archive: {len(batch_ids)} batches, duration [{dur_min:.0f}, {dur_max:.0f}] ms")

    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    population = [MultiPromptP2ICNN(prompts=BASE_PROMPTS).to(device) for _ in range(population_size)]
    n_params = sum(p.numel() for p in population[0].parameters())
    print(f"Network params: {n_params:,}")

    best_network = None
    best_fitness = -float('inf')
    history = []

    for gen in range(generations):
        gen_fitness = []
        gen_breakdown = []

        for net in population:
            avg_f, breakdown = evaluate_network(net, embedder, vlm_scores, batch_ids, nn_index, dur_min, dur_max, device)
            gen_fitness.append(avg_f)
            gen_breakdown.append(breakdown)

        gen_fitness = np.array(gen_fitness)
        best_idx = gen_fitness.argmax()

        if gen_fitness[best_idx] > best_fitness:
            best_fitness = gen_fitness[best_idx]
            best_network = copy.deepcopy(population[best_idx])

        if (gen + 1) % 10 == 0 or gen == 0:
            print(f"  Gen {gen+1:4d} | best={gen_fitness[best_idx]:.4f} mean={gen_fitness.mean():.4f} | "
                  f"global_best={best_fitness:.4f}")
            breakdown = gen_breakdown[best_idx]
            parts = ', '.join([f'{p}: {breakdown[p]:.3f}' for p in BASE_PROMPTS])
            print(f"           | {parts}")

        history.append({
            'generation': gen + 1,
            'best': float(gen_fitness.max()),
            'mean': float(gen_fitness.mean()),
            'worst': float(gen_fitness.min()),
            'best_breakdown': {k: float(v) for k, v in gen_breakdown[best_idx].items()},
        })

        # Selection + reproduction
        elite_count = max(2, int(population_size * 0.02))
        sorted_idx = np.argsort(gen_fitness)[::-1]
        elite = [population[i] for i in sorted_idx[:elite_count]]

        parent_pool_size = max(4, population_size // 5)
        parent_pool = [population[i] for i in sorted_idx[:parent_pool_size]]

        next_pop = [copy.deepcopy(e) for e in elite]
        while len(next_pop) < population_size:
            p1 = parent_pool[np.random.randint(len(parent_pool))]
            p2 = parent_pool[np.random.randint(len(parent_pool))]
            child = crossover(p1, p2)
            child = mutate(child)
            next_pop.append(child)

        population = next_pop[:population_size]

    # Final
    print(f"\n{'='*60}")
    print(f"Done! Best fitness: {best_fitness:.4f}")
    print(f"{'='*60}")

    seed_tag = f"seed{seed:03d}"

    model_path = MODELS_DIR / f'evo_cnn_multi_{seed_tag}.pt'
    torch.save({
        'model_state_dict': best_network.state_dict(),
        'best_fitness': float(best_fitness),
        'base_prompts': BASE_PROMPTS,
        'seed': seed,
        'network_type': 'CNN',
    }, model_path)

    history_path = RESULTS_DIR / f'evo_cnn_multi_{seed_tag}.json'
    with open(history_path, 'w') as f:
        json.dump({'seed': seed, 'network_type': 'CNN', 'history': history}, f, indent=2)

    # Plot
    plot_fitness(history, seed)

    return {'seed': seed, 'best_fitness': float(best_fitness), 'model_path': str(model_path)}


def plot_fitness(history, seed):
    import matplotlib.pyplot as plt

    gens = [h['generation'] for h in history]
    best = [h['best'] for h in history]
    mean = [h['mean'] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.plot(gens, best, 'b-', linewidth=2, label='Best')
    ax1.plot(gens, mean, 'g--', linewidth=1, alpha=0.7, label='Mean')
    ax1.set_ylabel('Fitness')
    ax1.set_title(f'Evolution CNN Multi-Prompt (seed {seed})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    colors = plt.cm.tab10(np.linspace(0, 1, len(BASE_PROMPTS)))
    for prompt, color in zip(BASE_PROMPTS, colors):
        vals = [h['best_breakdown'][prompt] for h in history]
        ax2.plot(gens, vals, '-', color=color, linewidth=1.5, label=prompt)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness')
    ax2.set_title('Per-Prompt Best Fitness')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = RESULTS_DIR / f'evo_cnn_multi_seed{seed:03d}.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot: {plot_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Evolve Multi-Prompt P2I (CNN)')
    parser.add_argument('--generations', type=int, default=300)
    parser.add_argument('--population-size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--runs', type=int, default=1, help='Multi-run: seeds 0..runs-1')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    if args.runs > 1:
        all_results = []
        for seed in range(args.runs):
            result = evolve_multi_prompt(
                generations=args.generations,
                population_size=args.population_size,
                seed=seed,
                device=args.device,
            )
            all_results.append(result)

        fitnesses = [r['best_fitness'] for r in all_results]
        print(f"\n{'='*60}")
        print(f"ALL {args.runs} RUNS (CNN)")
        print(f"Best: {max(fitnesses):.4f} (seed {np.argmax(fitnesses)})")
        print(f"Mean: {np.mean(fitnesses):.4f} ± {np.std(fitnesses):.4f}")
        print(f"{'='*60}")

        summary_path = RESULTS_DIR / 'evo_cnn_runs_summary.json'
        with open(summary_path, 'w') as f:
            json.dump({
                'runs': args.runs,
                'network_type': 'CNN',
                'results': all_results,
                'mean_fitness': float(np.mean(fitnesses)),
                'std_fitness': float(np.std(fitnesses)),
                'best_seed': int(np.argmax(fitnesses)),
                'best_fitness': float(max(fitnesses)),
            }, f, indent=2)
    else:
        evolve_multi_prompt(
            generations=args.generations,
            population_size=args.population_size,
            seed=args.seed,
            device=args.device,
        )


if __name__ == '__main__':
    main()
