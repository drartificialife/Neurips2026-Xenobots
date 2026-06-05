#!/usr/bin/env python3
"""
Continual Learning with Curriculum — TRAIN/TEST SPLIT VERSION.

Uses only train_archive.json (111 batches, 80% of full archive).
The remaining 28 batches (test_archive.json) are held out for video generalization test.

Approach:
  Phase 1: Train on 1 prompt (easy, high fitness baseline)
  Phase 2: Add head for prompt 2, fine-tune both heads
  Phase 3: Add head for prompt 3, fine-tune 3 heads
  ...
  Phase 5: All 5 prompts trained

Transfer learning: Shared backbone carries forward learned features.
Curriculum: Progressively harder (2 competing tasks → 5 competing tasks).

Usage:
    python evolution/train_test_split/evolve_continual.py --seed 0 --device cuda
"""

import argparse
import copy
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List

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
# Network — Multi-Head CNN that grows over time
# ---------------------------------------------------------------------------
class ProgressiveP2ICNN(nn.Module):
    """Multi-head CNN that supports adding heads dynamically.

    Backbone: shared features (frozen or fine-tuned)
    Heads: one per active prompt (added progressively)
    """

    def __init__(self, initial_prompts: List[str] = None, embedding_dim: int = 384):
        super().__init__()
        if initial_prompts is None:
            initial_prompts = []
        self.active_prompts = initial_prompts.copy()

        # 1D Conv backbone
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

        # Shared FC
        self.shared_fc = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Output heads (one per prompt)
        self.heads = nn.ModuleDict({
            _prompt_to_key(p): nn.Linear(32, 1) for p in initial_prompts
        })

    def add_head(self, prompt: str):
        """Add a new output head for a new prompt."""
        if prompt not in self.active_prompts:
            self.active_prompts.append(prompt)
            key = _prompt_to_key(prompt)
            device = next(self.parameters()).device
            self.heads[key] = nn.Linear(32, 1).to(device)
            # Initialize new head with small random weights
            nn.init.normal_(self.heads[key].weight, mean=0, std=0.01)
            nn.init.constant_(self.heads[key].bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = x.unsqueeze(1)  # (batch, 1, 384)
        x = self.conv(x)  # (batch, 16, 384)
        x = x.mean(dim=2)  # GAP → (batch, 16)
        x = self.shared_fc(x)  # (batch, 32)

        outputs = {}
        for prompt in self.active_prompts:
            key = _prompt_to_key(prompt)
            outputs[prompt] = torch.sigmoid(self.heads[key](x))
        return outputs


# ---------------------------------------------------------------------------
# Archive / fitness (same as multi-head)
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
            return 0.0, batch_id
        return float(score_entry.get('score', 0.0)), batch_id
    return 0.0, batch_id


def evaluate_network(network, embedder, vlm_scores, batch_ids, nn_index, dur_min, dur_max, device='cpu'):
    """Evaluate on all ACTIVE prompts."""
    network.eval()
    with torch.no_grad():
        emb = embedder.encode(BASE_PROMPTS[0], convert_to_tensor=False)
        tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
        outputs = network(tensor)

        fitness_by_prompt = {}
        for prompt in network.active_prompts:
            pred_norm = outputs[prompt].squeeze().item()
            score, _ = fitness_lookup(pred_norm, prompt, vlm_scores, batch_ids, nn_index, dur_min, dur_max)
            fitness_by_prompt[prompt] = score

        avg = np.mean(list(fitness_by_prompt.values())) if fitness_by_prompt else 0.0
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
# Continual Learning Evolution
# ---------------------------------------------------------------------------
def evolve_continual(
    generations_per_phase: int = 100,
    population_size: int = 100,
    seed: int = 0,
    device: str = 'cpu',
    patience: int = 15,  # Early stopping: no improvement for N gens → move to next phase
) -> Dict:
    """
    Progressive curriculum: add one prompt at a time.

    Phase 1: Prompt 0 only (easy baseline)
    Phase 2: Prompts 0-1 (add head, fine-tune)
    Phase 3: Prompts 0-2 (add head, fine-tune)
    ...
    Phase 5: All prompts 0-4
    """
    set_seed(seed)

    print(f"\n{'='*70}")
    print(f"Continual Learning Evolution (Progressive Multi-Head CNN)")
    print(f"Seed: {seed} | Gens/Phase: {generations_per_phase} | Pop: {population_size}")
    print(f"{'='*70}\n")

    vlm_scores, batch_ids, nn_index, dur_min, dur_max = load_archive()
    print(f"Archive: {len(batch_ids)} batches")

    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    all_history = {}
    best_network = None
    global_best_fitness = -float('inf')

    # Phase-by-phase training
    for phase, target_prompt_idx in enumerate(range(len(BASE_PROMPTS))):
        target_prompts = BASE_PROMPTS[:target_prompt_idx + 1]
        num_prompts = len(target_prompts)

        print(f"\n{'─'*70}")
        print(f"PHASE {phase + 1}/{len(BASE_PROMPTS)}: Training on {num_prompts} prompt(s)")
        print(f"Prompts: {target_prompts}")
        print(f"{'─'*70}")

        # Initialize or continue from previous phase
        if phase == 0:
            # Phase 1: Start fresh
            population = [ProgressiveP2ICNN(initial_prompts=target_prompts).to(device)
                         for _ in range(population_size)]
        else:
            # Phase 2+: Load best from previous phase, add new head
            best_prev = copy.deepcopy(best_network)
            new_prompt = BASE_PROMPTS[target_prompt_idx]
            best_prev.add_head(new_prompt)

            # Population: best_prev + mutations
            population = [best_prev]
            for _ in range(population_size - 1):
                child = mutate(best_prev, strength=0.03)
                population.append(child)

        phase_best_fitness = -float('inf')
        phase_history = []
        best_fitness_no_improve = 0  # Counter for early stopping

        # Evolve for this phase
        for gen in range(generations_per_phase):
            gen_fitness = []
            gen_breakdown = []

            for net in population:
                avg_f, breakdown = evaluate_network(
                    net, embedder, vlm_scores, batch_ids, nn_index, dur_min, dur_max, device
                )
                gen_fitness.append(avg_f)
                gen_breakdown.append(breakdown)

            gen_fitness = np.array(gen_fitness)
            best_idx = gen_fitness.argmax()

            if gen_fitness[best_idx] > phase_best_fitness:
                phase_best_fitness = gen_fitness[best_idx]
                best_network = copy.deepcopy(population[best_idx])
                best_fitness_no_improve = 0  # Reset counter
            else:
                best_fitness_no_improve += 1  # Increment no-improvement counter

            if gen_fitness[best_idx] > global_best_fitness:
                global_best_fitness = gen_fitness[best_idx]

            if (gen + 1) % 10 == 0 or gen == 0:
                breakdown = gen_breakdown[best_idx]
                parts = ', '.join([f'{p}: {breakdown[p]:.3f}' for p in target_prompts])
                status = f" [no improve: {best_fitness_no_improve}/{patience}]" if best_fitness_no_improve > 0 else ""
                print(f"  Gen {gen+1:3d} | best={gen_fitness[best_idx]:.4f} "
                      f"mean={gen_fitness.mean():.4f} | {parts}{status}")

            # Early stopping: if no improvement for 'patience' generations, move to next phase
            if best_fitness_no_improve >= patience:
                print(f"  → Early stopping: no improvement for {patience} gens. Moving to next phase.")
                break

            # Get best breakdown (per-task fitness) like multi-head does
            best_breakdown = gen_breakdown[best_idx]

            phase_history.append({
                'generation': gen + 1,
                'best': float(gen_fitness.max()),
                'mean': float(gen_fitness.mean()),
                'worst': float(gen_fitness.min()),
                'best_breakdown': {k: float(v) for k, v in best_breakdown.items()},
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

        # Save only per-phase summary (not full history)
        # This allows easy aggregation across 30 seeds despite different early-stopping points
        final_entry = phase_history[-1] if phase_history else {}
        final_breakdown = final_entry.get('best_breakdown', {})

        all_history[f"phase_{phase+1}"] = {
            'n_prompts': num_prompts,
            'prompts': target_prompts,
            'n_gens': len(phase_history),  # Actual gens this phase (with early stop)
            'final_fitness': float(phase_best_fitness),
            'per_task_final': {k: float(v) for k, v in final_breakdown.items()},
        }

        print(f"Phase {phase + 1} complete: best_fitness={phase_best_fitness:.4f}\n")

    # Final
    print(f"\n{'='*70}")
    print(f"CONTINUAL LEARNING COMPLETE")
    print(f"Global best fitness: {global_best_fitness:.4f}")
    print(f"Final network: {len(best_network.active_prompts)} prompts")
    print(f"{'='*70}\n")

    seed_tag = f"seed{seed:03d}"

    model_path = MODELS_DIR / f'p2i_continual_{seed_tag}.pt'
    torch.save({
        'model_state_dict': best_network.state_dict(),
        'active_prompts': best_network.active_prompts,
        'best_fitness': float(global_best_fitness),
        'base_prompts': BASE_PROMPTS,
        'seed': seed,
        'network_type': 'CNN_continual',
    }, model_path)

    results_path = RESULTS_DIR / f'evo_continual_{seed_tag}.json'
    with open(results_path, 'w') as f:
        json.dump({
            'seed': seed,
            'network_type': 'CNN_continual',
            'global_best_fitness': float(global_best_fitness),
            'num_phases_completed': len(all_history),
            'phases': all_history,
        }, f, indent=2)

    print(f"Model: {model_path}")
    print(f"Results: {results_path}\n")

    return {
        'seed': seed,
        'best_fitness': float(global_best_fitness),
        'model_path': str(model_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Continual Learning CNN P2I')
    parser.add_argument('--generations-per-phase', type=int, default=100)
    parser.add_argument('--population-size', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping: gens with no improvement before moving to next phase')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--runs', type=int, default=1, help='Multi-run: seeds 0..runs-1')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        print("[!] CUDA not available, using CPU")

    if args.runs > 1:
        all_results = []
        for seed in range(args.runs):
            result = evolve_continual(
                generations_per_phase=args.generations_per_phase,
                population_size=args.population_size,
                patience=args.patience,
                seed=seed,
                device=args.device,
            )
            all_results.append(result)

        fitnesses = [r['best_fitness'] for r in all_results]
        print(f"\n{'='*60}")
        print(f"ALL {args.runs} RUNS (Continual Learning)")
        print(f"Best: {max(fitnesses):.4f} (seed {np.argmax(fitnesses)})")
        print(f"Mean: {np.mean(fitnesses):.4f} +/- {np.std(fitnesses):.4f}")
        print(f"{'='*60}\n")

        summary_path = RESULTS_DIR / 'evo_continual_summary.json'
        with open(summary_path, 'w') as f:
            json.dump({
                'runs': args.runs,
                'network_type': 'continual_learning',
                'fitnesses': [float(f) for f in fitnesses],
                'mean_fitness': float(np.mean(fitnesses)),
                'std_fitness': float(np.std(fitnesses)),
                'best_fitness': float(max(fitnesses)),
                'results': all_results,
            }, f, indent=2)
        print(f"Summary: {summary_path}\n")
    else:
        evolve_continual(
            generations_per_phase=args.generations_per_phase,
            population_size=args.population_size,
            patience=args.patience,
            seed=args.seed,
            device=args.device,
        )


if __name__ == '__main__':
    main()
