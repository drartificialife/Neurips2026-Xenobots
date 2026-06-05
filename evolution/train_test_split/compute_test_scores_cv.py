#!/usr/bin/env python3
"""
compute_test_scores_cv.py

CV-based evaluation: replaces VLM scoring with objective kinematics.

MOTIVATION
----------
The VLM-based evaluation is potentially circular: the model was trained on
VLM scores, and evaluating it with the same VLM inflates performance estimates.

This script uses physically-measured kinematics (pre/post intervention speed
from video tracking) as ground truth. No VLM involved — just physics.

HOW IT WORKS
------------
For each (architecture, seed, archive):
  1. Load the trained P2I model
  2. For every test prompt → model predicts intervention duration D
  3. Find the nearest-neighbor batch in the archive by duration
  4. Look up that batch's kinematics from kinematics_cache.json
  5. Apply the behavioral criterion for the intended behavior:
       "stop moving" → post_speed < STOP_THRESHOLD
       "move slow"   → post_speed < SLOW_THRESHOLD
       "move fast"   → post_speed > FAST_THRESHOLD
       "go slower"   → post_speed < pre_speed * (1 - CHANGE_FRAC)
       "go faster"   → post_speed > pre_speed * (1 + CHANGE_FRAC)
  6. Score = 1.0 if criterion satisfied, 0.0 otherwise

COMPARISON WITH RANDOM BASELINE
--------------------------------
A random baseline uniformly samples a batch from the archive and applies
the same CV criterion. This gives the "chance level" given dataset bias
(e.g., if 57% of batches naturally go_slower, random gets 0.57 for that behavior).

Statistical significance is then computed as:
  - Binomial test: is model accuracy > random accuracy?
  - Cohen's h: effect size on proportions

USAGE
-----
  # Single seed
  python compute_test_scores_cv.py --arch continual --archive unseen --seed 0

  # All 30 seeds
  python compute_test_scores_cv.py --arch continual --archive unseen --runs 30

  # Run all 6 combinations
  run_cv_scoring.bat
"""

import argparse
import json
import sys
import numpy as np
import torch
from pathlib import Path
from scipy.stats import binomtest
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

SPLIT_DIR    = Path(__file__).parent
PROJECT_ROOT = SPLIT_DIR.parent.parent
MODELS_DIR   = SPLIT_DIR / 'models'
RESULTS_DIR  = SPLIT_DIR / 'results_cv'        # separate folder from VLM results
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
sys.path.insert(0, str(SPLIT_DIR))

from evolve_single_head import SingleHeadP2ICNN
from evolve_multi_prompt import MultiPromptP2ICNN
from evolve_continual   import ProgressiveP2ICNN

# ── Load prompts ──────────────────────────────────────────────────
with open(PROJECT_ROOT / 'scripts' / 'train_prompts.json') as f:
    BASE_PROMPTS = json.load(f)

with open(PROJECT_ROOT / 'scripts' / 'test_prompts.json') as f:
    TEST_PROMPTS_MAP = json.load(f)

# ── Load kinematics cache ─────────────────────────────────────────
KINEMATICS_CACHE_PATH = PROJECT_ROOT / 'cache' / 'kinematics_cache.json'
with open(KINEMATICS_CACHE_PATH) as f:
    _KINEMATICS_CACHE = json.load(f)

# ── CV behavioral thresholds ──────────────────────────────────────
# Same values used in characterize_archive.py.
# Relative criteria (go_slower, go_faster) are scale-invariant.
# Absolute criteria (stop, slow, fast) use px/sampled-frame at scale=0.5.
STOP_THRESHOLD  = 0.02    # post_speed < this → "stop moving"
SLOW_THRESHOLD  = 0.08    # post_speed < this (and >= STOP) → "move slow"
FAST_THRESHOLD  = 0.20    # post_speed > this → "move fast"
CHANGE_FRAC     = 0.15    # 15% relative change required


def cv_score(batch_id: str, base_prompt: str) -> float:
    """Return 1.0 if batch satisfies the behavioral criterion, else 0.0.

    base_prompt is one of the 5 training behaviors, used to select the criterion.
    Returns -1.0 if kinematics are unavailable for this batch.
    """
    k = _KINEMATICS_CACHE.get(batch_id)
    if k is None or 'error' in k:
        return -1.0   # no data — skip this batch

    pre  = k['pre_speed']
    post = k['post_speed']

    if base_prompt == 'stop moving':
        return 1.0 if post < STOP_THRESHOLD else 0.0

    elif base_prompt == 'move slow':
        # Absolute: post speed is low (includes stop)
        return 1.0 if post < SLOW_THRESHOLD else 0.0

    elif base_prompt == 'move fast':
        # Absolute: post speed is high
        return 1.0 if post > FAST_THRESHOLD else 0.0

    elif base_prompt == 'go slower':
        # Relative: post speed decreased by at least CHANGE_FRAC
        if pre == 0.0:
            # Was already stopped — "go slower" is trivially satisfied
            return 1.0 if post == 0.0 else 0.0
        return 1.0 if post < pre * (1.0 - CHANGE_FRAC) else 0.0

    elif base_prompt == 'go faster':
        # Relative: post speed increased by at least CHANGE_FRAC
        if pre == 0.0:
            # Started from rest — any movement is "go faster"
            return 1.0 if post > STOP_THRESHOLD else 0.0
        return 1.0 if post > pre * (1.0 + CHANGE_FRAC) else 0.0

    return -1.0  # unknown behavior


def random_baseline_score(base_prompt: str, batch_ids: list) -> float:
    """Fraction of archive batches that naturally satisfy the criterion.

    This is the expected accuracy of random batch selection — the baseline
    our model must beat to demonstrate learning.
    """
    scores = [cv_score(b, base_prompt) for b in batch_ids]
    valid  = [s for s in scores if s >= 0]
    if not valid:
        return 0.0
    return float(np.mean(valid))


# ── Archive loading (same as VLM version) ────────────────────────
def load_archive(archive_type: str):
    """Load batch IDs and build duration nearest-neighbor index."""
    path = SPLIT_DIR / ('train_archive.json' if archive_type == 'seen'
                        else 'test_archive.json')
    with open(path) as f:
        vlm_scores = json.load(f)

    batch_ids = sorted([k for k in vlm_scores
                        if vlm_scores[k].get('duration_ms') is not None])
    durations  = np.array([vlm_scores[b]['duration_ms']
                           for b in batch_ids]).reshape(-1, 1)
    nn_index   = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(durations)
    dur_min, dur_max = float(durations.min()), float(durations.max())
    return batch_ids, nn_index, dur_min, dur_max


# ── Model loading (same as VLM version) ──────────────────────────
def load_model(arch: str, seed: int, device: str):
    if arch == 'single':
        path  = MODELS_DIR / f'p2i_single_head_seed{seed:03d}.pt'
        model = SingleHeadP2ICNN().to(device)
    elif arch == 'multi':
        path  = MODELS_DIR / f'evo_cnn_multi_seed{seed:03d}.pt'
        model = MultiPromptP2ICNN(base_prompts=BASE_PROMPTS).to(device)
    else:
        path  = MODELS_DIR / f'p2i_continual_seed{seed:03d}.pt'
        model = ProgressiveP2ICNN(initial_prompts=BASE_PROMPTS).to(device)

    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


# ── Evaluate one seed ─────────────────────────────────────────────
def evaluate_seed(arch, seed, archive_type, embedder, train_embeddings, device):
    """Run CV evaluation for a single (arch, seed, archive) combination.

    Returns list of result dicts, one per test prompt.
    """
    out_path = RESULTS_DIR / f'cv_scores_{arch}_{archive_type}_seed{seed:03d}.json'

    batch_ids, nn_index, dur_min, dur_max = load_archive(archive_type)

    try:
        model = load_model(arch, seed, device)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return None

    # Resume: skip if already computed
    if out_path.exists():
        existing = json.load(open(out_path))
        print(f"  Seed {seed:03d}: already done ({len(existing.get('results', []))} prompts)")
        return existing.get('results', [])

    results = []
    with torch.no_grad():
        for base_prompt, test_list in TEST_PROMPTS_MAP.items():
            for test_prompt in test_list:

                # Step 1: embed test prompt, find closest base prompt head
                emb  = embedder.encode(test_prompt)
                sims = cosine_similarity([emb], train_embeddings)[0]
                best_idx   = int(np.argmax(sims))
                selected   = BASE_PROMPTS[best_idx]       # head chosen by similarity
                similarity = float(sims[best_idx])

                # Step 2: model predicts normalized duration → denormalize
                tensor    = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
                if arch == 'single':
                    pred_norm = model(tensor).squeeze().item()
                else:
                    outputs   = model(tensor)
                    pred_norm = outputs[selected].squeeze().item()

                duration = pred_norm * (dur_max - dur_min) + dur_min

                # Step 3: find nearest batch in archive by duration
                arr      = np.array([duration]).reshape(1, -1)
                _, idxs  = nn_index.kneighbors(arr)
                batch_id = batch_ids[min(idxs[0][0], len(batch_ids) - 1)]

                # Step 4: CV criterion check — uses the INTENDED base_prompt,
                # not the selected head. This tests whether the model actually
                # achieves the goal, regardless of which head it chose.
                score = cv_score(batch_id, base_prompt)

                results.append({
                    'prompt':          test_prompt,
                    'base_prompt':     base_prompt,      # intended behavior
                    'selected_head':   selected,         # head chosen by cosine sim
                    'head_correct':    int(selected == base_prompt),
                    'head_similarity': similarity,
                    'predicted_duration': float(duration),
                    'nearest_batch':   batch_id,
                    'cv_score':        score,            # 1.0 / 0.0 / -1.0 (missing)
                })

                status = '[OK]' if score == 1.0 else ('[??]' if score < 0 else '[--]')
                print(f"    {status} {test_prompt:<28} → {batch_id}  cv={score:.0f}")

    # Save
    json.dump({'seed': seed, 'arch': arch, 'archive': archive_type,
               'results': results}, open(out_path, 'w'), indent=2)
    return results


# ── Aggregate results + statistics ───────────────────────────────
def compute_stats(all_results_per_seed: list, archive_type: str):
    """Per-behavior accuracy and comparison to random baseline.

    Returns dict with per-behavior and overall statistics.
    """
    batch_ids, _, _, _ = load_archive(archive_type)

    # Random baseline per behavior
    baselines = {bp: random_baseline_score(bp, batch_ids)
                 for bp in TEST_PROMPTS_MAP}

    # Per-seed, per-behavior accuracy (exclude missing cv_score == -1)
    behavior_scores_per_seed = {bp: [] for bp in TEST_PROMPTS_MAP}
    for seed_results in all_results_per_seed:
        if seed_results is None:
            continue
        by_behavior = {bp: [] for bp in TEST_PROMPTS_MAP}
        for r in seed_results:
            if r['cv_score'] < 0:
                continue   # no kinematics data for this batch
            by_behavior[r['base_prompt']].append(r['cv_score'])
        for bp, scores in by_behavior.items():
            if scores:
                behavior_scores_per_seed[bp].append(np.mean(scores))

    stats = {}
    for bp in TEST_PROMPTS_MAP:
        seed_accs  = behavior_scores_per_seed[bp]
        if not seed_accs:
            continue
        model_acc  = float(np.mean(seed_accs))
        model_std  = float(np.std(seed_accs))
        baseline   = baselines[bp]

        # Binomial test: is model accuracy above random?
        # Use mean accuracy × n_prompts as successes over n_prompts trials
        n_prompts  = len(TEST_PROMPTS_MAP[bp])
        n_success  = round(model_acc * n_prompts)
        btest      = binomtest(n_success, n_prompts, p=baseline, alternative='greater')

        # Cohen's h (effect size for proportions)
        h = 2 * (np.arcsin(np.sqrt(model_acc)) - np.arcsin(np.sqrt(baseline)))

        stats[bp] = {
            'model_accuracy':    model_acc,
            'model_std':         model_std,
            'random_baseline':   baseline,
            'improvement':       model_acc - baseline,
            'p_value':           float(btest.pvalue),
            'cohen_h':           float(h),
            'n_seeds':           len(seed_accs),
        }

    return stats


# ── Main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',    choices=['single', 'multi', 'continual'], required=True)
    parser.add_argument('--archive', choices=['seen', 'unseen'],               required=True)
    parser.add_argument('--seed',    type=int, default=None)
    parser.add_argument('--runs',    type=int, default=30)
    parser.add_argument('--device',  type=str, default='cpu')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    print(f"\nArch={args.arch}  Archive={args.archive}  Device={args.device}")
    print(f"Evaluation: CV kinematics (no VLM)")
    print("=" * 60)

    embedder         = SentenceTransformer('all-MiniLM-L6-v2')
    train_embeddings = embedder.encode(BASE_PROMPTS)
    seeds = [args.seed] if args.seed is not None else list(range(args.runs))

    # Run per-seed evaluation
    all_results = []
    for seed in seeds:
        print(f"\nSeed {seed:03d}")
        results = evaluate_seed(
            args.arch, seed, args.archive,
            embedder, train_embeddings, args.device
        )
        all_results.append(results)

    # Compute statistics
    valid_results = [r for r in all_results if r is not None]
    if not valid_results:
        print("No results to analyze.")
        return

    print(f"\n{'='*60}")
    print(f"SUMMARY — Arch={args.arch}  Archive={args.archive}")
    print(f"{'='*60}")

    stats = compute_stats(valid_results, args.archive)

    print(f"\n{'Behavior':<14} {'Model':>7} {'±std':>6} {'Random':>8} {'Δ':>6} {'p-val':>8} {'Cohen h':>9}")
    print(f"{'-'*64}")
    for bp, s in stats.items():
        sig = '*' if s['p_value'] < 0.05 else ' '
        print(f"{bp:<14} {s['model_accuracy']:>6.3f}  {s['model_std']:>5.3f}  "
              f"{s['random_baseline']:>7.3f}  {s['improvement']:>+5.3f}  "
              f"{s['p_value']:>7.4f}{sig}  {s['cohen_h']:>8.3f}")

    # Overall accuracy (all behaviors combined)
    all_model_accs   = [s['model_accuracy']  for s in stats.values()]
    all_baselines    = [s['random_baseline'] for s in stats.values()]
    overall_model    = float(np.mean(all_model_accs))
    overall_baseline = float(np.mean(all_baselines))
    print(f"\n{'Overall':<14} {overall_model:>6.3f}  {'':>5}  "
          f"{overall_baseline:>7.3f}  {overall_model-overall_baseline:>+5.3f}")

    # Save summary
    summary = {
        'arch':             args.arch,
        'archive':          args.archive,
        'n_seeds':          len(valid_results),
        'overall_accuracy': overall_model,
        'overall_baseline': overall_baseline,
        'per_behavior':     stats,
    }
    summary_path = RESULTS_DIR / f'summary_cv_{args.arch}_{args.archive}.json'
    json.dump(summary, open(summary_path, 'w'), indent=2)
    print(f"\n[OK] {summary_path.name}")


if __name__ == '__main__':
    main()
