#!/usr/bin/env python3
"""
Compute test scores for train/test split experiment.

Pure lookup — no live VLM calls needed (scores pre-computed in both archives).

Two evaluation modes:
  --archive seen    → train_archive.json  (111 seen batches)   = Test A
  --archive unseen  → test_archive.json   (28 unseen batches)  = Test B

Three architectures:
  --arch single     → SingleHeadP2ICNN
  --arch multi      → MultiHeadP2ICNN
  --arch continual  → ProgressiveP2ICNN

Usage:
  python compute_test_scores.py --arch continual --archive unseen --runs 30 --device cuda
"""

import argparse
import json
import sys
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

SPLIT_DIR    = Path(__file__).parent
PROJECT_ROOT = SPLIT_DIR.parent.parent
MODELS_DIR   = SPLIT_DIR / 'models'
RESULTS_DIR  = SPLIT_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SPLIT_DIR))

from evolve_single_head import SingleHeadP2ICNN
from evolve_multi_prompt import MultiPromptP2ICNN
from evolve_continual   import ProgressiveP2ICNN

with open(PROJECT_ROOT / 'scripts' / 'train_prompts.json') as f:
    BASE_PROMPTS = json.load(f)

with open(PROJECT_ROOT / 'scripts' / 'test_prompts.json') as f:
    TEST_PROMPTS_MAP = json.load(f)


# ── Load archive ──────────────────────────────────────────────────
def load_archive(archive_type: str):
    path = SPLIT_DIR / ('train_archive.json' if archive_type == 'seen' else 'test_archive.json')
    with open(path) as f:
        vlm_scores = json.load(f)
    batch_ids = sorted([k for k in vlm_scores if vlm_scores[k].get('duration_ms') is not None])
    durations = np.array([vlm_scores[b]['duration_ms'] for b in batch_ids]).reshape(-1, 1)
    nn_index  = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(durations)
    dur_min, dur_max = float(durations.min()), float(durations.max())
    print(f"Archive ({archive_type}): {len(batch_ids)} batches, duration [{dur_min:.0f}, {dur_max:.0f}] ms")
    return vlm_scores, batch_ids, nn_index, dur_min, dur_max


# ── Load model ────────────────────────────────────────────────────
def load_model(arch: str, seed: int, device: str):
    if arch == 'single':
        path = MODELS_DIR / f'p2i_single_head_seed{seed:03d}.pt'
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = SingleHeadP2ICNN().to(device)
    elif arch == 'multi':
        path = MODELS_DIR / f'evo_cnn_multi_seed{seed:03d}.pt'
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = MultiPromptP2ICNN(base_prompts=BASE_PROMPTS).to(device)
    else:  # continual
        path = MODELS_DIR / f'p2i_continual_seed{seed:03d}.pt'
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = ProgressiveP2ICNN(initial_prompts=BASE_PROMPTS).to(device)

    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


# ── Predict duration for a test prompt ───────────────────────────
def predict_duration(arch, model, emb, train_embeddings, dur_min, dur_max, device):
    sims         = cosine_similarity([emb], train_embeddings)[0]
    best_idx     = int(np.argmax(sims))
    selected     = BASE_PROMPTS[best_idx]
    similarity   = float(sims[best_idx])

    tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        if arch == 'single':
            pred_norm = model(tensor).squeeze().item()
        else:
            outputs   = model(tensor)
            pred_norm = outputs[selected].squeeze().item()

    duration = pred_norm * (dur_max - dur_min) + dur_min
    return duration, selected, similarity


# ── Evaluate one seed ─────────────────────────────────────────────
def evaluate_seed(arch, seed, archive_type, embedder, train_embeddings, device):
    vlm_scores, batch_ids, nn_index, dur_min, dur_max = load_archive(archive_type)

    try:
        model = load_model(arch, seed, device)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return None

    results = []
    for base_prompt, test_list in TEST_PROMPTS_MAP.items():
        for test_prompt in test_list:
            emb = embedder.encode(test_prompt)
            duration, selected_head, similarity = predict_duration(
                arch, model, emb, train_embeddings, dur_min, dur_max, device
            )

            arr = np.array([duration]).reshape(1, -1)
            _, indices = nn_index.kneighbors(arr)
            batch_id   = batch_ids[min(indices[0][0], len(batch_ids) - 1)]

            score_entry = vlm_scores[batch_id].get(base_prompt, {})
            vlm_score   = float(score_entry.get('score', 0.0)) if isinstance(score_entry, dict) else 0.0

            results.append({
                'prompt':            test_prompt,
                'base_prompt':       base_prompt,
                'selected_head':     selected_head,
                'head_similarity':   similarity,
                'predicted_duration': float(duration),
                'nearest_batch':     batch_id,
                'vlm_score':         vlm_score,
            })

    all_scores  = [r['vlm_score'] for r in results]
    mean_score  = float(np.mean(all_scores))
    return {'seed': seed, 'mean_score': mean_score, 'results': results}


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
    print("=" * 60)

    embedder         = SentenceTransformer('all-MiniLM-L6-v2')
    train_embeddings = embedder.encode(BASE_PROMPTS)

    seeds = [args.seed] if args.seed is not None else list(range(args.runs))

    all_results, all_scores = [], []
    for seed in seeds:
        print(f"\nSeed {seed:03d}...")
        res = evaluate_seed(args.arch, seed, args.archive, embedder, train_embeddings, args.device)
        if res is None:
            continue
        all_results.append(res)
        all_scores.append(res['mean_score'])
        print(f"  Score: {res['mean_score']:.4f}")

    print(f"\n{'='*60}")
    print(f"Arch={args.arch}  Archive={args.archive}  Seeds={len(all_scores)}")
    print(f"Mean: {np.mean(all_scores):.4f} +/- {np.std(all_scores):.4f}")
    print(f"{'='*60}\n")

    out_file = RESULTS_DIR / f'test_scores_{args.arch}_{args.archive}.json'
    with open(out_file, 'w') as f:
        json.dump({
            'arch':       args.arch,
            'archive':    args.archive,
            'n_seeds':    len(all_scores),
            'mean_score': float(np.mean(all_scores)),
            'std_score':  float(np.std(all_scores)),
            'per_seed':   all_results,
        }, f, indent=2)
    print(f"[OK] Saved: {out_file.name}")


if __name__ == '__main__':
    main()
