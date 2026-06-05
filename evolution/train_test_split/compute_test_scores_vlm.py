#!/usr/bin/env python3
"""
Compute VLM test scores for train/test split experiment.

Real VLM evaluation: test_prompt + trajectory image → VLM → score

Two evaluation modes:
  --archive seen    → train_archive.json (111 seen batches)   = Test A
  --archive unseen  → test_archive.json  (28 unseen batches)  = Test B

Three architectures:
  --arch single     → SingleHeadP2ICNN
  --arch multi      → MultiPromptP2ICNN
  --arch continual  → ProgressiveP2ICNN

Usage:
  python compute_test_scores_vlm.py --arch continual --archive unseen --seed 0 --device cuda
  python compute_test_scores_vlm.py --arch continual --archive unseen --runs 30 --device cuda
"""

import argparse
import json
import sys
import time
import numpy as np
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

SPLIT_DIR    = Path(__file__).parent
PROJECT_ROOT = SPLIT_DIR.parent.parent
MODELS_DIR   = SPLIT_DIR / 'models'
RESULTS_DIR  = SPLIT_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BOT_TRAJECTORY_DIR = PROJECT_ROOT / 'bot_trajectory'

sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
sys.path.insert(0, str(SPLIT_DIR))

from vlm_interpret_trajectory import (
    SCENE_CONTEXT, encode_image_b64, ollama_query, parse_score_from_response, MODELS,
)
from evolve_single_head import SingleHeadP2ICNN
from evolve_multi_prompt import MultiPromptP2ICNN
from evolve_continual   import ProgressiveP2ICNN

with open(PROJECT_ROOT / 'scripts' / 'train_prompts.json') as f:
    BASE_PROMPTS = json.load(f)
with open(PROJECT_ROOT / 'scripts' / 'test_prompts.json') as f:
    TEST_PROMPTS_MAP = json.load(f)

VLM_MODEL_KEY = "qwen3.5:397b-cloud"
VLM_CFG = MODELS[VLM_MODEL_KEY]


# ── VLM prompt builder ────────────────────────────────────────────
def build_vlm_prompt(test_prompt: str, base_prompt: str) -> str:
    GROUP_TEMPLATES = {
        "stop moving": (
            f"Did the intervention cause the bot to {test_prompt}? "
            "If the post (red) trajectory is very short or clustered in one spot, the bot stopped. "
            "Compare pre (blue) total distance vs post (red) total distance from the legend. "
            "Summarize the behavioral change in 1-2 sentences. "
            "Then, rate the effectiveness on a scale of 0-1 (1 = completely stopped, 0 = no change or moved more). "
            "Format: <summary>\nScore: <number>"
        ),
        "move slow": (
            f"Is the bot moving slowly after intervention (i.e. did it '{test_prompt}')? "
            "Check the post (red) total distance: a small total distance means slow movement. "
            "This is about absolute post-intervention speed, not relative change. "
            "Summarize the post-intervention motion in 1-2 sentences. "
            "Then, rate how slow the bot is moving post-intervention on a scale of 0-1 "
            "(1 = very slow/nearly stopped, 0 = fast movement). "
            "Format: <summary>\nScore: <number>"
        ),
        "move fast": (
            f"Is the bot moving faster after intervention (i.e. did it '{test_prompt}')? "
            "Check the post (red) total distance vs pre (blue) total distance. "
            "This is about absolute post-intervention speed, not relative change. "
            "Summarize the post-intervention motion in 1-2 sentences. "
            "Then, rate on a scale of 0-1 (1 = clearly faster movement, 0 = no change or slower). "
            "Format: <summary>\nScore: <number>"
        ),
        "go slower": (
            f"Did the intervention cause the bot to {test_prompt}? "
            "Compare pre (blue) total distance vs post (red) total distance. "
            "A reduction means it went slower. "
            "Summarize the change in 1-2 sentences. "
            "Then, rate on a scale of 0-1 (1 = clearly slower than before, 0 = no change or faster). "
            "Format: <summary>\nScore: <number>"
        ),
        "go faster": (
            f"Did the intervention cause the bot to {test_prompt}? "
            "Compare pre (blue) total distance vs post (red) total distance. "
            "An increase means it went faster. "
            "Summarize the change in 1-2 sentences. "
            "Then, rate the effectiveness on a scale of 0-1 (1 = major speed increase, 0 = no change or slowed down). "
            "Format: <summary>\nScore: <number>"
        ),
    }
    body = GROUP_TEMPLATES.get(base_prompt)
    if body is None:
        raise ValueError(f"Unknown base_prompt: {base_prompt}")
    return f"{SCENE_CONTEXT} {body}"


# ── VLM scorer ────────────────────────────────────────────────────
def _score_one(task):
    image_path = BOT_TRAJECTORY_DIR / f"{task['nearest_batch']}_trajectory_heatmap.png"
    image_b64  = encode_image_b64(image_path)
    prompt_txt = build_vlm_prompt(task['prompt'], task['selected_head'])
    response   = ollama_query(
        image_b64, prompt_txt,
        VLM_CFG['model'], VLM_CFG['host'],
        timeout_sec=600, api_mode=VLM_CFG['api_mode'],
    )
    score = parse_score_from_response(response)
    return {**task, 'vlm_score': score, 'vlm_response': response}


# ── Load archive ──────────────────────────────────────────────────
def load_archive(archive_type: str):
    path = SPLIT_DIR / ('train_archive.json' if archive_type == 'seen' else 'test_archive.json')
    with open(path) as f:
        vlm_scores = json.load(f)
    batch_ids = sorted([k for k in vlm_scores if vlm_scores[k].get('duration_ms') is not None])
    durations = np.array([vlm_scores[b]['duration_ms'] for b in batch_ids]).reshape(-1, 1)
    nn_index  = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(durations)
    dur_min, dur_max = float(durations.min()), float(durations.max())
    print(f"Archive ({archive_type}): {len(batch_ids)} batches, [{dur_min:.0f}, {dur_max:.0f}] ms")
    return batch_ids, nn_index, dur_min, dur_max


# ── Load model ────────────────────────────────────────────────────
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
def evaluate_seed(arch, seed, archive_type, embedder, train_embeddings, workers, device):
    out_path = RESULTS_DIR / f'test_scores_vlm_{arch}_{archive_type}_seed{seed:03d}.json'

    batch_ids, nn_index, dur_min, dur_max = load_archive(archive_type)

    try:
        model = load_model(arch, seed, device)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return None

    # Resume support
    done_prompts, results = set(), []
    if out_path.exists():
        existing = json.load(open(out_path))
        results  = existing.get('results', [])
        done_prompts = {r['prompt'] for r in results}
        print(f"  Resuming: {len(done_prompts)} done")

    # Phase 1: P2I predictions
    tasks = []
    with torch.no_grad():
        for base_prompt, test_list in TEST_PROMPTS_MAP.items():
            for test_prompt in test_list:
                if test_prompt in done_prompts:
                    continue
                emb  = embedder.encode(test_prompt)
                sims = cosine_similarity([emb], train_embeddings)[0]
                best_idx     = int(np.argmax(sims))
                selected     = BASE_PROMPTS[best_idx]
                similarity   = float(sims[best_idx])

                tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
                if arch == 'single':
                    pred_norm = model(tensor).squeeze().item()
                else:
                    outputs   = model(tensor)
                    pred_norm = outputs[selected].squeeze().item()

                duration = pred_norm * (dur_max - dur_min) + dur_min
                arr = np.array([duration]).reshape(1, -1)
                _, indices = nn_index.kneighbors(arr)
                batch_id = batch_ids[min(indices[0][0], len(batch_ids) - 1)]

                tasks.append({
                    'prompt':             test_prompt,
                    'base_prompt':        base_prompt,
                    'selected_head':      selected,
                    'head_similarity':    similarity,
                    'predicted_duration': float(duration),
                    'nearest_batch':      batch_id,
                })

    if not tasks:
        print(f"  Seed {seed:03d}: already complete ({len(results)} prompts)")
        return results

    # Phase 2: VLM scoring
    print(f"  Seed {seed:03d}: {len(tasks)} VLM calls ({workers} workers)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_score_one, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            entry = future.result()
            results.append(entry)
            print(f"    [{i}/{len(tasks)}] {entry['prompt']:25s} -> score={entry['vlm_score']:.2f}")
            json.dump({'seed': seed, 'arch': arch, 'archive': archive_type, 'results': results},
                      open(out_path, 'w'), indent=2)

    elapsed = time.time() - t0
    all_scores = [r['vlm_score'] for r in results]
    print(f"  Done in {elapsed:.0f}s | mean={np.mean(all_scores):.4f}")
    return results


# ── Main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',    choices=['single', 'multi', 'continual'], required=True)
    parser.add_argument('--archive', choices=['seen', 'unseen'],               required=True)
    parser.add_argument('--seed',    type=int, default=None)
    parser.add_argument('--runs',    type=int, default=30)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device',  type=str, default='cpu')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    print(f"\nArch={args.arch}  Archive={args.archive}  Device={args.device}")
    print("=" * 60)

    embedder         = SentenceTransformer('all-MiniLM-L6-v2')
    train_embeddings = embedder.encode(BASE_PROMPTS)
    seeds = [args.seed] if args.seed is not None else list(range(args.runs))

    all_means = []
    for seed in seeds:
        print(f"\nSeed {seed:03d}")
        results = evaluate_seed(
            args.arch, seed, args.archive,
            embedder, train_embeddings, args.workers, args.device
        )
        if results:
            mean = np.mean([r['vlm_score'] for r in results])
            all_means.append(mean)

    if all_means:
        print(f"\n{'='*60}")
        print(f"Arch={args.arch}  Archive={args.archive}  n={len(all_means)}")
        print(f"Mean: {np.mean(all_means):.4f} +/- {np.std(all_means):.4f}")
        print(f"{'='*60}")

        summary_path = RESULTS_DIR / f'summary_vlm_{args.arch}_{args.archive}.json'
        json.dump({
            'arch': args.arch, 'archive': args.archive,
            'n_seeds': len(all_means),
            'mean_score': float(np.mean(all_means)),
            'std_score':  float(np.std(all_means)),
            'per_seed_means': [float(m) for m in all_means],
        }, open(summary_path, 'w'), indent=2)
        print(f"[OK] {summary_path.name}")


if __name__ == '__main__':
    main()
