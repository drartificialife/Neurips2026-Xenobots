#!/usr/bin/env python3
"""
Pre-compute VLM fitness scores for all 30 prompt variants across all archived batches.
Uses qwen3.5:397b-cloud with concurrent requests (rate-limit aware).

Reads trajectory heatmaps from bot_trajectory/.
Loads prompts from evolution/train_test_split/prompts.json (5 behaviors × 6 variants).
Saves to cache/archive_vlm_scores_all_prompts.json.

Usage:
    python scripts/precompute_archive_vlm_scores.py
    python scripts/precompute_archive_vlm_scores.py --workers 4
    python scripts/precompute_archive_vlm_scores.py --batch batch-000148  # single batch
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from vlm_interpret_trajectory import interpret_trajectory

# Config
MODEL_KEY = "qwen3.5:397b-cloud"
BOT_TRAJECTORY_DIR = Path('bot_trajectory')
ARCHIVE_ROOT = Path('D:/xenobot_videos')
CACHE_DIR = Path('cache')
CACHE_DIR.mkdir(exist_ok=True)
SCORES_FILE = CACHE_DIR / 'archive_vlm_scores_all_prompts.json'

# Load prompts from evolution/train_test_split/prompts.json (all 30 variants)
PROMPTS_FILE = Path('evolution/train_test_split/prompts.json')
with open(PROMPTS_FILE) as f:
    prompts_dict = json.load(f)

# Flatten: {behavior: [variants...]} -> [all variants...]
TRAIN_PROMPTS = []
for behavior in sorted(prompts_dict.keys()):
    TRAIN_PROMPTS.extend(prompts_dict[behavior])


def get_duration_ms(batch_id: str) -> int | None:
    """Read duration_ms from command metadata if available."""
    commands_dir = ARCHIVE_ROOT / batch_id / 'commands'
    if not commands_dir.exists():
        return None
    electrical_files = list(commands_dir.glob('electrical_*.json'))
    if not electrical_files:
        return None
    try:
        with open(electrical_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data['instructions'][0]['duration_ms']
    except Exception:
        return None


def get_trajectory_image(batch_id: str) -> Path | None:
    path = BOT_TRAJECTORY_DIR / f"{batch_id}_trajectory_heatmap.png"
    return path if path.exists() else None


def process_task(task: Tuple[str, str], max_retries: int = 3) -> Dict[str, Any] | None:
    """Process one (batch_id, prompt) task with retries."""
    batch_id, prompt = task

    image_path = get_trajectory_image(batch_id)
    if not image_path:
        return None

    for attempt in range(max_retries):
        try:
            result = interpret_trajectory(str(image_path), prompt, model_key=MODEL_KEY)
            score = result.get('score', -1.0)
            desc = result.get('description', '')

            # Check for API errors worth retrying
            if isinstance(desc, str) and desc.startswith('Error:'):
                if ('429' in desc or '524' in desc) and attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)
                    print(f"    [{batch_id}|{prompt}] Retry {attempt+1}, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                # Non-retryable error
                return {'batch_id': batch_id, 'prompt': prompt,
                        'score': 0.0, 'desc': desc}

            if score < 0:
                score = 0.0

            return {'batch_id': batch_id, 'prompt': prompt,
                    'score': float(score), 'desc': desc}

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3)
                continue
            return {'batch_id': batch_id, 'prompt': prompt,
                    'score': 0.0, 'desc': f'Error: {e}'}

    return {'batch_id': batch_id, 'prompt': prompt,
            'score': 0.0, 'desc': 'Error: All retries exhausted'}


def load_existing() -> Dict[str, Any]:
    """Load from both new file and legacy v2 file, merge them."""
    results = {}

    # Load legacy v2 scores if they exist
    legacy_file = CACHE_DIR / 'archive_vlm_scores_v2.json'
    if legacy_file.exists():
        print(f"Loading legacy scores from {legacy_file}...")
        with open(legacy_file) as f:
            legacy = json.load(f)
            results.update(legacy)
        print(f"  Loaded {len(legacy)} batches from legacy file")

    # Load new file if it exists (takes precedence)
    if SCORES_FILE.exists():
        print(f"Loading new scores from {SCORES_FILE}...")
        with open(SCORES_FILE) as f:
            new_scores = json.load(f)
            results.update(new_scores)
        print(f"  Loaded {len(new_scores)} batches from new file")

    return results


def save_results(results: Dict[str, Any]):
    with open(SCORES_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Pre-compute VLM fitness scores')
    parser.add_argument('--workers', type=int, default=4,
                        help='Concurrent API requests (default: 4)')
    parser.add_argument('--batch', type=str, help='Process single batch only')
    args = parser.parse_args()

    print(f"Prompts: {TRAIN_PROMPTS}")
    print(f"Model: {MODEL_KEY}")
    print(f"Workers: {args.workers}")
    print()

    # Discover batches with trajectory images
    if args.batch:
        all_batches = [args.batch]
    else:
        all_batches = sorted(
            p.stem.replace('_trajectory_heatmap', '')
            for p in BOT_TRAJECTORY_DIR.glob('*_trajectory_heatmap.png')
        )

    print(f"Found {len(all_batches)} batches with trajectory heatmaps")

    # Load existing, build task list
    existing = load_existing()
    tasks = []
    for batch_id in all_batches:
        for prompt in TRAIN_PROMPTS:
            if batch_id in existing and prompt in existing[batch_id]:
                # Skip already computed
                entry = existing[batch_id][prompt]
                if isinstance(entry, dict) and entry.get('score', -1) >= 0:
                    desc = entry.get('desc', '')
                    if not str(desc).startswith('Error:'):
                        continue
            tasks.append((batch_id, prompt))

    print(f"Tasks: {len(tasks)} (skipping {len(all_batches) * len(TRAIN_PROMPTS) - len(tasks)} already done)")

    if not tasks:
        print("Nothing to do!")
        return

    # Process with thread pool
    results = existing.copy()
    completed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_task, t): t for t in tasks}

        for future in as_completed(futures):
            batch_id, prompt = futures[future]
            result = future.result()
            completed += 1

            if result is None:
                print(f"  [{completed}/{len(tasks)}] {batch_id}|{prompt} — no image, skipped")
                continue

            score = result['score']
            desc = result['desc']

            # Init batch entry if needed
            if batch_id not in results:
                results[batch_id] = {'duration_ms': get_duration_ms(batch_id)}

            results[batch_id][prompt] = {'score': score, 'desc': desc}

            status = "OK" if not str(desc).startswith('Error:') else "ERR"
            print(f"  [{completed}/{len(tasks)}] {batch_id}|{prompt} — {status} score={score:.2f}")

            # Incremental save every 10 tasks
            if completed % 10 == 0:
                save_results(results)

    # Final save
    save_results(results)
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.0f}s — {completed} tasks, saved to {SCORES_FILE}")

    # Summary
    print("\nScore summary:")
    for prompt in TRAIN_PROMPTS:
        scores = [
            v[prompt]['score'] for v in results.values()
            if isinstance(v.get(prompt), dict) and isinstance(v[prompt].get('score'), (int, float))
            and v[prompt]['score'] >= 0
        ]
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  {prompt}: n={len(scores)}, mean={avg:.3f}, "
                  f"min={min(scores):.2f}, max={max(scores):.2f}")


if __name__ == '__main__':
    main()
