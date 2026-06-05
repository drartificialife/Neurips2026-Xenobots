#!/usr/bin/env python3
"""
Retry VLM scores for batches with errors in the archive.

Finds all entries with "Error:" in vlm_description and retries them.
Updates archive_vlm_scores.json with valid scores.

Usage:
    python scripts/retry_archive_errors.py --max-retries 5 --delay 3
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent))
from vlm_interpret_trajectory import interpret_trajectory, PROMPTS, SCENE_CONTEXT

ARCHIVE_VLM_SCORES = Path('cache/archive_vlm_scores.json')
ARCHIVE_ROOT = Path('D:\\xenobot_videos')
BASE_PROMPTS = ["slow down", "stop moving", "go fast", "move faster"]


def get_vlm_prompt(base_prompt: str) -> str:
    """Get VLM prompt template for base prompt."""
    if base_prompt == "slow down":
        return (
            f"{SCENE_CONTEXT} "
            f"Did the electrical intervention cause the bot to slow down? "
            "Compare the total distance traveled (in pixels) before and after: shorter distance after means the bot slowed down (if timesteps are equal). "
            "Summarize the behavioral change in 1-2 sentences. "
            "Then, rate the effectiveness of the intervention in slowing the bot, on a scale of 0-1 (1 = maximum slowing, 0 = no change). "
            "Format: <summary>\nScore: <number>"
        )
    elif base_prompt == "stop moving":
        return (
            f"{SCENE_CONTEXT} "
            f"Did the electrical intervention cause the bot to stop moving? "
            "If the post-intervention trajectory is very short or nearly flat, the bot stopped. "
            "Summarize the behavioral change in 1-2 sentences. "
            "Then, rate the effectiveness of the intervention in stopping the bot, on a scale of 0-1 (1 = completely stopped, 0 = no change). "
            "Format: <summary>\nScore: <number>"
        )
    elif base_prompt == "go fast":
        return (
            f"{SCENE_CONTEXT} "
            f"Did the electrical intervention cause the bot to go fast? "
            "Compare the total distance traveled (in pixels) before and after: longer distance after means the bot went faster (if timesteps are equal). "
            "Summarize the behavioral change in 1-2 sentences. "
            "Then, rate the effectiveness of the intervention in making the bot go fast, on a scale of 0-1 (1 = maximum increase in speed, 0 = no change). "
            "Format: <summary>\nScore: <number>"
        )
    elif base_prompt == "move faster":
        return (
            f"{SCENE_CONTEXT} "
            f"Did the electrical intervention cause the bot to move faster? "
            "Compare the total distance traveled (in pixels) before and after: longer distance after means the bot moved faster (if timesteps are equal). "
            "Summarize the behavioral change in 1-2 sentences. "
            "Then, rate the effectiveness of the intervention in increasing the bot's speed, on a scale of 0-1 (1 = maximum increase, 0 = no change). "
            "Format: <summary>\nScore: <number>"
        )


def score_batch(batch_id: str, base_prompt: str, max_retries: int = 3) -> Tuple[float, str]:
    """
    Call VLM to score a batch for a base prompt.

    Returns:
        (score, description)
    """
    trajectory_image = ARCHIVE_ROOT / batch_id / 'trajectory_macro50_prepost.png'

    if not trajectory_image.exists():
        return -1.0, f"Image not found: {trajectory_image}"

    vlm_prompt = get_vlm_prompt(base_prompt)

    for attempt in range(max_retries):
        try:
            result = interpret_trajectory(str(trajectory_image), vlm_prompt)
            score = result.get('score', -1.0)
            desc = result.get('description', '')

            # Check if got valid response (not another error)
            if 'Error' not in desc and score >= 0:
                return score, desc

            # If got error response, retry
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"      [Retry {attempt+1}/{max_retries}] Invalid response, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                return score, desc

        except Exception as e:
            error_str = str(e)
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"      [Retry {attempt+1}/{max_retries}] Exception, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                return -1.0, f"Exception: {error_str[:100]}"

    return -1.0, "All retries exhausted"


def retry_archive_errors(max_retries: int = 3, delay: int = 2):
    """Retry all batches with errors in the archive."""

    print(f"\n{'='*80}")
    print("RETRYING ARCHIVE VLM SCORE ERRORS")
    print(f"{'='*80}\n")

    # Load archive
    print("Loading archive...")
    with open(ARCHIVE_VLM_SCORES) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} batches\n")

    # Find error entries
    error_entries = []
    for batch_id, batch_info in data.items():
        for base_prompt in BASE_PROMPTS:
            if base_prompt in batch_info:
                score_info = batch_info[base_prompt]

                if isinstance(score_info, dict):
                    if 'Error' in str(score_info.get('desc', '')):
                        error_entries.append((batch_id, base_prompt))

    print(f"Found {len(error_entries)} error entries to retry\n")

    # Retry each error
    retried = 0
    fixed = 0

    for idx, (batch_id, base_prompt) in enumerate(error_entries):
        print(f"[{idx+1}/{len(error_entries)}] {batch_id} / {base_prompt}")

        score, desc = score_batch(batch_id, base_prompt, max_retries)

        if score >= 0 and 'Error' not in desc:
            print(f"  ✓ Fixed: Score {score:.2f}")
            data[batch_id][base_prompt] = {
                'score': score,
                'desc': desc
            }
            fixed += 1

            # Save immediately after each fix
            with open(ARCHIVE_VLM_SCORES, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"  Saved to archive")
        else:
            print(f"  ✗ Still failed: {desc[:60]}")

        retried += 1

        if retried < len(error_entries):
            print(f"  Waiting {delay}s before next...")
            time.sleep(delay)

    # Print final results
    print(f"\n{'='*80}")
    print(f"RESULTS: Fixed {fixed}/{len(error_entries)} errors")
    print(f"{'='*80}\n")
    print(f"Archive updated: {ARCHIVE_VLM_SCORES}\n")


def main():
    parser = argparse.ArgumentParser(description='Retry failed VLM scores in archive')
    parser.add_argument('--max-retries', type=int, default=3, help='Max retries per entry')
    parser.add_argument('--delay', type=int, default=2, help='Delay between entries (seconds)')

    args = parser.parse_args()

    retry_archive_errors(max_retries=args.max_retries, delay=args.delay)


if __name__ == '__main__':
    main()
