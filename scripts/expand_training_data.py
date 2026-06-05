#!/usr/bin/env python3
"""
Expand training data for P2I policy by adding synonym prompts.

Strategy:
  - Keep 4 base prompts from evolution (with known optimal durations + fitness)
  - Add synonym/paraphrase prompts for each base prompt
  - Assign same fitness/duration as base prompt (no VLM calls needed!)
  - Result: larger training set → better generalization

Usage:
    python scripts/expand_training_data.py

    # Or import as module:
    from expand_training_data import get_expanded_training_data
    train_data = get_expanded_training_data()
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

RESULTS_DIR = Path('results')

# Base prompts from evolution (known fitness + duration)
BASE_PROMPTS = {
    "slow down": {
        "duration_ms": 244470,
        "fitness": 1.0,
        "synonyms": [
            "reduce speed",
            "move slower",
            "decrease velocity",
            "slow movement",
            "decelerate",
            "lower speed",
            "slow the motion",
            "make it slower",
        ]
    },
    "stop moving": {
        "duration_ms": 249821,
        "fitness": 0.9,
        "synonyms": [
            "halt",
            "freeze",
            "stop",
            "no movement",
            "cease motion",
            "come to stop",
            "immobilize",
            "pause motion",
            "stationary",
        ]
    },
    "go fast": {
        "duration_ms": 87352,
        "fitness": 1.0,
        "synonyms": [
            "increase speed",
            "accelerate",
            "move quickly",
            "high speed",
            "speed up fast",
            "faster motion",
            "quick movement",
            "rapid motion",
        ]
    },
    "move faster": {
        "duration_ms": 51879,
        "fitness": 1.0,
        "synonyms": [
            "speed up",
            "faster movement",
            "increase velocity",
            "quicken pace",
            "hasten",
            "accelerate movement",
            "move quicker",
            "boost speed",
        ]
    }
}


def get_expanded_training_data() -> Dict[str, Dict]:
    """
    Generate expanded training data with base prompts + synonyms.

    Returns:
        Dict mapping prompt → {duration_ms, fitness, base_prompt}
        Example:
        {
            "slow down": {"duration_ms": 244470, "fitness": 1.0, "base_prompt": "slow down"},
            "reduce speed": {"duration_ms": 244470, "fitness": 1.0, "base_prompt": "slow down"},
            "move slower": {"duration_ms": 244470, "fitness": 1.0, "base_prompt": "slow down"},
            ...
        }
    """
    expanded_data = {}

    for base_prompt, base_info in BASE_PROMPTS.items():
        # Add base prompt
        expanded_data[base_prompt] = {
            "duration_ms": base_info["duration_ms"],
            "fitness": base_info["fitness"],
            "base_prompt": base_prompt,
            "is_base": True,
        }

        # Add synonyms with same duration + fitness
        for synonym in base_info["synonyms"]:
            expanded_data[synonym] = {
                "duration_ms": base_info["duration_ms"],
                "fitness": base_info["fitness"],
                "base_prompt": base_prompt,
                "is_base": False,
            }

    return expanded_data


def print_expanded_data_summary():
    """Print summary of expanded training data."""
    expanded_data = get_expanded_training_data()

    print("\n" + "="*70)
    print("EXPANDED TRAINING DATA SUMMARY")
    print("="*70)

    total_prompts = len(expanded_data)
    base_prompts = sum(1 for v in expanded_data.values() if v["is_base"])
    synonym_prompts = total_prompts - base_prompts

    print(f"\nTotal prompts: {total_prompts}")
    print(f"  - Base prompts: {base_prompts}")
    print(f"  - Synonym prompts: {synonym_prompts}")

    print(f"\nBreakdown by base prompt:")
    for base_prompt, base_info in BASE_PROMPTS.items():
        n_synonyms = len(base_info["synonyms"])
        total = 1 + n_synonyms  # base + synonyms
        print(f"\n  '{base_prompt}' (fitness={base_info['fitness']:.2f})")
        print(f"    Duration: {base_info['duration_ms']:.0f} ms")
        print(f"    Training samples: {total} (1 base + {n_synonyms} synonyms)")
        print(f"    Synonyms:")
        for syn in base_info["synonyms"][:3]:
            print(f"      - {syn}")
        if len(base_info["synonyms"]) > 3:
            print(f"      ... and {len(base_info['synonyms']) - 3} more")

    print(f"\n{'='*70}\n")


def save_expanded_data(output_path: Path = None):
    """Save expanded training data to JSON file."""
    if output_path is None:
        output_path = Path('cache') / 'expanded_training_data.json'

    output_path.parent.mkdir(exist_ok=True, parents=True)

    expanded_data = get_expanded_training_data()

    with open(output_path, 'w') as f:
        json.dump(expanded_data, f, indent=2)

    print(f"Expanded training data saved to: {output_path}")


def load_expanded_data(data_path: Path = None) -> Dict[str, Dict]:
    """Load expanded training data from JSON file."""
    if data_path is None:
        data_path = Path('cache') / 'expanded_training_data.json'

    if not data_path.exists():
        print(f"File not found: {data_path}")
        print("Generating from scratch...")
        return get_expanded_training_data()

    with open(data_path, 'r') as f:
        return json.load(f)


def get_train_test_split(test_size: float = 0.3, seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    Split expanded data into train/test sets.

    Strategy:
      - Train: All base prompts + some synonyms
      - Test: Remaining synonyms (unseen paraphrases)

    Args:
        test_size: Fraction of synonyms to use as test
        seed: Random seed

    Returns:
        (train_prompts, test_prompts)
    """
    import random
    random.seed(seed)

    expanded_data = get_expanded_training_data()

    train_prompts = []
    test_prompts = []

    # Group by base prompt
    prompts_by_base = {}
    for prompt, info in expanded_data.items():
        base = info["base_prompt"]
        if base not in prompts_by_base:
            prompts_by_base[base] = {"base": [], "synonyms": []}

        if info["is_base"]:
            prompts_by_base[base]["base"].append(prompt)
        else:
            prompts_by_base[base]["synonyms"].append(prompt)

    # Split: always include base, split synonyms
    for base_prompt, groups in prompts_by_base.items():
        # Add base prompts to training
        train_prompts.extend(groups["base"])

        # Split synonyms
        synonyms = groups["synonyms"]
        n_test = max(1, int(len(synonyms) * test_size))
        test_sample = random.sample(synonyms, n_test)
        train_sample = [s for s in synonyms if s not in test_sample]

        train_prompts.extend(train_sample)
        test_prompts.extend(test_sample)

    return train_prompts, test_prompts


def print_train_test_split():
    """Print train/test split summary."""
    train_prompts, test_prompts = get_train_test_split()

    print("\n" + "="*70)
    print("TRAIN/TEST SPLIT")
    print("="*70)

    expanded_data = get_expanded_training_data()

    print(f"\nTrain set: {len(train_prompts)} prompts")
    for prompt in train_prompts[:5]:
        info = expanded_data[prompt]
        is_base = " [BASE]" if info["is_base"] else ""
        print(f"  - '{prompt}' -> {info['duration_ms']:.0f} ms{is_base}")
    if len(train_prompts) > 5:
        print(f"  ... and {len(train_prompts) - 5} more")

    print(f"\nTest set (unseen): {len(test_prompts)} prompts")
    for prompt in test_prompts[:5]:
        info = expanded_data[prompt]
        base = info["base_prompt"]
        print(f"  - '{prompt}' -> {base} ({info['duration_ms']:.0f} ms)")
    if len(test_prompts) > 5:
        print(f"  ... and {len(test_prompts) - 5} more")

    print(f"\n{'='*70}\n")


def main():
    print("\nGenerating expanded training data for P2I policy...")

    # Print summary
    print_expanded_data_summary()

    # Save to cache
    save_expanded_data()

    # Print train/test split
    print_train_test_split()


if __name__ == '__main__':
    main()
