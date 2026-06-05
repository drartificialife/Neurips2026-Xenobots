#!/usr/bin/env python3
"""
Test P2I generalization on unseen synonym prompts.

After evolving P2I networks on 4 base prompts, test if they generalize
to unseen synonyms without additional training.

Pipeline:
  1. Test prompt (e.g., "reduce speed") → unseen synonym
  2. Get base prompt ("slow down")
  3. Use P2I_slow_down to predict intervention duration
  4. Lookup nearest batch in archive
  5. Call VLM to score this batch FOR THE TEST PROMPT
  6. Collect VLM scores and analyze statistically

Usage:
    python scripts/test_p2i_generalization.py
    python scripts/test_p2i_generalization.py --no-vlm
"""

import argparse
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import copy

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("Installing PyTorch...")
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'torch'])
    import torch
    import torch.nn as nn

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing sentence-transformers...")
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'sentence-transformers'])
    from sentence_transformers import SentenceTransformer

try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    print("Installing scikit-learn...")
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'scikit-learn'])
    from sklearn.neighbors import NearestNeighbors

# Import VLM scoring function
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Lazy import to avoid circular dependency issues
def get_interpret_trajectory():
    from vlm_interpret_trajectory import interpret_trajectory
    return interpret_trajectory

# Paths
CACHE_DIR = Path('cache')
MODELS_DIR = Path('models')
RESULTS_DIR = Path('results')
ARCHIVE_VLM_SCORES = CACHE_DIR / 'archive_vlm_scores.json'
NN_INDEX_FILE = CACHE_DIR / 'nn_index.pkl'
TEST_PROMPTS_FILE = CACHE_DIR / 'test_prompts.json'

# P2I network (same as in evolve script)
class P2INetwork(nn.Module):
    def __init__(self, embedding_dim: int = 384, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(embedding))


def load_vlm_scores() -> Dict[str, Dict]:
    """Load pre-computed VLM scores."""
    if not ARCHIVE_VLM_SCORES.exists():
        raise FileNotFoundError(f"VLM scores not found: {ARCHIVE_VLM_SCORES}")
    with open(ARCHIVE_VLM_SCORES) as f:
        return json.load(f)


def load_test_prompts() -> List[Dict]:
    """Load test prompts (unseen synonyms)."""
    if not TEST_PROMPTS_FILE.exists():
        raise FileNotFoundError(f"Test prompts not found: {TEST_PROMPTS_FILE}")
    with open(TEST_PROMPTS_FILE) as f:
        data = json.load(f)
        # Convert dict format to list of dicts
        test_prompts = []
        for base_prompt, prompts in data.items():
            for prompt in prompts:
                test_prompts.append({
                    'prompt': prompt,
                    'base_prompt': base_prompt
                })
        return test_prompts


def load_nn_index(vlm_scores: Dict[str, Dict]):
    """Load or build NN index."""
    if NN_INDEX_FILE.exists():
        with open(NN_INDEX_FILE, 'rb') as f:
            return pickle.load(f)

    durations = np.array([batch_info['duration_ms'] for batch_info in vlm_scores.values()]).reshape(-1, 1)
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(durations)

    NN_INDEX_FILE.parent.mkdir(exist_ok=True, parents=True)
    with open(NN_INDEX_FILE, 'wb') as f:
        pickle.dump(nn, f)

    return nn


def get_duration_range(vlm_scores: Dict[str, Dict]) -> Tuple[float, float]:
    """Get min/max duration from archive."""
    durations = [batch_info['duration_ms'] for batch_info in vlm_scores.values()]
    return min(durations), max(durations)


def get_vlm_prompt_for_test(test_prompt: str, base_prompt: str) -> str:
    """
    Generate VLM query for test prompt using base prompt's template.

    Example:
      test_prompt: "reduce speed"
      base_prompt: "slow down"

    Returns: "Did the intervention cause bot to reduce speed?..."
    """
    from vlm_interpret_trajectory import PROMPTS, SCENE_CONTEXT

    if base_prompt not in PROMPTS:
        return test_prompt  # Fallback to raw prompt

    base_template = PROMPTS[base_prompt]

    # Replace {command_prompt} placeholder if it exists, otherwise extract pattern
    # Base templates: "Did the intervention cause bot to [BASE]?"
    # We replace [BASE] with test_prompt

    # Extract the structure but replace the specific behavior word
    if base_prompt == "slow down":
        return (
            f"{SCENE_CONTEXT} "
            f"Did the electrical intervention cause the bot to {test_prompt}? "
            "Compare the total distance traveled (in pixels) before and after: shorter distance after means the bot slowed down (if timesteps are equal). "
            "Summarize the behavioral change in 1-2 sentences. "
            "Then, rate the effectiveness of the intervention in slowing the bot, on a scale of 0-1 (1 = maximum slowing, 0 = no change). "
            "Format: <summary>\nScore: <number>"
        )
    elif base_prompt == "stop moving":
        return (
            f"{SCENE_CONTEXT} "
            f"Did the electrical intervention cause the bot to {test_prompt}? "
            "If the post-intervention trajectory is very short or nearly flat, the bot stopped. "
            "Summarize the behavioral change in 1-2 sentences. "
            "Then, rate the effectiveness of the intervention in stopping the bot, on a scale of 0-1 (1 = completely stopped, 0 = no change). "
            "Format: <summary>\nScore: <number>"
        )
    elif base_prompt == "go fast":
        return (
            f"{SCENE_CONTEXT} "
            f"Did the electrical intervention cause the bot to {test_prompt}? "
            "Compare the total distance traveled (in pixels) before and after: longer distance after means the bot went faster (if timesteps are equal). "
            "Summarize the behavioral change in 1-2 sentences. "
            "Then, rate the effectiveness of the intervention in making the bot go fast, on a scale of 0-1 (1 = maximum increase in speed, 0 = no change). "
            "Format: <summary>\nScore: <number>"
        )
    elif base_prompt == "move faster":
        return (
            f"{SCENE_CONTEXT} "
            f"Did the electrical intervention cause the bot to {test_prompt}? "
            "Compare the total distance traveled (in pixels) before and after: longer distance after means the bot moved faster (if timesteps are equal). "
            "Summarize the behavioral change in 1-2 sentences. "
            "Then, rate the effectiveness of the intervention in increasing the bot's speed, on a scale of 0-1 (1 = maximum increase, 0 = no change). "
            "Format: <summary>\nScore: <number>"
        )

    return test_prompt  # Fallback


def score_batch_for_prompt(batch_id: str, test_prompt: str, base_prompt: str, max_retries: int = 3) -> Tuple[float, str]:
    """
    Call VLM to score a batch trajectory for a test prompt.
    Uses base prompt's template but with test prompt inserted.
    Retries up to max_retries times on failure (for API errors).

    Returns: (score, description)
    """
    import time

    archive_root = Path('D:\\xenobot_videos')
    trajectory_image = archive_root / batch_id / 'trajectory_macro50_prepost.png'

    if not trajectory_image.exists():
        return 0.0, f"Image not found: {trajectory_image}"

    interpret_trajectory = get_interpret_trajectory()
    vlm_prompt = get_vlm_prompt_for_test(test_prompt, base_prompt)

    for attempt in range(max_retries):
        try:
            result = interpret_trajectory(str(trajectory_image), vlm_prompt)
            score = result.get('score', 0.0)
            desc = result.get('description', '')

            # Check if score is valid
            if score is None or score < 0:
                # -1 indicates parsing error, but we got a response
                return score if score >= 0 else 0.0, desc

            return score, desc

        except Exception as e:
            error_str = str(e)

            # Retry on server errors (429, 524)
            if ('429' in error_str or '524' in error_str) and attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)  # 5s, 10s, 15s
                print(f"     [Retry {attempt+1}/{max_retries}] Server error, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            # If all retries exhausted, return error
            return 0.0, f"VLM Error: {error_str[:100]}"

    return 0.0, "All retries exhausted"


def find_nearest_batch(
    predicted_duration_norm: float,
    vlm_scores: Dict[str, Dict],
    nn_index,
    duration_min: float,
    duration_max: float
) -> str:
    """Find nearest batch to predicted duration."""
    actual_duration = predicted_duration_norm * (duration_max - duration_min) + duration_min
    actual_duration_arr = np.array([actual_duration]).reshape(1, -1)
    _, indices = nn_index.kneighbors(actual_duration_arr)

    batch_ids = list(vlm_scores.keys())
    batch_idx = min(indices[0][0], len(batch_ids) - 1)
    return batch_ids[batch_idx]


def load_evolved_p2i(base_prompt: str, device: str = 'cpu') -> P2INetwork:
    """Load evolved P2I network for a base prompt."""
    model_path = MODELS_DIR / f'p2i_evolved_{base_prompt.replace(" ", "_")}.pt'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    network = P2INetwork().to(device)
    network.load_state_dict(checkpoint['model_state_dict'])
    return network


def predict_duration(
    network: P2INetwork,
    prompt: str,
    embedder: SentenceTransformer,
    duration_min: float,
    duration_max: float,
    device: str = 'cpu'
) -> float:
    """Predict duration for prompt using evolved P2I."""
    network.eval()
    with torch.no_grad():
        embedding = embedder.encode(prompt, convert_to_tensor=False)
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
        pred_norm = network(embedding_tensor).squeeze().item()
        pred_duration = pred_norm * (duration_max - duration_min) + duration_min
        return pred_duration


def test_generalization(device: str = 'cpu', call_vlm: bool = True):
    """Test P2I generalization on unseen prompts."""

    print(f"\n{'='*80}")
    print("P2I Generalization Testing on Unseen Prompts")
    print(f"{'='*80}\n")

    # Load data
    print("Loading data...")
    vlm_scores = load_vlm_scores()
    test_prompts_data = load_test_prompts()
    nn_index = load_nn_index(vlm_scores)
    duration_min, duration_max = get_duration_range(vlm_scores)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    print(f"  Loaded {len(test_prompts_data)} test prompts (unseen synonyms)\n")

    # Load evolved P2I networks (one for each base prompt)
    print("Loading evolved P2I networks...")
    p2i_networks = {}
    base_prompts = ["slow down", "stop moving", "go fast", "move faster"]

    for base_prompt in base_prompts:
        try:
            p2i_networks[base_prompt] = load_evolved_p2i(base_prompt, device)
            print(f"  ✓ {base_prompt}")
        except FileNotFoundError as e:
            print(f"  ✗ {base_prompt}: {e}")

    if len(p2i_networks) == 0:
        print("No evolved P2I networks found!")
        return

    print(f"Loaded {len(p2i_networks)} P2I networks\n")

    # Test generalization
    print(f"{'='*80}")
    print("Testing Generalization on Unseen Prompts")
    print(f"{'='*80}\n")

    results = []
    output_path = RESULTS_DIR / 'p2i_generalization_test.json'

    for i, test_data in enumerate(test_prompts_data):
        test_prompt = test_data['prompt']
        base_prompt = test_data['base_prompt']

        if base_prompt not in p2i_networks:
            print(f"[SKIP] No P2I network for base prompt: {base_prompt}")
            continue

        print(f"[{i+1}/{len(test_prompts_data)}] Testing '{test_prompt}' (base: '{base_prompt}')")

        # Predict duration using P2I network trained on base_prompt
        pred_duration = predict_duration(
            p2i_networks[base_prompt], test_prompt, embedder,
            duration_min, duration_max, device
        )

        # Find nearest batch
        nearest_batch = find_nearest_batch(
            pred_duration / (duration_max - duration_min),  # normalize
            vlm_scores, nn_index,
            duration_min, duration_max
        )

        # Call VLM to score this batch FOR THE TEST PROMPT
        # Uses base_prompt's template but with test_prompt inserted
        if call_vlm:
            vlm_score, vlm_desc = score_batch_for_prompt(nearest_batch, test_prompt, base_prompt)
            print(f"  -> Score: {vlm_score:.2f}, Duration: {pred_duration:.0f} ms, Batch: {nearest_batch}")
            if vlm_score == 0.0:
                print(f"     WARNING: Zero score - {vlm_desc[:80]}")
        else:
            vlm_score = 0.0
            vlm_desc = "VLM call disabled"

        results.append({
            'prompt': test_prompt,
            'base_prompt': base_prompt,
            'predicted_duration': pred_duration,
            'nearest_batch': nearest_batch,
            'vlm_score': vlm_score,
            'vlm_description': vlm_desc,
        })

        # Write incremental results to JSON
        vlm_scores_list = [r['vlm_score'] for r in results]
        with open(output_path, 'w') as f:
            json.dump({
                'total_tests': len(results),
                'vlm_score_mean': float(np.mean(vlm_scores_list)),
                'vlm_score_median': float(np.median(vlm_scores_list)),
                'vlm_score_std': float(np.std(vlm_scores_list)),
                'vlm_score_cv': float(np.std(vlm_scores_list) / np.mean(vlm_scores_list)) if np.mean(vlm_scores_list) > 0 else float('inf'),
                'min_score': float(np.min(vlm_scores_list)),
                'max_score': float(np.max(vlm_scores_list)),
                'results': results
            }, f, indent=2)

    # Sort by base prompt then by VLM score
    results.sort(key=lambda x: (x['base_prompt'], -x['vlm_score']))

    # Print results
    print(f"\n{'='*100}")
    print(f"{'Prompt':<25} {'Base':<15} {'Predicted':<12} {'VLM Score':<12} {'Batch':<12}")
    print(f"{'='*100}")

    for r in results:
        prompt_short = r['prompt'][:25]
        base_short = r['base_prompt'][:15]
        print(f"{prompt_short:<25} {base_short:<15} {r['predicted_duration']:>11.0f} {r['vlm_score']:>11.2f}  {r['nearest_batch']:<12}")

    # Statistical Analysis
    print(f"\n{'='*80}")
    print("Statistical Analysis")
    print(f"{'='*80}\n")

    total = len(results)
    vlm_scores_list = [r['vlm_score'] for r in results]

    mean_score = np.mean(vlm_scores_list)
    median_score = np.median(vlm_scores_list)
    std_score = np.std(vlm_scores_list)
    min_score = np.min(vlm_scores_list)
    max_score = np.max(vlm_scores_list)
    cv_score = std_score / mean_score if mean_score > 0 else float('inf')

    print(f"Total test prompts: {total}")
    print(f"\nVLM Score Distribution:")
    print(f"  Mean:     {mean_score:.4f}")
    print(f"  Median:   {median_score:.4f}")
    print(f"  Std Dev:  {std_score:.4f}")
    print(f"  Min:      {min_score:.4f}")
    print(f"  Max:      {max_score:.4f}")
    print(f"  CV (σ/μ): {cv_score:.4f}")

    # Interpretation
    print(f"\nInterpretation:")
    if mean_score > 0.80 and std_score < 0.10:
        print(f"  ✅ Excellent generalization (high mean, low variance)")
    elif mean_score > 0.70:
        print(f"  ✓ Good generalization (mean > 0.70)")
    elif mean_score > 0.55:
        print(f"  ⚠️ Partial generalization (mean > 0.55)")
    else:
        print(f"  ❌ Poor generalization (mean < 0.55)")

    # Per-base-prompt breakdown
    print(f"\n{'='*80}")
    print("Per-Base-Prompt Breakdown")
    print(f"{'='*80}\n")

    for base_prompt in base_prompts:
        base_results = [r for r in results if r['base_prompt'] == base_prompt]
        if not base_results:
            continue

        base_scores = [r['vlm_score'] for r in base_results]
        base_mean = np.mean(base_scores)
        base_median = np.median(base_scores)
        base_std = np.std(base_scores)
        base_cv = base_std / base_mean if base_mean > 0 else float('inf')

        print(f"  {base_prompt}")
        print(f"    Count:     {len(base_results)}")
        print(f"    Mean:      {base_mean:.4f}")
        print(f"    Median:    {base_median:.4f}")
        print(f"    Std Dev:   {base_std:.4f}")
        print(f"    CV:        {base_cv:.4f}")
        for r in base_results:
            print(f"      - {r['prompt']:<25} {r['vlm_score']:.3f}")
        print()

    print(f"\nResults saved to: {output_path}\n")


def main():
    parser = argparse.ArgumentParser(description='Test P2I Generalization on Unseen Prompts')
    parser.add_argument('--no-vlm', action='store_true', help='Skip VLM calls (for testing)')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    test_generalization(device=args.device, call_vlm=not args.no_vlm)


if __name__ == '__main__':
    main()
