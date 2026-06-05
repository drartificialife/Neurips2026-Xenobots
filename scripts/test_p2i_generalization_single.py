#!/usr/bin/env python3
"""
Test single P2I prompt (useful for retrying failed tests).

Usage:
    python scripts/test_p2i_generalization_single.py --prompt "accelerate" --base "go fast"
"""

import argparse
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))

def get_interpret_trajectory():
    from vlm_interpret_trajectory import interpret_trajectory
    return interpret_trajectory

# Paths
CACHE_DIR = Path('cache')
MODELS_DIR = Path('models')
RESULTS_DIR = Path('results')
ARCHIVE_VLM_SCORES = CACHE_DIR / 'archive_vlm_scores.json'
NN_INDEX_FILE = CACHE_DIR / 'nn_index.pkl'

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
    with open(ARCHIVE_VLM_SCORES) as f:
        return json.load(f)


def load_nn_index(vlm_scores: Dict[str, Dict]):
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
    durations = [batch_info['duration_ms'] for batch_info in vlm_scores.values()]
    return min(durations), max(durations)


def get_vlm_prompt_for_test(test_prompt: str, base_prompt: str) -> str:
    from vlm_interpret_trajectory import SCENE_CONTEXT

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
    return test_prompt


def score_batch_for_prompt(batch_id: str, test_prompt: str, base_prompt: str, max_retries: int = 3) -> Tuple[float, str]:
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
            if score is None or score < 0:
                return score if score >= 0 else 0.0, desc
            return score, desc
        except Exception as e:
            error_str = str(e)
            if ('429' in error_str or '524' in error_str) and attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"     [Retry {attempt+1}/{max_retries}] Server error, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            return 0.0, f"VLM Error: {error_str[:100]}"

    return 0.0, "All retries exhausted"


def find_nearest_batch(predicted_duration_norm: float, vlm_scores: Dict[str, Dict], nn_index, duration_min: float, duration_max: float) -> str:
    actual_duration = predicted_duration_norm * (duration_max - duration_min) + duration_min
    actual_duration_arr = np.array([actual_duration]).reshape(1, -1)
    _, indices = nn_index.kneighbors(actual_duration_arr)
    batch_ids = list(vlm_scores.keys())
    batch_idx = min(indices[0][0], len(batch_ids) - 1)
    return batch_ids[batch_idx]


def load_evolved_p2i(base_prompt: str, device: str = 'cpu') -> P2INetwork:
    model_path = MODELS_DIR / f'p2i_evolved_{base_prompt.replace(" ", "_")}.pt'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    network = P2INetwork().to(device)
    network.load_state_dict(checkpoint['model_state_dict'])
    return network


def predict_duration(network: P2INetwork, prompt: str, embedder: SentenceTransformer, duration_min: float, duration_max: float, device: str = 'cpu') -> float:
    network.eval()
    with torch.no_grad():
        embedding = embedder.encode(prompt, convert_to_tensor=False)
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
        pred_norm = network(embedding_tensor).squeeze().item()
        pred_duration = pred_norm * (duration_max - duration_min) + duration_min
        return pred_duration


def test_single_prompt(test_prompt: str, base_prompt: str, device: str = 'cpu'):
    """Test a single prompt and append result to existing results file."""

    print(f"\n{'='*80}")
    print(f"Testing: '{test_prompt}' (base: '{base_prompt}')")
    print(f"{'='*80}\n")

    # Load data
    vlm_scores = load_vlm_scores()
    nn_index = load_nn_index(vlm_scores)
    duration_min, duration_max = get_duration_range(vlm_scores)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Load P2I network
    print(f"Loading P2I network for '{base_prompt}'...")
    p2i = load_evolved_p2i(base_prompt, device)

    # Predict duration
    print(f"Predicting duration for '{test_prompt}'...")
    pred_duration = predict_duration(p2i, test_prompt, embedder, duration_min, duration_max, device)

    # Find nearest batch
    nearest_batch = find_nearest_batch(
        pred_duration / (duration_max - duration_min),
        vlm_scores, nn_index,
        duration_min, duration_max
    )
    print(f"Predicted duration: {pred_duration:.0f} ms → {nearest_batch}")

    # Call VLM
    print(f"Calling VLM...")
    vlm_score, vlm_desc = score_batch_for_prompt(nearest_batch, test_prompt, base_prompt)
    print(f"VLM Score: {vlm_score:.2f}")

    # Prepare result
    result = {
        'prompt': test_prompt,
        'base_prompt': base_prompt,
        'predicted_duration': pred_duration,
        'nearest_batch': nearest_batch,
        'vlm_score': vlm_score,
        'vlm_description': vlm_desc,
    }

    # Load existing results and append
    output_path = RESULTS_DIR / 'p2i_generalization_test.json'
    if output_path.exists():
        with open(output_path) as f:
            data = json.load(f)
        results = data.get('results', [])

        # Remove duplicate if exists
        results = [r for r in results if r['prompt'] != test_prompt]
        results.append(result)
        results.sort(key=lambda x: (x['base_prompt'], -x['vlm_score']))
    else:
        results = [result]

    # Recalculate stats
    vlm_scores_list = [r['vlm_score'] for r in results]
    output_data = {
        'total_tests': len(results),
        'vlm_score_mean': float(np.mean(vlm_scores_list)),
        'vlm_score_median': float(np.median(vlm_scores_list)),
        'vlm_score_std': float(np.std(vlm_scores_list)),
        'vlm_score_cv': float(np.std(vlm_scores_list) / np.mean(vlm_scores_list) if np.mean(vlm_scores_list) > 0 else float('inf')),
        'min_score': float(np.min(vlm_scores_list)),
        'max_score': float(np.max(vlm_scores_list)),
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Result saved!")
    print(f"Updated results file: {output_path}")
    print(f"Total results now: {len(results)}")


def main():
    parser = argparse.ArgumentParser(description='Test single P2I prompt')
    parser.add_argument('--prompt', type=str, required=True, help='Test prompt to evaluate')
    parser.add_argument('--base', type=str, required=True, help='Base prompt')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    test_single_prompt(args.prompt, args.base, device=args.device)


if __name__ == '__main__':
    main()
