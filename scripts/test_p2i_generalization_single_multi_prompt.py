#!/usr/bin/env python3
"""
Test single prompt with Multi-Prompt P2I checkpoint.

Useful for retrying failed tests with the multi-prompt model.

Usage:
    python scripts/test_p2i_generalization_single_multi_prompt.py --prompt "slow down" --base "slow down"
"""

import argparse
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import torch
import torch.nn as nn
import time

try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'scikit-learn'])
    from sklearn.neighbors import NearestNeighbors

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'sentence-transformers'])
    from sentence_transformers import SentenceTransformer

# Paths
CACHE_DIR = Path('cache')
MODELS_DIR = Path('models')
RESULTS_DIR = Path('results')
ARCHIVE_VLM_SCORES = CACHE_DIR / 'archive_vlm_scores.json'
NN_INDEX_FILE = CACHE_DIR / 'nn_index.pkl'
MODEL_FILE = MODELS_DIR / 'p2i_evolved_multi_prompt.pt'

BASE_PROMPTS = ["slow down", "stop moving", "go fast", "move faster"]


class MultiPromptP2INetwork(nn.Module):
    """P2I network with 4 outputs."""

    def __init__(self, embedding_dim: int = 384, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
        )

        self.heads = nn.ModuleDict({
            "slow_down": nn.Linear(32, 1),
            "stop_moving": nn.Linear(32, 1),
            "go_fast": nn.Linear(32, 1),
            "move_faster": nn.Linear(32, 1),
        })

    def forward(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_out = self.shared(embedding)
        outputs = {}
        key_mapping = {
            "slow_down": "slow down",
            "stop_moving": "stop moving",
            "go_fast": "go fast",
            "move_faster": "move faster",
        }
        for head_key, head in self.heads.items():
            base_prompt = key_mapping[head_key]
            outputs[base_prompt] = torch.sigmoid(head(shared_out))
        return outputs


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


def load_model(device: str = 'cpu') -> MultiPromptP2INetwork:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Multi-prompt model not found: {MODEL_FILE}")

    network = MultiPromptP2INetwork().to(device)
    checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=False)
    network.load_state_dict(checkpoint['model_state_dict'])
    return network


def score_batch_for_prompt(batch_id: str, test_prompt: str, base_prompt: str, max_retries: int = 3) -> Tuple[float, str]:
    """Call VLM to score a batch."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from vlm_interpret_trajectory import interpret_trajectory
    from test_p2i_generalization_single import get_vlm_prompt_for_test

    archive_root = Path('D:\\xenobot_videos')
    trajectory_image = archive_root / batch_id / 'trajectory_macro50_prepost.png'

    if not trajectory_image.exists():
        return 0.0, f"Image not found: {trajectory_image}"

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


def find_nearest_batch(
    predicted_duration_norm: float,
    vlm_scores: Dict[str, Dict],
    nn_index,
    duration_min: float,
    duration_max: float
) -> str:
    actual_duration = predicted_duration_norm * (duration_max - duration_min) + duration_min
    actual_duration_arr = np.array([actual_duration]).reshape(1, -1)
    _, indices = nn_index.kneighbors(actual_duration_arr)
    batch_ids = list(vlm_scores.keys())
    batch_idx = min(indices[0][0], len(batch_ids) - 1)
    return batch_ids[batch_idx]


def predict_duration(
    network: MultiPromptP2INetwork,
    prompt: str,
    base_prompt: str,
    embedder: SentenceTransformer,
    duration_min: float,
    duration_max: float,
    device: str = 'cpu'
) -> float:
    network.eval()
    with torch.no_grad():
        embedding = embedder.encode(prompt, convert_to_tensor=False)
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
        outputs = network(embedding_tensor)
        pred_norm = outputs[base_prompt].squeeze().item()
        pred_duration = pred_norm * (duration_max - duration_min) + duration_min
        return pred_duration


def test_single_prompt(test_prompt: str, base_prompt: str, device: str = 'cpu'):
    """Test single prompt with multi-prompt model."""

    print(f"\n{'='*80}")
    print(f"Testing: '{test_prompt}' (base: '{base_prompt}')")
    print(f"Model: Multi-Prompt P2I")
    print(f"{'='*80}\n")

    # Load data
    print("Loading data...")
    vlm_scores = load_vlm_scores()
    nn_index = load_nn_index(vlm_scores)
    duration_min, duration_max = get_duration_range(vlm_scores)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Load model
    print("Loading multi-prompt P2I network...")
    network = load_model(device)
    print()

    # Predict duration
    print(f"Predicting duration for '{test_prompt}'...")
    pred_duration = predict_duration(
        network, test_prompt, base_prompt, embedder,
        duration_min, duration_max, device
    )

    # Find nearest batch
    nearest_batch = find_nearest_batch(
        pred_duration / (duration_max - duration_min),
        vlm_scores, nn_index,
        duration_min, duration_max
    )
    print(f"Predicted duration: {pred_duration:.0f} ms → {nearest_batch}\n")

    # Call VLM
    print(f"Calling VLM...")
    vlm_score, vlm_desc = score_batch_for_prompt(nearest_batch, test_prompt, base_prompt)
    print(f"VLM Score: {vlm_score:.2f}\n")

    # Prepare result
    result = {
        'prompt': test_prompt,
        'base_prompt': base_prompt,
        'predicted_duration': pred_duration,
        'nearest_batch': nearest_batch,
        'vlm_score': vlm_score,
        'vlm_description': vlm_desc,
    }

    # Load existing multi-prompt results and update
    output_path = RESULTS_DIR / 'p2i_generalization_test_multi_prompt.json'
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

    print(f"Results updated in: {output_path}\n")


def main():
    parser = argparse.ArgumentParser(description='Test single prompt with multi-prompt P2I model')
    parser.add_argument('--prompt', type=str, required=True, help='Test prompt')
    parser.add_argument('--base', type=str, required=True, help='Base prompt')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    test_single_prompt(args.prompt, args.base, args.device)


if __name__ == '__main__':
    main()
