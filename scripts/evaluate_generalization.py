#!/usr/bin/env python3
"""
Evaluate P2I network generalization
Test on held-out prompts (GT1) and held-out batches (GT2)
"""

import torch
import numpy as np
from pathlib import Path
import json
import argparse

# Import project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.p2i_network import P2INetwork, PromptEncoder
from src.vlm_scorer import VLMScorer


def evaluate_generalization(
    checkpoint_path,
    eval_protocol="both",
    batch_size=32,
):
    """
    Evaluate trained model on generalization tests

    Args:
        checkpoint_path: Path to trained model checkpoint
        eval_protocol: "gt1" (novel prompts), "gt2" (novel prompts + batches), or "both"
        batch_size: Batch size for evaluation

    Returns:
        results: Dictionary with per-behavior and overall scores
    """

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    hyperparams = checkpoint.get('hyperparameters', {})
    num_behaviors = hyperparams.get('num_behaviors', 8)

    # Initialize network
    network = P2INetwork(num_behaviors=num_behaviors)
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()

    # Initialize components
    encoder = PromptEncoder()
    vlm_scorer = VLMScorer(use_ollama=True)

    print(f"\nLoaded model from: {checkpoint_path}")
    print(f"Hyperparameters: {hyperparams}")

    results = {
        'checkpoint': str(checkpoint_path),
        'eval_protocol': eval_protocol,
        'per_behavior': {},
        'overall': {}
    }

    # GT1: Novel prompts, training batches
    if eval_protocol in ["gt1", "both"]:
        print("\n" + "="*60)
        print("GT1: Generalization to Novel Prompts")
        print("="*60)

        # TODO: Implement GT1 evaluation
        # 1. Load test prompts
        # 2. For each test prompt: encode -> network -> nearest batch -> VLM score
        # 3. Compute per-behavior and overall scores

        gt1_results = {
            'mean_score': 0.0,
            'per_behavior': {}
        }
        results['gt1'] = gt1_results
        print("GT1 evaluation complete")

    # GT2: Novel prompts and batches
    if eval_protocol in ["gt2", "both"]:
        print("\n" + "="*60)
        print("GT2: Generalization to Novel Prompts & Batches")
        print("="*60)

        # TODO: Implement GT2 evaluation
        # 1. Load test prompts and test batches
        # 2. For each test prompt: encode -> network -> nearest test batch -> VLM score
        # 3. Compute per-behavior and overall scores

        gt2_results = {
            'mean_score': 0.0,
            'per_behavior': {}
        }
        results['gt2'] = gt2_results
        print("GT2 evaluation complete")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate P2I network generalization")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--protocol", type=str, choices=["gt1", "gt2", "both"],
                       default="both", help="Evaluation protocol")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    results = evaluate_generalization(
        checkpoint_path=args.checkpoint,
        eval_protocol=args.protocol,
        batch_size=args.batch_size,
    )

    # Save results
    output_path = Path(args.checkpoint).parent / "evaluation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {output_path}")
