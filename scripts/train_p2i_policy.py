#!/usr/bin/env python3
"""
Train a Prompt-to-Intervention (P2I) Policy Network.

Architecture:
    Prompt (text)
        ↓
    [SBERT Embedding: 384D]
        ↓
    [MLP Network]
        ↓
    [Intervention Parameters]

Training:
    - Input: 4 prompts (from training set)
    - Target: Optimal interventions (from evolution)
    - Loss: MSE between predicted and target interventions
    - Test: Unseen prompts (paraphrases, similar meanings)

Usage:
    python scripts/train_p2i_policy.py --epochs 1000 --lr 0.001
    python scripts/train_p2i_policy.py --load-model models/p2i_policy.pt --eval-only
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("Installing PyTorch...")
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'torch'])
    import torch
    import torch.nn as nn
    import torch.optim as optim

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing sentence-transformers...")
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'sentence-transformers'])
    from sentence_transformers import SentenceTransformer

# Paths
RESULTS_DIR = Path('results')
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)

# Expanded training data (from expand_training_data.py)
EXPANDED_DATA_FILE = Path('cache') / 'expanded_training_data.json'


class P2INetwork(nn.Module):
    """Prompt-to-Intervention Policy Network."""

    def __init__(self, embedding_dim: int = 384, hidden_dims: List[int] = None, output_dim: int = 1):
        """
        Args:
            embedding_dim: Dimension of SBERT embeddings (usually 384)
            hidden_dims: Hidden layer dimensions (default: [256, 128, 64])
            output_dim: Output dimension (intervention parameters)
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        input_dim = embedding_dim

        # Build MLP
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_expanded_training_data() -> Dict[str, Dict]:
    """Load expanded training data (base prompts + synonyms)."""
    if not EXPANDED_DATA_FILE.exists():
        print(f"Error: Expanded data not found: {EXPANDED_DATA_FILE}")
        print("Please run: python scripts/expand_training_data.py")
        exit(1)

    with open(EXPANDED_DATA_FILE, 'r') as f:
        return json.load(f)


def prepare_training_data(embedder: SentenceTransformer, expanded_data: Dict, split: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data: embeddings → intervention targets.

    Args:
        embedder: SBERT embedder
        expanded_data: Expanded training data dict
        split: 'train' or 'test'

    Returns:
        (embeddings, targets) where:
        - embeddings: (n_samples, 384)
        - targets: (n_samples, 1) normalized intervention durations
    """
    from expand_training_data import get_train_test_split

    embeddings = []
    targets = []
    prompts_used = []

    # Get appropriate split
    if split == 'train':
        train_prompts, _ = get_train_test_split()
        prompts_list = train_prompts
    else:  # test
        _, test_prompts = get_train_test_split()
        prompts_list = test_prompts

    for prompt in prompts_list:
        if prompt not in expanded_data:
            continue

        # Get embedding
        emb = embedder.encode(prompt, convert_to_tensor=False)
        embeddings.append(emb)

        # Get target intervention (duration_ms)
        duration = expanded_data[prompt]['duration_ms']
        targets.append([duration])
        prompts_used.append(prompt)

    embeddings = np.array(embeddings)
    targets = np.array(targets)

    # Normalize targets to [0, 1]
    targets_min = targets.min()
    targets_max = targets.max()
    targets_norm = (targets - targets_min) / (targets_max - targets_min)

    print(f"Loaded {len(prompts_used)} {split} prompts")
    print(f"Target range: {targets_min:.0f} - {targets_max:.0f} ms")

    return embeddings, targets_norm, targets_min, targets_max, prompts_used


def train_p2i_policy(
    epochs: int = 1000,
    lr: float = 0.001,
    batch_size: int = 1,
    device: str = 'cpu'
):
    """Train P2I policy network on expanded training data."""

    print(f"\n{'='*70}")
    print("Training Prompt-to-Intervention (P2I) Policy Network")
    print("Using EXPANDED training data (4 base + 25 synonyms = 29 total)")
    print(f"{'='*70}")

    # Load data
    print("\nLoading expanded training data...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, small SBERT
    expanded_data = load_expanded_training_data()
    embeddings, targets_norm, targets_min, targets_max, prompts_used = prepare_training_data(
        embedder, expanded_data, split='train'
    )

    print(f"Embedding dim: {embeddings.shape[1]}")

    # Setup network
    print("\nInitializing P2I Network...")
    network = P2INetwork(embedding_dim=384, output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=lr)

    # Convert to tensors
    X_train = torch.tensor(embeddings, dtype=torch.float32).to(device)
    y_train = torch.tensor(targets_norm, dtype=torch.float32).to(device)

    print(f"Network architecture:")
    print(network)

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        outputs = network(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss = {loss.item():.6f}")

    # Save model
    model_path = MODELS_DIR / 'p2i_policy.pt'
    torch.save({
        'model_state_dict': network.state_dict(),
        'embedding_dim': 384,
        'output_dim': 1,
        'targets_min': targets_min,
        'targets_max': targets_max,
    }, model_path)
    print(f"\nModel saved to: {model_path}")

    # Plot training loss
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(losses, linewidth=1.5)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('MSE Loss', fontsize=11)
        ax.set_title('P2I Policy Network Training Loss', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_plot = MODELS_DIR / 'training_loss.png'
        fig.savefig(loss_plot, dpi=150)
        plt.close()
        print(f"Loss plot saved to: {loss_plot}")
    except Exception as e:
        print(f"Could not save loss plot: {e}")

    return network, embedder, targets_min, targets_max


def evaluate_generalization(network, embedder, expanded_data, targets_min, targets_max, device: str = 'cpu'):
    """Evaluate P2I on unseen test prompts."""
    from expand_training_data import get_train_test_split

    print(f"\n{'='*70}")
    print("Generalization Testing on Unseen Test Prompts")
    print(f"{'='*70}\n")

    network.eval()

    # Get test prompts
    _, test_prompts = get_train_test_split()

    print(f"Test set size: {len(test_prompts)} unseen prompts\n")

    with torch.no_grad():
        results = []
        for test_prompt in test_prompts:
            if test_prompt not in expanded_data:
                continue

            # Get embedding
            emb = embedder.encode(test_prompt, convert_to_tensor=False)
            X_test = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)

            # Predict
            pred_norm = network(X_test).item()

            # Denormalize
            pred_duration = pred_norm * (targets_max - targets_min) + targets_min

            # Expected (from expanded data)
            expected_duration = expanded_data[test_prompt]['duration_ms']
            base_prompt = expanded_data[test_prompt]['base_prompt']

            # Error
            error_pct = abs(pred_duration - expected_duration) / expected_duration * 100

            results.append({
                'prompt': test_prompt,
                'base_prompt': base_prompt,
                'expected': expected_duration,
                'predicted': pred_duration,
                'error_pct': error_pct
            })

        # Sort by error
        results.sort(key=lambda x: x['error_pct'])

        print(f"Results (sorted by error):\n")
        for i, r in enumerate(results):
            status = "OK" if r['error_pct'] < 20 else "WARN"
            print(f"{i+1}. '{r['prompt']:25}' | Expected: {r['expected']:>7.0f} ms | Predicted: {r['predicted']:>7.0f} ms | Error: {r['error_pct']:>5.1f}% [{status}]")
            print(f"   Base: '{r['base_prompt']}'")

        # Statistics
        print(f"\n{'='*70}")
        mean_error = np.mean([r['error_pct'] for r in results])
        max_error = np.max([r['error_pct'] for r in results])
        print(f"Mean error: {mean_error:.2f}%")
        print(f"Max error:  {max_error:.2f}%")
        print(f"Good predictions (< 20% error): {sum(1 for r in results if r['error_pct'] < 20)}/{len(results)}")


def main():
    parser = argparse.ArgumentParser(description='Train P2I Policy Network')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size (default: 1)')
    parser.add_argument('--device', type=str, default='cpu', help='Device: cpu or cuda (default: cpu)')
    parser.add_argument('--load-model', type=str, default=None, help='Path to pre-trained model to load')
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate, do not train')

    args = parser.parse_args()

    # Determine device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Load expanded data
    expanded_data = load_expanded_training_data()

    if args.load_model:
        # Load pre-trained model
        print(f"Loading model from: {args.load_model}")
        checkpoint = torch.load(args.load_model, map_location=args.device)
        network = P2INetwork(embedding_dim=384, output_dim=checkpoint['output_dim']).to(args.device)
        network.load_state_dict(checkpoint['model_state_dict'])
        targets_min = checkpoint['targets_min']
        targets_max = checkpoint['targets_max']
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
    else:
        # Train new model
        if args.eval_only:
            print("Error: --eval-only requires --load-model")
            return
        network, embedder, targets_min, targets_max = train_p2i_policy(
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device
        )

    # Evaluate on unseen prompts
    evaluate_generalization(network, embedder, expanded_data, targets_min, targets_max, device=args.device)

    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
