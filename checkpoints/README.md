# Pre-trained Model Checkpoints

## Downloading the Multi-Head Model

The multi-head P2I network trained on 30 random seeds is available for download:

### Option 1: Direct Download
```bash
wget https://zenodo.org/record/[ID]/multihead_cmaes_30seeds.pth
# or
curl -o multihead_cmaes_30seeds.pth https://zenodo.org/record/[ID]/multihead_cmaes_30seeds.pth
```

### Option 2: Using the provided script
```bash
bash scripts/download_pretrained.sh
```

## Checkpoint Contents

The checkpoint file (`multihead_cmaes_30seeds.pth`) contains:
- **model_state_dict**: Trained network weights
- **hyperparameters**: Configuration used during training
- **training_metadata**: Training fitness curves (30 seeds)
- **validation_metadata**: Validation performance metrics

## Usage

Load and use the checkpoint:

```python
import torch
from src.p2i_network import P2INetwork

# Load checkpoint
checkpoint = torch.load("checkpoints/multihead_cmaes_30seeds.pth")
network = P2INetwork(num_behaviors=8)
network.load_state_dict(checkpoint['model_state_dict'])
network.eval()

# Use for inference
embeddings = torch.randn(1, 384)  # SentenceBERT embedding
outputs = network(embeddings)  # (1, 8) - one output per behavior
```

## Training Details

The checkpoint was trained using:
- **Algorithm**: CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
- **Variant**: Separable CMA-ES (sep-CMA-ES) for O(n) memory
- **Population size**: 32
- **Generations**: 100
- **Mutation strength**: σ₀ = 0.3
- **Random seeds**: 30 independent runs
- **Training time**: ~15 hours total (CPU-only)

## Generalization Performance

On held-out test data:
- **GT1** (novel prompts, training batches): 0.844 ± 0.015 average score
- **GT2** (novel prompts, novel batches): 0.627 ± 0.052 average score

Per-behavior results available in evaluation output.

## Data Access

The video archive (143 xenobot intervention videos) is available upon request due to institutional data policies.
See [DATA_ACCESS.md](../DATA_ACCESS.md) for details.
