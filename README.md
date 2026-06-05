# Language-Guided Biological Control via Offline Learning

This repository contains code for training and evaluating a Prompt-to-Intervention (P2I) network that translates natural language descriptions into biological interventions.

## Overview

We demonstrate that a policy network can learn to map natural language prompts to biological interventions using only archived intervention-outcome video data, without any new experiments. The approach uses:

1. **Prompt encoding** (SentenceBERT) to convert text to fixed representations
2. **Multi-head P2I network** with shared backbone and behavior-specific output heads
3. **CMA-ES optimization** with vision-language model rewards
4. **Offline learning** from fixed video archives

**Key result**: Multi-head architecture achieves 66% average improvement over single-head baseline through decoupled optimization of conflicting behavioral objectives.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

Train a multi-head P2I network:

```bash
python scripts/train_multihead.py \
    --behaviors 8 \
    --population 32 \
    --generations 100 \
    --seeds 30
```

### Evaluation

Evaluate generalization on held-out data:

```bash
python scripts/evaluate_generalization.py checkpoints/multihead_seed_00.pth --protocol both
```

### Using Pre-trained Model

```python
import torch
from src.p2i_network import P2INetwork, PromptEncoder

# Load checkpoint
checkpoint = torch.load("checkpoints/multihead_cmaes_30seeds.pth")
network = P2INetwork()
network.load_state_dict(checkpoint['model_state_dict'])

# Encode a prompt
encoder = PromptEncoder()
embedding = encoder.encode("move faster")

# Get intervention predictions
with torch.no_grad():
    intervention = network(embedding.unsqueeze(0))
print(f"Intervention duration per behavior: {intervention}")
```

## Architecture

### P2I Network

```
Input: Natural language prompt (string)
  ↓
SentenceBERT embedding (384-dim vector)
  ↓
Shared CNN Backbone (Conv1d: 384→64→32→16)
  ↓
Multi-head outputs: 8 independent FC heads
  ↓
Output: Intervention duration [0,1] per behavior
```

**Parameters**:
- Shared backbone: 81,520 parameters
- Per-head: 577 parameters
- Total (multi-head): 86,136 parameters
- Total (single-head baseline): 81,577 parameters

### Optimization

- **Algorithm**: CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
- **Variant**: Separable CMA-ES for O(n) memory complexity
- **Population size**: λ = 32
- **Generations**: 100
- **Mutation strength**: σ₀ = 0.3
- **Reward signal**: Vision-language model alignment scores

## Reproducibility

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| SBERT model | all-MiniLM-L6-v2 |
| Population size | 32 |
| Generations | 100 |
| Initial mutation | 0.3 |
| Dropout rate | 0.2 |
| Conv kernel size | 3 |
| Random seeds | 30 |

### Compute Requirements

- **Hardware**: Intel i9-14900KF (24 cores, 128GB RAM) or equivalent
- **Training time**: ~15 hours total (30 seeds)
- **Per-seed time**: ~30 minutes
- **GPU**: Not required (CPU-only)
- **VLM scoring**: Cloud-based (Ollama API, ~4 parallel workers)

### Dataset Access

The video archive (143 xenobot videos) is available upon request.
See [DATA_ACCESS.md](DATA_ACCESS.md) for information on how to request access.

## Results

### Performance Comparison

Multi-head architecture vs. single-head baseline (on training batches):

| Behavior | Baseline | Single-Head | Multi-Head | Improvement |
|----------|----------|-------------|-----------|-------------|
| stop | 0.750 | 0.937 | 0.955 | +27.3% |
| motion_reduction | 0.763 | 0.872 | 0.967 | +26.7% |
| motion_increase | 0.140 | 0.674 | 0.926 | +561% |
| vigor | 0.156 | 0.254 | 0.833 | +434% |
| response_magnitude | 0.768 | 0.958 | 0.997 | +29.8% |
| directedness | 0.148 | 0.216 | 0.815 | +450% |
| confinement | 0.648 | 0.756 | 0.967 | +49.1% |
| elimination | 0.702 | 0.666 | 0.955 | +36.0% |
| **Mean** | **0.559** | **0.647** | **0.927** | **+66%** |

### Generalization Tests

- **GT1** (novel prompts, training batches): 0.844 ± 0.015
- **GT2** (novel prompts, novel batches): 0.627 ± 0.052

## File Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── DATA_ACCESS.md                     # How to access video data
├── paper/
│   ├── neurips_2026.pdf              # Published paper
│   └── figures/                       # Paper figures
├── src/
│   ├── __init__.py
│   ├── p2i_network.py                # Multi-head P2I network
│   ├── vlm_scorer.py                 # VLM reward interface
│   └── cmaes_optimizer.py            # CMA-ES trainer (optional)
├── scripts/
│   ├── train_multihead.py            # Main training script
│   ├── evaluate_generalization.py    # GT1/GT2 evaluation
│   └── download_pretrained.sh        # Download pre-trained weights
└── checkpoints/
    └── README.md                     # Checkpoint instructions
```

## Citation

If you use this code, please cite the published work:

```bibtex
@article{decoding2026,
  title={Decoding Language that Organoids Obey: Bridging Language and Biology with Foundation Models},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

## License

[To be determined upon publication]

## Acknowledgments

We thank the collaborators who provided the xenobot video archive and valuable feedback throughout this research.

---

For questions or issues, please open a GitHub issue or contact the corresponding author.
