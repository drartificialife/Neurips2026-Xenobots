# Language-Guided Biological Control via Offline Learning (Neurips 2026)

This repository contains code and analysis for "Decoding Language that Organoids Obey: Bridging Language and Biology with Foundation Models."

## Overview

We demonstrate how to translate natural language prompts into biological interventions for in-vitro motile organoids (xenobots) using offline learning from archived intervention-outcome pairs, without any new experiments.

**Key contributions:**
1. **Offline learning framework** that leverages fixed archives of intervention-outcome videos
2. **VLM-based reward signals** using vision-language models as zero-shot evaluators  
3. **Multi-head architecture** that resolves objective interference between opposing behavioral goals

**Results:** 
- 6 of 8 behaviors significantly improve over random-retrieval baseline
- GT1 (novel prompts, training batches): 0.844 average performance
- GT2 (novel prompts, novel batches): 0.627 average performance
- Multi-head achieves 4-5× improvement on hard behaviors vs single-head

## Quick Start

### Requirements
- Python 3.9+
- PyTorch 2.0+
- SBERT for embeddings
- CMA-ES for optimization

```bash
pip install -r requirements.txt
```

### Core Training Script

```bash
# Train multi-head P2I network with CMA-ES (30 seeds)
python evolution/train_test_split/evolve_multi_prompt.py \
    --num_behaviors 8 \
    --population_size 32 \
    --generations 100 \
    --num_seeds 30
```

### Evaluation

```bash
# Run generalization tests (GT1 and GT2)
python evolution/train_test_split/compute_test_scores_vlm.py \
    --checkpoint path/to/trained_model.pth \
    --eval_protocol both
```

### Analysis & Figures

See notebooks in `paper/experiment/`:
- `cmaes_8behavior.ipynb` - Training dynamics
- `generalization_test_cmaes.ipynb` - GT1/GT2 evaluation
- `analysis_results.ipynb` - Statistical analysis

## Dataset

The xenobot intervention-outcome archive (143 videos) is available upon request with institutional approval:
- **143 total batches:** 99 training, 44 test
- **Collection period:** October 2025 – January 2026
- **Video specs:** 30 fps, 1280×960 resolution, pre- and post-intervention phases
- **Interventions:** Electrical stimulation (current, frequency, duration, angle)
- **Behaviors:** 8 motion-related phenotypes (stop, motion_increase, vigor, etc.)

To access: Contact the authors or see institutional data policies.

## Architecture

### P2I Network (Prompt-to-Intervention)

```
Input: Natural language prompt
  ↓
SentenceBERT embedding (384-dim)
  ↓
Shared CNN Backbone (Conv1d cascade: 384→64→32→16)
  ↓
Multi-head architecture: 8 independent FC heads
  ↓
Output: Intervention duration [0,1] per behavior
```

**Parameters:**
- Shared backbone: 81,520 params
- Per-head: 577 params × 8 = 4,616 params
- Total: 86,136 params

### Optimization

- **Algorithm:** CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
- **Variant:** Separable CMA-ES (sep-CMA-ES) for O(n) memory
- **Population:** 32
- **Generations:** 100
- **Fitness:** Mean VLM score across training prompts

### Reward Signal

Vision-Language Model (Qwen3.5-397B) scores behavior-prompt alignment:
1. Compute optical flow pre- and post-intervention
2. Generate motion heatmap visualization
3. Query VLM: "Does this xenobot behavior match the prompt?"
4. Output: Alignment score [0,1]

## File Structure

```
.
├── evolution/
│   └── train_test_split/
│       ├── evolve_multi_prompt.py      # Main multi-head training
│       ├── evolve_single_head.py        # Single-head baseline
│       ├── compute_test_scores_vlm.py   # GT1/GT2 evaluation
│       └── plot_training_comparison.py  # Training curves
├── paper/
│   ├── neurips_2026.tex                 # Main paper
│   ├── checklist.tex                    # NeurIPS reproducibility checklist
│   └── experiment/
│       ├── cmaes_8behavior.ipynb        # Training analysis
│       ├── generalization_test_cmaes.ipynb # Generalization tests
│       └── analysis_results.ipynb       # Statistical analysis
├── video_preprocessing/
│   └── generate_motion_heatmap.py       # Optical flow pipeline
├── requirements.txt
├── .gitignore
└── README.md (this file)
```

## Reproducibility

### Key Hyperparameters
- CMA-ES population size: 32
- Initial mutation strength (σ₀): 0.3
- Number of generations: 100
- SBERT model: all-MiniLM-L6-v2 (384-dim)
- VLM model: Qwen3.5-397B (via Ollama Cloud)

### Compute Requirements
- **Hardware:** Intel i9-14900KF (24 cores, 128GB RAM) or similar
- **Training time:** ~15 hours wall-clock (30 seeds × 30 min each)
- **VLM scoring:** Cloud-based (Ollama API, ~4 parallel workers)
- **GPU:** Not required (CPU-only)

### Reproducibility Checklist
- ✅ Complete hyperparameter specification (Appendix B)
- ✅ Data splits documented (99 train, 44 test batches)
- ✅ Prompt sets listed (40 training + 40 test, Appendix E)
- ✅ Statistical methods (Wilcoxon, Cohen's d, Appendix C)
- ✅ Ablation studies (single-head vs multi-head, Appendix A.1)
- ✅ Per-behavior generalization statistics (Appendix F)

## Results Summary

### Multi-Head vs Single-Head (on training batches, GT1)

| Behavior | Baseline | Single-Head | Multi-Head | Improvement |
|----------|----------|-------------|-----------|-------------|
| stop | 0.750 | 0.937 | 0.955 | +27.3% |
| motion_reduction | 0.763 | 0.872 | 0.967 | +26.7% |
| motion_increase | 0.140 | 0.674 | 0.926 | +560% |
| vigor | 0.156 | 0.254 | 0.833 | +434% |
| response_magnitude | 0.768 | 0.958 | 0.997 | +29.8% |
| directedness | 0.148 | 0.216 | 0.815 | +450% |
| confinement | 0.648 | 0.756 | 0.967 | +49.1% |
| elimination | 0.702 | 0.666 | 0.955 | +36.0% |
| **Mean** | **0.559** | **0.647** | **0.927** | **+66%** |

### Generalization (GT2: Novel Prompts × Novel Batches)
- 6 of 8 behaviors maintain significant improvements
- motion_reduction, confinement show distribution shift effects
- Hard behaviors (vigor, directedness, motion_increase) sustain 100%+ improvements

## Citation

```bibtex
@article{le2026decoding,
  title={Decoding Language that Organoids Obey: Bridging Language and Biology with Foundation Models},
  author={Le, Nam and Blackiston, Douglas and Levin, Michael and Bongard, Josh},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

## Broader Impact

This work enables natural language control of biological systems, lowering barriers for non-expert researchers. Potential benefits include:
- Drug delivery optimization
- Biological material synthesis
- Microplastic remediation

Key risks:
- Non-experts may not recognize unintended consequences
- Misuse potential for creating destructive interventions

**Safeguards:** System retrieves only from fixed archive; cannot generate novel interventions.

## License

[To be determined upon acceptance]

## Contact

For questions or data access requests, contact the authors or submit an issue.

---

Generated for NeurIPS 2026 submission. Updated: June 2026.
