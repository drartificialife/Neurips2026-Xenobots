# Dataset Access

## Xenobot Intervention Video Archive

The video archive used in this research contains 143 intervention-outcome videos of in-vitro motile organoids (xenobots).

### Dataset Details

- **Total videos**: 143
  - Training split: 99 videos
  - Test split: 44 videos
- **Recording specs**: 30 fps, 1280×960 resolution
- **Collection period**: October 2025 – January 2026
- **Pre-processing**: Optical flow analysis with bubble denoising

### Accessing the Data

The dataset is available upon request with institutional approval. To request access:

1. Contact the corresponding author
2. Provide:
   - Your institution and affiliation
   - Description of intended use
   - Data handling and security practices

3. Complete any required institutional data agreements

### Data Contents

Each video batch includes:
- **Raw video file** (MP4): Pre- and post-intervention phases
- **Command log** (JSON): Exact intervention parameters
  - Current magnitude: 0-50 mA
  - Stimulation angle: 0-360°
  - Frequency: 0-100 Hz
  - Duration: 1-327 seconds
- **Trajectory files**: Movement path visualizations
- **VLM scores**: Pre-computed alignment scores for all 8 behaviors × 40 prompts

### Reproducibility without Data

The code provided in this repository can be:
1. **Tested on synthetic data** using the provided scripts
2. **Evaluated on your own video data** by implementing the optical flow pipeline
3. **Extended to new domains** by implementing domain-specific VLM queries

The pre-trained checkpoint enables evaluation without retraining.

### Dataset Ethics

Research involving xenobots follows institutional guidelines for synthetic biology research. Questions about ethical use should be directed to the corresponding institution.

### Citation

If using this dataset, please cite:

```
[Citation information to be added upon publication]
```
