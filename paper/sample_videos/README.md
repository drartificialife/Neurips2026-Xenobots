# Sample Xenobot Intervention Videos & Optical Flow Heatmaps

This folder contains sample videos and optical flow visualizations from the xenobot intervention archive, demonstrating the pre- and post-intervention behavior of motile organoids under electrical stimulation.

The heatmaps are the primary input to the Vision-Language Model (VLM) for computing behavior-prompt alignment scores.

## Contents

### Batch 000070 - Videos
- **Pre-intervention** (14 MB): Baseline xenobot motion before electrical stimulation
- **Post-intervention** (13.7 MB): Xenobot response during/after electrical stimulation
- Duration: ~131 seconds (typical batch)
- Resolution: 1280×960 pixels, 30 fps

### Batch 000070 - Optical Flow Visualizations
- **motion_comparison.png** (1.83 MB): Side-by-side pre/post intervention motion heatmaps
  - Shows optical flow magnitude: blue (pre-intervention), red (post-intervention)
  - This is the PRIMARY INPUT to VLM for scoring
- **heatmap_denoised.png** (0.22 MB): Cleaned motion heatmap (bubble artifacts removed)

### Batch 000073 - Videos
- **Pre-intervention** (12.2 MB): Baseline xenobot motion before electrical stimulation
- **Post-intervention** (11.3 MB): Xenobot response during/after electrical stimulation
- Duration: ~131 seconds (typical batch)
- Resolution: 1280×960 pixels, 30 fps

### Batch 000073 - Optical Flow Visualizations
- **heatmap_denoised.png** (0.21 MB): Cleaned motion heatmap (bubble artifacts removed)
  - Shows optical flow magnitude after denoising
  - Input to VLM for behavior-prompt alignment scoring

## Purpose

These sample videos illustrate:
1. **Raw video format** of the xenobot intervention archive
2. **Pre-post intervention phases** captured in the dataset
3. **Input to the optical flow pipeline** (described in Methods section)
4. **Behavioral variability** across different xenobot specimens

## VLM Pipeline - How Heatmaps are Used

The optical flow heatmaps are the core input to the vision-language model scoring system:

```
Raw Video → Optical Flow Analysis → Motion Heatmap → VLM
                                    (pre/post)       + Prompt
                                                       ↓
                                                   Alignment Score [0,1]
```

### Heatmap Processing Steps:
1. **Frame Differencing**: Detect motion regions between consecutive frames
2. **Bubble Denoising**: Remove optical artifacts (HSV saturation thresholding)
3. **Optical Flow**: Compute dense motion vectors (Farneback method)
4. **Heatmap Visualization**: Accumulate motion magnitude across frames
   - Blue regions: pre-intervention baseline motion
   - Red regions: post-intervention response motion
5. **VLM Scoring**: Vision-language model evaluates behavior-prompt alignment
   - Input: Pre/post motion heatmaps + language description
   - Output: Alignment score [0, 1]

### Example VLM Query:
```
Prompt: "move faster"
Heatmap: [pre-intervention baseline] vs [post-intervention response]
VLM Response: "The xenobot shows increased motion velocity post-intervention.
              Alignment score: 0.87"
```

See `paper/neurips_2026.pdf` (Section Methods) for complete pipeline description and mathematical formulation.

## Data Access

This is a small sample for illustration purposes. The complete video archive (143 batches) is available upon request with institutional approval.

To request access to the full dataset:
- See [DATA_ACCESS.md](../DATA_ACCESS.md) in the repository root
- Provide institutional affiliation and intended use
- Complete any required data use agreements

## Video Specifications

- **Format**: MP4 (H.264 codec)
- **Resolution**: 1280×960 pixels
- **Frame rate**: 30 fps
- **Recording setup**: Multi-electrode array chamber with optical microscopy
- **Pre-intervention duration**: ~5-30 seconds baseline
- **Post-intervention duration**: ~100-300 seconds response period

## Citation

If using these sample videos in presentations or derivative work, please cite:

```
[Citation information to be added upon publication]
```

---

**Note**: These are representative samples. The intervention parameters (current, frequency, angle, duration) vary across batches. See the full archive documentation for complete intervention metadata.
