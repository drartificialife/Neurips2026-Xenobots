# Sample Xenobot Intervention Videos

This folder contains sample videos from the xenobot intervention archive, demonstrating the pre- and post-intervention behavior of motile organoids under electrical stimulation.

## Contents

### Batch 000070
- **Pre-intervention** (14 MB): Baseline xenobot motion before electrical stimulation
- **Post-intervention** (13.7 MB): Xenobot response during/after electrical stimulation
- Duration: ~131 seconds (typical batch)
- Resolution: 1280×960 pixels, 30 fps

### Batch 000073
- **Pre-intervention** (12.2 MB): Baseline xenobot motion before electrical stimulation
- **Post-intervention** (11.3 MB): Xenobot response during/after electrical stimulation
- Duration: ~131 seconds (typical batch)
- Resolution: 1280×960 pixels, 30 fps

## Purpose

These sample videos illustrate:
1. **Raw video format** of the xenobot intervention archive
2. **Pre-post intervention phases** captured in the dataset
3. **Input to the optical flow pipeline** (described in Methods section)
4. **Behavioral variability** across different xenobot specimens

## Usage

To visualize the optical flow analysis:
1. Process pre-intervention frames → motion heatmap
2. Process post-intervention frames → motion heatmap
3. Difference shows behavioral change induced by stimulation
4. Vision-language model scores alignment between observed behavior and language prompt

See `paper/neurips_2026.pdf` (Section Methods, Data) for complete pipeline description.

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
