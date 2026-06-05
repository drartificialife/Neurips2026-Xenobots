#!/usr/bin/env python3
"""
Create video visualization for a test prompt generalization result.

Shows pre-intervention frames → intervention info → post-intervention frames

Usage:
    python scripts/create_test_visualization_video.py --prompt "move slowly" --base "slow down" --model single
    python scripts/create_test_visualization_video.py --prompt "accelerate" --base "go fast" --model multi
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from datetime import datetime

# Paths
RESULTS_DIR = Path('results')
ARCHIVE_ROOT = Path('D:\\xenobot_videos')
VIDEO_OUTPUT_DIR = Path('test_generalization_video')
VIDEO_OUTPUT_DIR.mkdir(exist_ok=True)

SINGLE_RESULTS = RESULTS_DIR / 'p2i_generalization_test.json'
MULTI_RESULTS = RESULTS_DIR / 'p2i_generalization_test_multi_prompt.json'


def load_test_results(model: str = 'single') -> Dict:
    """Load test results for specified model."""
    if model == 'single':
        results_file = SINGLE_RESULTS
    else:
        results_file = MULTI_RESULTS

    if not results_file.exists():
        raise FileNotFoundError(f"Results not found: {results_file}")

    with open(results_file) as f:
        return json.load(f)


def find_test_result(results: Dict, prompt: str, base: str) -> Dict:
    """Find test result for specific prompt."""
    for result in results['results']:
        if result['prompt'] == prompt and result['base_prompt'] == base:
            return result

    raise ValueError(f"Test result not found: '{prompt}' / '{base}'")


def get_intervention_times(batch_id: str) -> Tuple[str, str]:
    """Get intervention start and end times from batch commands."""
    batch_dir = ARCHIVE_ROOT / batch_id
    commands_dir = batch_dir / 'commands'

    if not commands_dir.exists():
        raise FileNotFoundError(f"Commands directory not found: {commands_dir}")

    # Find electrical intervention command
    electrical_commands = list(commands_dir.glob('electrical_*.json'))

    if not electrical_commands:
        raise FileNotFoundError(f"No electrical commands found in {commands_dir}")

    # Read the first (should be only) electrical command
    with open(electrical_commands[0]) as f:
        command = json.load(f)

    start_time = command.get('start_time')
    end_time = command.get('end_time')

    if not start_time or not end_time:
        raise ValueError(f"start_time or end_time not found in {electrical_commands[0]}")

    return start_time, end_time


def get_batch_frames_split(batch_id: str, start_time: str, end_time: str) -> Tuple[List[Path], List[Path]]:
    """
    Get frame images split by intervention timing.

    Returns:
        (pre_intervention_frames, post_intervention_frames)
    """
    from datetime import datetime
    import re

    batch_dir = ARCHIVE_ROOT / batch_id
    images_dir = batch_dir / 'images'

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Get all jpg files, sorted by name
    all_frames = sorted(images_dir.glob('*.jpg'))

    if not all_frames:
        raise FileNotFoundError(f"No frame images found in {images_dir}")

    # Parse intervention times (these have timezone info)
    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))

    pre_frames = []
    post_frames = []

    # Regex to find ISO datetime pattern: YYYY-MM-DDTHH.MM.SS
    timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}T\d{2}\.\d{2}\.\d{2})')

    for frame_path in all_frames:
        filename = frame_path.stem  # Remove .jpg

        try:
            # Use regex to find the ISO datetime pattern
            match = timestamp_pattern.search(filename)
            if not match:
                raise ValueError("No ISO datetime found in filename")

            timestamp_str = match.group(1)  # e.g., "2025-10-07T16.04.37"

            # Split at T
            parts = timestamp_str.split('T')
            if len(parts) != 2:
                raise ValueError("Invalid timestamp format")

            date_part = parts[0]  # "2025-10-07"
            time_part = parts[1]  # "16.04.37"

            # Convert dots to colons in time
            time_fixed = time_part.replace('.', ':')  # "16:04:37"

            frame_time_str = f"{date_part}T{time_fixed}"
            # Frame timestamps are naive (no timezone), so compare with naive versions of intervention times
            frame_dt = datetime.fromisoformat(frame_time_str)

            # Compare using only the naive datetime part (strip timezone from intervention times for comparison)
            start_naive = start_dt.replace(tzinfo=None)
            end_naive = end_dt.replace(tzinfo=None)

            if frame_dt < start_naive:
                pre_frames.append(frame_path)
            elif frame_dt > end_naive:
                post_frames.append(frame_path)
            # Frames during intervention are skipped

        except (ValueError, IndexError) as e:
            print(f"  Warning: Could not parse timestamp from {filename}: {e}")
            continue

    return pre_frames, post_frames


def create_video(
    prompt: str,
    base: str,
    batch_id: str,
    predicted_duration: float,
    vlm_score: float,
    model: str = 'single',
    fps: int = 60
) -> Path:
    """
    Create video visualization for test prompt.

    Args:
        prompt: Test prompt text
        base: Base prompt
        batch_id: Batch directory name
        predicted_duration: P2I network predicted duration (ms)
        vlm_score: VLM score (0-1)
        model: 'single' or 'multi'
        fps: Frames per second
    """

    # Get intervention timing
    print(f"Reading intervention timing from {batch_id}...")
    start_time, end_time = get_intervention_times(batch_id)
    print(f"  Start: {start_time}")
    print(f"  End: {end_time}")

    # Get frames split by intervention timing
    print(f"Loading frames from {batch_id}...")
    pre_frames, post_frames = get_batch_frames_split(batch_id, start_time, end_time)

    print(f"  Pre-intervention: {len(pre_frames)} frames")
    print(f"  Post-intervention: {len(post_frames)} frames")

    if len(pre_frames) == 0 or len(post_frames) == 0:
        raise ValueError(f"Not enough frames: pre={len(pre_frames)}, post={len(post_frames)}")

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(pre_frames[0]))
    if first_frame is None:
        raise RuntimeError(f"Could not read frame: {pre_frames[0]}")

    orig_height, orig_width = first_frame.shape[:2]
    print(f"  Original frame dimensions: {orig_width}x{orig_height}")

    # Resize to reasonable video size (max 1920 width, maintain aspect ratio)
    max_width = 1920
    if orig_width > max_width:
        scale = max_width / orig_width
        width = int(orig_width * scale)
        height = int(orig_height * scale)
    else:
        width = orig_width
        height = orig_height

    print(f"  Resized to: {width}x{height}")

    # Create video writer (include batch ID in filename)
    video_name = f"{prompt.replace(' ', '_')}_{base.replace(' ', '_')}_{model}_score_{vlm_score:.2f}_{batch_id}.mp4"
    output_path = VIDEO_OUTPUT_DIR / video_name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Could not create video writer for {output_path}")

    # Helper to add text
    def add_text(frame, text_lines, y_offset=50):
        """Add multiple lines of text to frame."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (0, 255, 0)  # Green

        for i, text in enumerate(text_lines):
            y = y_offset + (i * 40)
            cv2.putText(frame, text, (20, y), font, font_scale, color, thickness)

        return frame

    # Helper to resize frame
    def resize_frame(frame):
        """Resize frame to target dimensions."""
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return frame

    # Write pre-intervention frames
    print("Writing pre-intervention frames...")
    for frame_path in pre_frames:
        frame = cv2.imread(str(frame_path))
        frame = resize_frame(frame)
        frame = add_text(frame, ["[PRE-INTERVENTION]"], 30)
        out.write(frame)

    # Create transition frame with intervention info
    print("Writing intervention info...")
    transition_frame = np.zeros((height, width, 3), dtype=np.uint8)

    text_lines = [
        f"INTERVENTION APPLIED",
        f"",
        f"Prompt: {prompt}",
        f"Duration: {predicted_duration:.0f} ms",
        f"Score: {vlm_score:.2f}",
    ]

    transition_frame = add_text(transition_frame, text_lines, 100)

    # Show transition frame for 3 seconds
    for _ in range(int(fps * 3)):
        out.write(transition_frame)

    # Write post-intervention frames
    print("Writing post-intervention frames...")
    for frame_path in post_frames:
        frame = cv2.imread(str(frame_path))
        frame = resize_frame(frame)
        frame = add_text(frame, ["[POST-INTERVENTION]"], 30)
        out.write(frame)

    out.release()

    print(f"\nVideo saved to: {output_path}")
    print(f"Total duration: ~{(len(pre_frames) + int(fps*2) + len(post_frames)) / fps:.1f} seconds")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Create test visualization video for P2I generalization')
    parser.add_argument('--prompt', type=str, required=True, help='Test prompt')
    parser.add_argument('--base', type=str, required=True, help='Base prompt')
    parser.add_argument('--model', type=str, default='single', choices=['single', 'multi'],
                        help='Which model results to use')
    parser.add_argument('--fps', type=int, default=20, help='Frames per second (default: 20)')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"Creating Visualization Video")
    print(f"{'='*80}\n")
    print(f"Prompt: {args.prompt}")
    print(f"Base: {args.base}")
    print(f"Model: {args.model}\n")

    # Load results
    print(f"Loading {args.model}-prompt results...")
    results = load_test_results(args.model)

    # Find test result
    print(f"Finding test result...")
    result = find_test_result(results, args.prompt, args.base)

    batch_id = result['nearest_batch']
    predicted_duration = result['predicted_duration']
    vlm_score = result['vlm_score']

    print(f"  Batch: {batch_id}")
    print(f"  Duration: {predicted_duration:.0f} ms")
    print(f"  Score: {vlm_score:.2f}\n")

    # Create video
    output_path = create_video(
        args.prompt,
        args.base,
        batch_id,
        predicted_duration,
        vlm_score,
        args.model,
        args.fps
    )

    print(f"\n{'='*80}")
    print(f"SUCCESS!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
