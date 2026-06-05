#!/usr/bin/env python3
"""
extract_kinematics_from_video.py

Extract pre/post kinematics using the SAME pipeline as generate_trajectory_heatmap.py.

WHY THIS EXISTS
---------------
The earlier kinematics_cache.json used step=10, scale=0.5, and a jump filter —
which introduced tracking errors (see spot_check results: batch-000082, 000072).

The heatmap pipeline (step=5, full resolution, no jump filter) is validated:
we visually confirmed its output matches real bot behavior.

This script runs that same pipeline but saves distances to JSON instead of plotting.

OUTPUT
------
  cache/kinematics_v2.json   — pre_dist, post_dist, pre_n, post_n, pre_speed, post_speed

USAGE
-----
  python scripts/extract_kinematics_from_video.py              # 4 workers, step=5
  python scripts/extract_kinematics_from_video.py --workers 8  # faster
  python scripts/extract_kinematics_from_video.py --recalc     # recompute all
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from parse_interventions import load_batch_with_interventions

CACHE_V2     = PROJECT_ROOT / 'cache' / 'kinematics_v2.json'
ARCHIVE_JSON = PROJECT_ROOT / 'cache' / 'archive_vlm_scores_v2.json'


# ── Tracking functions — exact copy of generate_trajectory_heatmap.py ──
# No scale reduction. No jump filter. step=5 (fast but accurate).

def _init_bot_template(first_image_path: Path) -> Optional[dict]:
    """Detect bot in first frame and build template. Full resolution."""
    img = cv2.imread(str(first_image_path))
    if img is None:
        return None

    h, w   = img.shape[:2]
    cx, cy = w // 2, h // 2
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 10
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    dish_radius = int(min(w, h) * 0.4)
    cv2.circle(mask, (cx, cy), dish_radius, 255, -1)
    adaptive = cv2.bitwise_and(adaptive, adaptive, mask=mask)

    contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if 5 < area < 5000:
            M = cv2.moments(c)
            if M['m00'] > 0:
                bx = M['m10'] / M['m00']
                by = M['m01'] / M['m00']
                dist = np.hypot(bx - cx, by - cy)
                candidates.append((dist, area, bx, by))

    if not candidates:
        return None

    candidates.sort()
    bot_x, bot_y = candidates[0][2], candidates[0][3]

    tmpl_half = 80
    tx1 = max(0, int(bot_x - tmpl_half))
    ty1 = max(0, int(bot_y - tmpl_half))
    tx2 = min(w, int(bot_x + tmpl_half))
    ty2 = min(h, int(bot_y + tmpl_half))
    template = gray[ty1:ty2, tx1:tx2]

    if template.size == 0:
        return None

    th, tw = template.shape
    search_mask = np.zeros((h - th + 1, w - tw + 1), dtype=np.uint8)
    cv2.circle(search_mask, (cx - tmpl_half, cy - tmpl_half), dish_radius, 255, -1)

    return {
        'template':    template,
        'search_mask': search_mask,
        'tmpl_half':   tmpl_half,
    }


def _detect(image_path: Path, tracker: dict) -> Optional[Tuple[float, float]]:
    """Detect bot in one frame via template matching. Full resolution."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(gray, tracker['template'], cv2.TM_CCOEFF_NORMED)

    result_masked = np.where(tracker['search_mask'] > 0, result, -1.0)
    _, max_val, _, max_loc = cv2.minMaxLoc(result_masked)

    if max_val < 0.5:
        return None

    tmpl_half = tracker['tmpl_half']
    return (float(max_loc[0] + tmpl_half), float(max_loc[1] + tmpl_half))


def _total_distance(positions: List[Tuple[float, float]]) -> float:
    dist = 0.0
    for i in range(len(positions) - 1):
        dx = positions[i+1][0] - positions[i][0]
        dy = positions[i+1][1] - positions[i][1]
        dist += np.hypot(dx, dy)
    return dist


def _track(image_paths: List[str], step: int,
           tracker: dict) -> List[Tuple[float, float]]:
    """Track bot across frames. No jump filter — same as heatmap pipeline."""
    positions = []
    total   = len(image_paths)
    indices = list(range(0, total, step))
    if total > 0 and (total - 1) not in indices:
        indices.append(total - 1)
    for i in indices:
        pos = _detect(Path(image_paths[i]), tracker)
        if pos is not None:
            positions.append(pos)
    return positions


# ── Per-batch computation ─────────────────────────────────────────

def compute_one(batch_id: str, step: int) -> dict:
    """
    Track bot for one batch and return kinematics.

    Returns dict with pre_dist, post_dist, speeds, or 'error' key on failure.
    """
    try:
        records = load_batch_with_interventions(batch_id)
    except Exception as e:
        return {'error': str(e)}

    if not records:
        return {'error': 'no records'}

    record = records[0]
    if record.get('type') != 'intervention':
        return {'error': 'not intervention type'}

    pre_images  = record.get('pre_images',  [])
    post_images = record.get('post_images', [])

    if not pre_images or not post_images:
        return {'error': 'missing pre/post images'}

    tracker = _init_bot_template(Path(pre_images[0]))
    if tracker is None:
        return {'error': 'template init failed'}

    pre_pos  = _track(pre_images,  step, tracker)
    post_pos = _track(post_images, step, tracker)

    if len(pre_pos) < 2 or len(post_pos) < 2:
        return {'error': f'too few positions: pre={len(pre_pos)}, post={len(post_pos)}'}

    pre_dist  = _total_distance(pre_pos)
    post_dist = _total_distance(post_pos)
    pre_n     = len(pre_pos)
    post_n    = len(post_pos)

    # Speed = total displacement / number of tracked points
    # Units: pixels per tracked frame (at original resolution)
    pre_speed  = pre_dist  / pre_n  if pre_n  > 0 else 0.0
    post_speed = post_dist / post_n if post_n > 0 else 0.0

    return {
        'pre_dist':    float(pre_dist),
        'post_dist':   float(post_dist),
        'pre_n':       pre_n,
        'post_n':      post_n,
        'pre_speed':   float(pre_speed),   # px / tracked-frame, full-res
        'post_speed':  float(post_speed),
        'duration_ms': record.get('duration_ms'),
        'step':        step,
    }


def _worker(args):
    batch_id, step = args
    return batch_id, compute_one(batch_id, step)


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step',    type=int, default=5,
                        help='Frame step (default 5 — same quality as step=1 heatmap, 5x faster)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Parallel workers (default 4)')
    parser.add_argument('--recalc',  action='store_true',
                        help='Ignore cache and recompute all')
    args = parser.parse_args()

    # Load archive batch list
    with open(ARCHIVE_JSON) as f:
        archive = json.load(f)
    batch_ids = sorted([k for k in archive
                        if archive[k].get('duration_ms') is not None])
    print(f"Archive: {len(batch_ids)} batches")

    # Load cache
    cache = {}
    if CACHE_V2.exists() and not args.recalc:
        with open(CACHE_V2) as f:
            cache = json.load(f)
        print(f"Cache: {len(cache)} entries loaded")

    todo = [b for b in batch_ids if b not in cache or args.recalc]
    print(f"To process: {len(todo)}  (step={args.step}, workers={args.workers}, "
          f"full resolution, no jump filter)")

    if not todo:
        print("All done — nothing to recompute.")
    else:
        tasks     = [(b, args.step) for b in todo]
        done      = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, t): t[0] for t in tasks}
            for future in as_completed(futures):
                batch_id, result = future.result()
                cache[batch_id]  = result
                done += 1
                if 'error' in result:
                    print(f"  [{done}/{len(todo)}] {batch_id}  ERROR: {result['error']}")
                else:
                    print(f"  [{done}/{len(todo)}] {batch_id}"
                          f"  pre={result['pre_dist']:.0f}px"
                          f"  post={result['post_dist']:.0f}px"
                          f"  pre_spd={result['pre_speed']:.3f}"
                          f"  post_spd={result['post_speed']:.3f}")
                with open(CACHE_V2, 'w') as f:
                    json.dump(cache, f, indent=2)

    # Summary
    valid  = {b: cache[b] for b in batch_ids
              if b in cache and 'error' not in cache[b]}
    errors = {b: cache[b]['error'] for b in batch_ids
              if b in cache and 'error' in cache[b]}
    print(f"\nValid: {len(valid)} | Errors: {len(errors)}")
    if errors:
        for b, e in list(errors.items())[:5]:
            print(f"  {b}: {e}")

    if valid:
        pre_speeds  = [v['pre_speed']  for v in valid.values()]
        post_speeds = [v['post_speed'] for v in valid.values()]
        import numpy as np
        print(f"\nPre-speed  (px/tracked-frame): "
              f"median={np.median(pre_speeds):.3f}  max={np.max(pre_speeds):.1f}")
        print(f"Post-speed (px/tracked-frame): "
              f"median={np.median(post_speeds):.3f}  max={np.max(post_speeds):.1f}")
        print(f"\nSaved: {CACHE_V2}")


if __name__ == '__main__':
    main()
