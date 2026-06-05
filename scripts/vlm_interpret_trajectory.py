#!/usr/bin/env python3
"""
Send trajectory heatmap to VLM for fitness scoring.

Supports local Ollama models (qwen3-vl) and cloud (qwen3.5:397b-cloud).

Usage:
    # Single batch, single prompt:
    python scripts/vlm_interpret_trajectory.py --batch batch-000148 --prompt "stop moving" --model qwen3-vl

    # Compare models on same batch+prompt:
    python scripts/vlm_interpret_trajectory.py --batch batch-000148 --prompt "stop moving" --compare
"""

import argparse
import base64
import os
import re
import time
import requests
from pathlib import Path

# ---------------------------------------------------------------------------
# Scene context: describes the NEW trajectory heatmap format (single plot)
# ---------------------------------------------------------------------------
SCENE_CONTEXT = (
    "The image shows the trajectory of a single bio-bot before and after an "
    "electrical intervention, plotted on a single canvas. "
    "Blue line/dots = pre-intervention trajectory. Red line/dots = post-intervention trajectory. "
    "Cyan circle = pre start, navy X = pre end, orange circle = post start, dark-red X = post end. "
    "Axes are in pixel coordinates. The legend shows point count and total distance (px) for each phase. "
    "Longer total distance = more movement. Compare pre vs post distance to judge speed/motion changes."
)

# ---------------------------------------------------------------------------
# Generic prompt generator for any behavior
# ---------------------------------------------------------------------------
def make_prompt(behavior_name: str) -> str:
    """
    Generate a prompt for VLM to evaluate a specific behavior.

    Args:
        behavior_name: Any behavior description (e.g., "stop moving", "accelerate", "go quicker")

    Returns:
        Prompt string for VLM to score trajectory against this behavior.
    """
    return f"""{SCENE_CONTEXT}

Evaluate behavior: "{behavior_name}"

Compare pre (blue) vs post (red) trajectories. How well does the post-intervention phase match the target behavior?

Rate 0.0-1.0:
Score: <number>
Reasoning: <brief>"""

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------
MODELS = {
    # Local Ollama
    "qwen3-vl": {"host": "http://localhost:11434", "model": "qwen3-vl:latest", "api_mode": "chat"},
    "qwen3.5:35b": {"host": "http://localhost:11434", "model": "qwen3.5:35b", "api_mode": "chat"},
    "qwen3.5:122b": {"host": "http://localhost:11434", "model": "qwen3.5:122b", "api_mode": "chat"},
    # Cloud Ollama
    "qwen3.5:397b-cloud": {"host": "https://api.ollama.com", "model": "qwen3.5:397b-cloud", "api_mode": "chat"},
}

TIMEOUT_SEC = 600
BOT_TRAJECTORY_DIR = Path("bot_trajectory")


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def encode_image_b64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_score_from_response(response: str) -> float:
    # Only match "Score: <number>" pattern, no fallback to avoid grabbing random numbers
    m = re.search(r"Score:\s*([0-9]+(?:\.[0-9]*)?)", response)
    if m:
        val = float(m.group(1))
        # Clamp to 0-1 range (reject scores like 6, 10 that are clearly wrong)
        if val > 1.0:
            return -1.0
        return val
    return -1.0


def ollama_query(image_b64: str, prompt: str, model: str, host: str,
                 timeout_sec: int, api_mode: str, max_retries: int = 3) -> str:
    base = host.rstrip("/")
    headers = {}

    # Auth for cloud API
    if "api.ollama.com" in host:
        api_key = os.environ.get("OLLAMA_API_KEY")
        if not api_key:
            key_path = Path(__file__).parent.parent / "ollama_api_key.txt"
            if key_path.exists():
                api_key = key_path.read_text().strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    if api_mode == "chat":
        url = f"{base}/api/chat"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 1024},
        }
    else:
        url = f"{base}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 1024},
        }

    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout_sec)
            r.raise_for_status()
            data = r.json()
            if api_mode == "chat":
                text = (data.get("message") or {}).get("content", "")
            else:
                text = data.get("response", "")
            return text.strip()
        except requests.exceptions.ReadTimeout:
            return "Error: Timeout"
        except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
            err = str(e)
            if ('429' in err or '524' in err) and attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"    [Retry {attempt+1}/{max_retries}] waiting {wait}s...")
                time.sleep(wait)
                continue
            return f"Error: {err}"
        except Exception as e:
            return f"Error: {e}"

    return "Error: All retries exhausted"


def interpret_trajectory(image_path, prompt, model_key="qwen3-vl"):
    """Main callable for scoring."""
    cfg = MODELS.get(model_key)
    if not cfg:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")

    # Generate prompt from behavior name (accepts any behavior string)
    prompt_text = make_prompt(prompt)

    image_b64 = encode_image_b64(Path(image_path))
    response = ollama_query(image_b64, prompt_text, cfg["model"], cfg["host"],
                            TIMEOUT_SEC, cfg["api_mode"])
    score = parse_score_from_response(response)
    return {"description": response, "score": score}


def get_trajectory_image(batch_id: str) -> Path:
    """Find trajectory heatmap for a batch."""
    path = BOT_TRAJECTORY_DIR / f"{batch_id}_trajectory_heatmap.png"
    if path.exists():
        return path
    raise FileNotFoundError(f"Trajectory heatmap not found: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="VLM trajectory fitness scoring")
    parser.add_argument('--batch', type=str, required=True, help='Batch ID (batch-xxxxx)')
    parser.add_argument('--prompt', type=str, required=True,
                        help='Behavior to evaluate (e.g., "stop moving", "accelerate", "go quicker")')
    parser.add_argument('--model', type=str, default=None,
                        help=f'Model key: {", ".join(MODELS.keys())} (default: all for --compare)')
    parser.add_argument('--image', type=str, help='Override image path')
    parser.add_argument('--compare', action='store_true',
                        help='Compare all models on the same batch+prompt')
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else get_trajectory_image(args.batch)
    print(f"Image: {image_path}")
    print(f"Prompt: {args.prompt}")
    print()

    if args.compare:
        models_to_run = list(MODELS.keys())
    else:
        models_to_run = [args.model or "qwen3-vl"]

    for model_key in models_to_run:
        cfg = MODELS.get(model_key)
        if not cfg:
            print(f"[{model_key}] Unknown model, skipping\n")
            continue

        print(f"{'='*60}")
        print(f"Model: {model_key} ({cfg['model']})")
        print(f"{'='*60}")

        t0 = time.time()
        result = interpret_trajectory(image_path, args.prompt, model_key=model_key)
        elapsed = time.time() - t0

        print(f"Time: {elapsed:.1f}s")
        print(f"Score: {result['score']}")
        print(f"Response:\n{result['description']}")
        print()


if __name__ == '__main__':
    main()
