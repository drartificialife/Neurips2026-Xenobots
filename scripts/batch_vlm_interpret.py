import os
import json
from pathlib import Path
from tqdm import tqdm
from vlm_interpret_trajectory import interpret_trajectory

# Prompts to test
PROMPTS = [
    "slow down",
    "move fast"
]

INPUT_DIR = Path("all_trajectories")
OUTPUT_DIR = INPUT_DIR

# Find all trajectory images
images = sorted(INPUT_DIR.glob("*_trajectory_macro100_prepost.png"))

for img_path in tqdm(images, desc="Processing trajectories"):
    batch_id = img_path.stem.split("_trajectory_")[0]
    for prompt in PROMPTS:
        result = interpret_trajectory(
            image_path=img_path,
            prompt=prompt,
            model="qwen3-vl:235b-cloud",
            host="http://localhost:11434",
            timeout_sec=120,
            api_mode="chat"
        )
        # result should be a dict with keys: description, score
        out_json = OUTPUT_DIR / f"{batch_id}_macro100_{prompt.replace(' ', '_')}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "batch_id": batch_id,
                "prompt": prompt,
                "description": result.get("description", ""),
                "score": result.get("score", None)
            }, f, indent=2)
