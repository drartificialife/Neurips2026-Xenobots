#!/usr/bin/env python3
"""
Comprehensive prompt embedding analysis.
Computes similarity matrices, confusion matrices, and statistics for paper.

Output:
- cache/prompt_embeddings.json: embeddings + similarity matrix
- cache/prompt_similarity_stats.json: detailed statistics
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load prompts
prompts_file = Path('evolution/train_test_split/prompts.json')
with open(prompts_file) as f:
    prompts_dict = json.load(f)

# Organize
behavior_groups = {}
all_prompts = []
prompt_to_idx = {}

idx = 0
for behavior in sorted(prompts_dict.keys()):
    variants = prompts_dict[behavior]
    behavior_groups[behavior] = variants
    for v in variants:
        all_prompts.append((behavior, v))
        prompt_to_idx[(behavior, v)] = idx
        idx += 1

print(f"Total prompts: {len(all_prompts)}")
print(f"Behaviors: {len(behavior_groups)}")

# Generate embeddings
print("\nComputing embeddings...")
prompt_texts = [v for _, v in all_prompts]
embeddings = model.encode(prompt_texts, show_progress_bar=False)

# Similarity matrix
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

print("Computing similarity matrix (30x30)...")
sim_matrix = np.zeros((len(all_prompts), len(all_prompts)))
for i in range(len(all_prompts)):
    for j in range(len(all_prompts)):
        sim_matrix[i, j] = cosine_similarity(embeddings[i], embeddings[j])

# Compute statistics
print("\nComputing statistics...")

stats = {
    "total_prompts": len(all_prompts),
    "behaviors": len(behavior_groups),
    "similarity_matrix": sim_matrix.tolist(),
    "prompt_labels": [f"{b}:{p}" for b, p in all_prompts],
    "within_group": {},
    "between_group": {},
    "overlap_analysis": {},
}

# Within-group statistics
for behavior, variants in behavior_groups.items():
    indices = [i for i, (b, _) in enumerate(all_prompts) if b == behavior]

    within_sims = []
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            within_sims.append(sim_matrix[indices[i], indices[j]])

    stats["within_group"][behavior] = {
        "n_variants": len(variants),
        "n_pairs": len(within_sims),
        "variants": variants,
        "mean_similarity": float(np.mean(within_sims)),
        "std_similarity": float(np.std(within_sims)),
        "min_similarity": float(np.min(within_sims)),
        "max_similarity": float(np.max(within_sims)),
        "similarities": [float(s) for s in within_sims],
    }

# Between-group statistics
behaviors = sorted(behavior_groups.keys())
between_all = []

for bi, b1 in enumerate(behaviors):
    for bj, b2 in enumerate(behaviors):
        if bi >= bj:
            continue

        indices1 = [i for i, (b, _) in enumerate(all_prompts) if b == b1]
        indices2 = [i for i, (b, _) in enumerate(all_prompts) if b == b2]

        between_sims = []
        for i in indices1:
            for j in indices2:
                between_sims.append(sim_matrix[i, j])
                between_all.append(sim_matrix[i, j])

        key = f"{b1}_vs_{b2}"
        stats["between_group"][key] = {
            "behaviors": [b1, b2],
            "n_pairs": len(between_sims),
            "mean_similarity": float(np.mean(between_sims)),
            "std_similarity": float(np.std(between_sims)),
            "min_similarity": float(np.min(between_sims)),
            "max_similarity": float(np.max(between_sims)),
            "similarities": [float(s) for s in between_sims],
        }

# Overlap analysis
all_within = []
for behavior in behavior_groups:
    all_within.extend(stats["within_group"][behavior]["similarities"])

all_within = np.array(all_within)
all_between = np.array(between_all)

within_mean = np.mean(all_within)
within_std = np.std(all_within)
between_mean = np.mean(all_between)
between_std = np.std(all_between)

# Count overlaps
overlap_threshold_low = within_mean - within_std
overlap_threshold_high = within_mean + within_std

overlaps = np.sum((all_between >= overlap_threshold_low) & (all_between <= overlap_threshold_high))
overlap_total = len(all_between)

stats["overlap_analysis"] = {
    "within_group": {
        "mean": float(within_mean),
        "std": float(within_std),
        "min": float(np.min(all_within)),
        "max": float(np.max(all_within)),
    },
    "between_group": {
        "mean": float(between_mean),
        "std": float(between_std),
        "min": float(np.min(all_between)),
        "max": float(np.max(all_between)),
    },
    "separation": float(within_mean - between_mean),
    "overlap_pairs": int(overlaps),
    "total_between_pairs": int(overlap_total),
    "overlap_percentage": float(100 * overlaps / overlap_total),
}

# Find most confusing pairs
confusing_pairs = []
for i in range(len(all_prompts)):
    for j in range(i+1, len(all_prompts)):
        b1, p1 = all_prompts[i]
        b2, p2 = all_prompts[j]
        if b1 != b2:
            confusing_pairs.append({
                "similarity": float(sim_matrix[i, j]),
                "behavior1": b1,
                "prompt1": p1,
                "behavior2": b2,
                "prompt2": p2,
            })

confusing_pairs.sort(key=lambda x: x["similarity"], reverse=True)
stats["top_confusing_pairs"] = confusing_pairs[:20]

# Save results
cache_dir = Path('cache')
cache_dir.mkdir(exist_ok=True)

output_file = cache_dir / 'prompt_similarity_stats.json'
with open(output_file, 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\nSaved to: {output_file}")

# Print summary
print("\n" + "="*80)
print("PROMPT EMBEDDING SIMILARITY ANALYSIS - SUMMARY")
print("="*80)

print(f"\nWithin-Group Similarity (same behavior):")
print("-"*80)
for behavior in behaviors:
    d = stats["within_group"][behavior]
    print(f"  {behavior:15}: mean={d['mean_similarity']:.3f} ± {d['std_similarity']:.3f}, "
          f"n={d['n_variants']} variants")

print(f"\nBetween-Group Similarity (different behaviors):")
print("-"*80)
for key, d in stats["between_group"].items():
    b1, b2 = d["behaviors"]
    print(f"  {b1:15} <-> {b2:15}: mean={d['mean_similarity']:.3f} ± {d['std_similarity']:.3f}")

print(f"\n" + "="*80)
print("OVERLAP & CONFUSION ANALYSIS")
print("="*80)

overlap = stats["overlap_analysis"]
print(f"\nWithin-group:  mean={overlap['within_group']['mean']:.3f} +/- {overlap['within_group']['std']:.3f}")
print(f"Between-group: mean={overlap['between_group']['mean']:.3f} +/- {overlap['between_group']['std']:.3f}")
print(f"\nSeparation: {overlap['separation']:.3f}")
print(f"Overlap pairs: {overlap['overlap_pairs']}/{overlap['total_between_pairs']} ({overlap['overlap_percentage']:.1f}%)")

if overlap['overlap_percentage'] > 40:
    print("\nRISK LEVEL: CRITICAL - High embedding space confusion")
elif overlap['overlap_percentage'] > 20:
    print("\nRISK LEVEL: MODERATE - Some cross-behavior confusion expected")
else:
    print("\nRISK LEVEL: LOW - Good behavior separation")

print(f"\nTop 10 Most Confusing Cross-Behavior Pairs:")
print("-"*80)
for rank, pair in enumerate(stats["top_confusing_pairs"][:10], 1):
    print(f"{rank:2}. sim={pair['similarity']:.3f} | '{pair['prompt1']:30}' ({pair['behavior1']})")
    print(f"          | '{pair['prompt2']:30}' ({pair['behavior2']})")

# Generate heatmap
print(f"\nGenerating visualization...")

fig, ax = plt.subplots(figsize=(14, 12))

# Create labels for heatmap
labels = []
colors_map = []
behavior_colors = {
    'go faster': 0,
    'go slower': 1,
    'move fast': 2,
    'move slow': 3,
    'stop moving': 4,
}
color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

for behavior, prompt in all_prompts:
    labels.append(f"{prompt[:20]}\n({behavior})")
    colors_map.append(color_palette[behavior_colors[behavior]])

# Plot
sns.heatmap(sim_matrix,
            cmap='RdYlGn',
            center=0.5,
            vmin=0,
            vmax=1,
            xticklabels=[p[:15] for _, p in all_prompts],
            yticklabels=[p[:15] for _, p in all_prompts],
            cbar_kws={'label': 'Cosine Similarity'},
            ax=ax)

ax.set_title('Prompt Embedding Similarity Matrix (30 prompts)\nGreen=Similar, Red=Dissimilar',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Prompts', fontsize=11)
ax.set_ylabel('Prompts', fontsize=11)

# Rotate labels
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)

plt.tight_layout()
heatmap_file = cache_dir / 'prompt_similarity_heatmap.png'
plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
print(f"Saved heatmap: {heatmap_file}")
plt.close()

# Confusion matrix
print("Generating confusion matrix...")

confusion = np.zeros((len(behaviors), len(behaviors)))
for bi, b1 in enumerate(behaviors):
    for bj, b2 in enumerate(behaviors):
        if bi == bj:
            continue
        key = f"{b1}_vs_{b2}" if bi < bj else f"{b2}_vs_{b1}"
        if key in stats["between_group"]:
            confusion[bi, bj] = stats["between_group"][key]["mean_similarity"]

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(confusion,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            xticklabels=behaviors,
            yticklabels=behaviors,
            cbar_kws={'label': 'Cross-Behavior Similarity'},
            ax=ax)

ax.set_title('Cross-Behavior Confusion Matrix\n(Higher = More Likely VLM Confusion)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Behavior', fontsize=11)
ax.set_ylabel('Behavior', fontsize=11)

plt.tight_layout()
confusion_file = cache_dir / 'prompt_confusion_matrix.png'
plt.savefig(confusion_file, dpi=150, bbox_inches='tight')
print(f"Saved confusion matrix: {confusion_file}")
plt.close()

print("\n" + "="*80)
print("DONE - Results saved to cache/")
print("="*80)
