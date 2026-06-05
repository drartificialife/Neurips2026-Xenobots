#!/usr/bin/env python3
"""
Validate test prompts: for each test prompt, rank all 5 base prompts
by cosine similarity and check where the intended group lands.

Rank 1 = correct assignment, Rank 2+ = misassignment risk.
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).parent.parent

with open(PROJECT_ROOT / 'scripts' / 'test_prompts.json') as f:
    test_prompts_map = json.load(f)

with open(PROJECT_ROOT / 'scripts' / 'train_prompts.json') as f:
    base_prompts = json.load(f)

embedder         = SentenceTransformer('all-MiniLM-L6-v2')
base_embeddings  = embedder.encode(base_prompts)

print("=" * 70)
print("TEST PROMPT VALIDATION — Cosine Similarity Ranking")
print("=" * 70)

all_ranks = []
misassigned = []

for intended_base, test_list in test_prompts_map.items():
    print(f"\n[{intended_base}]")
    print(f"  {'Test Prompt':<28} {'Rank':>5}  {'Top-1 match':<16} {'Sim':>6}  {'Intended sim':>12}")
    print(f"  {'-'*28} {'-'*5}  {'-'*16} {'-'*6}  {'-'*12}")

    for test_prompt in test_list:
        emb   = embedder.encode(test_prompt)
        sims  = cosine_similarity([emb], base_embeddings)[0]

        # Rank all base prompts by similarity (high → low)
        ranked_idx = np.argsort(sims)[::-1]
        ranked     = [(base_prompts[i], sims[i]) for i in ranked_idx]

        # Find rank of intended base prompt
        intended_rank = next(i+1 for i, (bp, _) in enumerate(ranked) if bp == intended_base)
        top1_name, top1_sim = ranked[0]
        intended_sim = sims[base_prompts.index(intended_base)]

        marker = "[OK]" if intended_rank == 1 else f"[!!] rank {intended_rank}"
        print(f"  {test_prompt:<28} {intended_rank:>5}  {top1_name:<16} {top1_sim:>6.3f}  {intended_sim:>12.3f}  {marker}")

        all_ranks.append(intended_rank)
        if intended_rank > 1:
            misassigned.append({
                'prompt':   test_prompt,
                'intended': intended_base,
                'got':      top1_name,
                'sim_got':  float(top1_sim),
                'sim_intended': float(intended_sim),
            })

# ── Summary ───────────────────────────────────────────────────────
total = len(all_ranks)
rank1 = sum(1 for r in all_ranks if r == 1)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total test prompts : {total}")
print(f"Rank 1 (correct)   : {rank1}/{total} ({100*rank1/total:.0f}%)")
print(f"Rank 2+  (risk)    : {total-rank1}/{total}")

if misassigned:
    print(f"\nMisassigned prompts:")
    for m in misassigned:
        print(f"  '{m['prompt']}'")
        print(f"    intended={m['intended']} (sim={m['sim_intended']:.3f})")
        print(f"    got     ={m['got']} (sim={m['sim_got']:.3f})")
else:
    print("\nAll prompts correctly assigned!")
