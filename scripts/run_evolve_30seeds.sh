#!/bin/bash
# Run 30 P2I evolution seeds sequentially
# Usage: bash scripts/run_evolve_30seeds.sh

PYTHON="/c/Users/namle/miniconda3/envs/mombot/python.exe"
GENS=100
POP=100

echo "Running 30 evolution seeds sequentially (gens=$GENS, pop=$POP)..."
echo ""

for seed in $(seq 1 29); do
    echo "=== Seed $seed / 29 ==="
    $PYTHON scripts/evolve_p2i_multi_prompt.py \
        --generations $GENS \
        --population-size $POP \
        --seed $seed \
        --device cuda \
        2>&1 | tee results/p2i_evolve_seed${seed}.log
    echo ""
done

echo "Done! All 30 seeds complete."
