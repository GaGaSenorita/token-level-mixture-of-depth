#!/bin/bash
set -e

SEEDS=(42 123 456)

echo "=========================================="
echo "  DenseBERT Baseline — AG News + IMDB"
echo "=========================================="

echo ""
echo "---------- AG News ----------"
for seed in "${SEEDS[@]}"; do
    echo ">>> AG News | seed=$seed"
    python main_densebert.py --config configs/bert_baseline_agnews.yaml --seed "$seed"
    echo ""
done

echo "---------- IMDB ----------"
for seed in "${SEEDS[@]}"; do
    echo ">>> IMDB | seed=$seed"
    python main_densebert.py --config configs/bert_baseline_imdb.yaml --seed "$seed"
    echo ""
done

echo "=========================================="
echo "  All baseline runs complete."
echo "=========================================="
