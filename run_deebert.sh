#!/bin/bash
set -e

echo "=========================================="
echo "  DeeBERT — AG News + IMDB"
echo "=========================================="

echo ""
echo "---------- AG News ----------"
echo ">>> AG News | seed=42 | Stage1 + Stage2"
python main_deebert.py \
    --config configs/deebert_agnews.yaml \
    --run_name run_01 --seed 42 \
    --output_root ./experiments/deebert/agnews
echo ""

echo "---------- IMDB ----------"
echo ">>> IMDB | seed=42 | Stage1 + Stage2"
python main_deebert.py \
    --config configs/deebert_imdb.yaml \
    --run_name run_01 --seed 42 \
    --output_root ./experiments/deebert/imdb
echo ""

echo "=========================================="
echo "  All DeeBERT runs complete."
echo "=========================================="
