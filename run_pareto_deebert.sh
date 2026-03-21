#!/bin/bash
set -e

echo "=========================================="
echo "  DeeBERT Pareto — AG News + IMDB"
echo "=========================================="

echo ""
echo "---------- AG News ----------"
echo ">>> AG News | seed=42 | Stage1 + Stage2"
python main_deebert.py \
    --config configs/deebert_agnews.yaml \
    --run_name run_01 --seed 42 \
    --output_root ./experiments_pareto/deebert/agnews
echo ""

echo ">>> AG News | Threshold sweep"
python eval_pareto_deebert.py \
    --ckpt    "./experiments_pareto/deebert/agnews/run_01/best_model_step2.pt" \
    --dataset ag_news \
    --run_name run_01 \
    --output_root ./experiments_pareto/deebert/agnews \
    --max_length 128
echo ""

echo "---------- IMDB ----------"
echo ">>> IMDB | seed=42 | Stage1 + Stage2"
python main_deebert.py \
    --config configs/deebert_imdb.yaml \
    --run_name run_01 --seed 42 \
    --output_root ./experiments_pareto/deebert/imdb
echo ""

echo ">>> IMDB | Threshold sweep"
python eval_pareto_deebert.py \
    --ckpt    "./experiments_pareto/deebert/imdb/run_01/best_model_step2.pt" \
    --dataset imdb \
    --run_name run_01 \
    --output_root ./experiments_pareto/deebert/imdb \
    --max_length 256
echo ""

echo "=========================================="
echo "  Pareto experiment complete."
echo "=========================================="
