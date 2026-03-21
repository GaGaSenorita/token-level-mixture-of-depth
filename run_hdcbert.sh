#!/bin/bash
set -e

echo "=========================================="
echo "  HDC-BERT — AG News + IMDB"
echo "  Stage 1: train once (seed=42)"
echo "  Stage 2+3: 3 seeds (42, 123, 456)"
echo "=========================================="

# ==================== AG News ====================
echo ""
echo "========== AG News =========="

echo ""
echo ">>> AG News | seed=42 | Stage1 + Stage2 + Stage3"
python main_hdcbert.py \
    --config configs/hdc_agnews.yaml \
    --run_name run_01 --seed 42 \
    --output_root ./experiments/hdc/agnews
echo ""

echo ">>> AG News | seed=123 | Stage2 + Stage3 (reuse Stage1 ckpt from run_01)"
python main_hdcbert.py \
    --config configs/hdc_agnews.yaml \
    --run_name run_02 --seed 123 \
    --output_root ./experiments/hdc/agnews \
    --resume_step1_ckpt ./experiments/hdc/agnews/run_01/best_model_step1.pt
echo ""

echo ">>> AG News | seed=456 | Stage2 + Stage3 (reuse Stage1 ckpt from run_01)"
python main_hdcbert.py \
    --config configs/hdc_agnews.yaml \
    --run_name run_03 --seed 456 \
    --output_root ./experiments/hdc/agnews \
    --resume_step1_ckpt ./experiments/hdc/agnews/run_01/best_model_step1.pt
echo ""

# ==================== IMDB ====================
echo "========== IMDB =========="

echo ""
echo ">>> IMDB | seed=42 | Stage1 + Stage2 + Stage3"
python main_hdcbert.py \
    --config configs/hdc_imdb.yaml \
    --run_name run_01 --seed 42 \
    --output_root ./experiments/hdc/imdb
echo ""

echo ">>> IMDB | seed=123 | Stage2 + Stage3 (reuse Stage1 ckpt from run_01)"
python main_hdcbert.py \
    --config configs/hdc_imdb.yaml \
    --run_name run_02 --seed 123 \
    --output_root ./experiments/hdc/imdb \
    --resume_step1_ckpt ./experiments/hdc/imdb/run_01/best_model_step1.pt
echo ""

echo ">>> IMDB | seed=456 | Stage2 + Stage3 (reuse Stage1 ckpt from run_01)"
python main_hdcbert.py \
    --config configs/hdc_imdb.yaml \
    --run_name run_03 --seed 456 \
    --output_root ./experiments/hdc/imdb \
    --resume_step1_ckpt ./experiments/hdc/imdb/run_01/best_model_step1.pt
echo ""

echo "=========================================="
echo "  All HDC-BERT runs complete."
echo "=========================================="
