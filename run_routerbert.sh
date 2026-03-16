#!/bin/bash
set -e

echo "=========================================="
echo "  Router-Tuning BERT — AG News + IMDB"
echo "  Step1: train once (seed=42)"
echo "  Step2: 3 seeds (42, 123, 456)"
echo "=========================================="

# ==================== AG News ====================
echo ""
echo "========== AG News =========="

echo ""
echo ">>> AG News | seed=42 | Step1 (fine-tune) + Step2 (train router)"
python main_routerbert.py \
    --config configs/router_tuning_agnews.yaml \
    --run_name run_01 --seed 42 \
    --output_root ./experiments/router_tuning/agnews
echo ""

echo ">>> AG News | seed=123 | Step2 only (reuse Step1 ckpt from run_01)"
python main_routerbert.py \
    --config configs/router_tuning_agnews.yaml \
    --run_name run_02 --seed 123 \
    --output_root ./experiments/router_tuning/agnews \
    --resume_step1_ckpt ./experiments/router_tuning/agnews/run_01/best_model_step1.pt
echo ""

echo ">>> AG News | seed=456 | Step2 only (reuse Step1 ckpt from run_01)"
python main_routerbert.py \
    --config configs/router_tuning_agnews.yaml \
    --run_name run_03 --seed 456 \
    --output_root ./experiments/router_tuning/agnews \
    --resume_step1_ckpt ./experiments/router_tuning/agnews/run_01/best_model_step1.pt
echo ""

# ==================== IMDB ====================
echo "========== IMDB =========="

echo ""
echo ">>> IMDB | seed=42 | Step1 (fine-tune) + Step2 (train router)"
python main_routerbert.py \
    --config configs/router_tuning_imdb.yaml \
    --run_name run_01 --seed 42 \
    --output_root ./experiments/router_tuning/imdb
echo ""

echo ">>> IMDB | seed=123 | Step2 only (reuse Step1 ckpt from run_01)"
python main_routerbert.py \
    --config configs/router_tuning_imdb.yaml \
    --run_name run_02 --seed 123 \
    --output_root ./experiments/router_tuning/imdb \
    --resume_step1_ckpt ./experiments/router_tuning/imdb/run_01/best_model_step1.pt
echo ""

echo ">>> IMDB | seed=456 | Step2 only (reuse Step1 ckpt from run_01)"
python main_routerbert.py \
    --config configs/router_tuning_imdb.yaml \
    --run_name run_03 --seed 456 \
    --output_root ./experiments/router_tuning/imdb \
    --resume_step1_ckpt ./experiments/router_tuning/imdb/run_01/best_model_step1.pt
echo ""

echo "=========================================="
echo "  All Router-Tuning runs complete."
echo "=========================================="
