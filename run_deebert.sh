#!/bin/bash
set -e

echo "=========================================="
echo "  DeeBERT — AG News + IMDB"
echo "=========================================="

# -------- AG News --------
echo ""
echo "---------- AG News ----------"

echo ">>> AG News | seed=42 | Stage1 + Stage2"
python main_deebert.py \
    --config configs/deebert_agnews.yaml \
    --run_name run_01 --seed 42 \
    --output_root ./experiments/deebert/agnews
echo ""

AGNEWS_S1_CKPT="./experiments/deebert/agnews/run_01/best_model_step1.pt"

echo ">>> AG News | seed=123 | Stage2 only"
python main_deebert.py \
    --config configs/deebert_agnews.yaml \
    --run_name run_02 --seed 123 \
    --output_root ./experiments/deebert/agnews \
    --resume_step1_ckpt "$AGNEWS_S1_CKPT"
echo ""

echo ">>> AG News | seed=456 | Stage2 only"
python main_deebert.py \
    --config configs/deebert_agnews.yaml \
    --run_name run_03 --seed 456 \
    --output_root ./experiments/deebert/agnews \
    --resume_step1_ckpt "$AGNEWS_S1_CKPT"
echo ""

# -------- IMDB --------
echo "---------- IMDB ----------"

echo ">>> IMDB | seed=42 | Stage1 + Stage2"
python main_deebert.py \
    --config configs/deebert_imdb.yaml \
    --run_name run_01 --seed 42 \
    --output_root ./experiments/deebert/imdb
echo ""

IMDB_S1_CKPT="./experiments/deebert/imdb/run_01/best_model_step1.pt"

echo ">>> IMDB | seed=123 | Stage2 only"
python main_deebert.py \
    --config configs/deebert_imdb.yaml \
    --run_name run_02 --seed 123 \
    --output_root ./experiments/deebert/imdb \
    --resume_step1_ckpt "$IMDB_S1_CKPT"
echo ""

echo ">>> IMDB | seed=456 | Stage2 only"
python main_deebert.py \
    --config configs/deebert_imdb.yaml \
    --run_name run_03 --seed 456 \
    --output_root ./experiments/deebert/imdb \
    --resume_step1_ckpt "$IMDB_S1_CKPT"
echo ""

echo "=========================================="
echo "  All DeeBERT runs complete."
echo "=========================================="
