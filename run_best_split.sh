#!/bin/bash
set -e

echo "=========================================="
echo "  HDC-BERT Split Layer Ablation"
echo "  sweep: split_layer in [2, 4, 6, 8, 10]"
echo "=========================================="

# ==================== AG News ====================
echo ""
echo "---------- AG News ----------"

AGNEWS_SHARED_STEP1="./experiments_split/agnews/shared_step1.pt"

echo ">>> AG News | Step1 (shared, run once)"
python main_hdcbert.py \
    --config configs/hdc_agnews.yaml \
    --run_name "shared_step1" --seed 42 \
    --split_layer 6 \
    --stage2_epochs 0 \
    --stage3_epochs 0 \
    --output_root ./experiments_split/agnews
cp "./experiments_split/agnews/shared_step1/best_model_step1.pt" \
   "${AGNEWS_SHARED_STEP1}"
rm -f "./experiments_split/agnews/shared_step1/best_model_step1.pt" \
      "./experiments_split/agnews/shared_step1/best_model_step2.pt" \
      "./experiments_split/agnews/shared_step1/best_model_step3.pt"
echo "Shared Step1 saved to ${AGNEWS_SHARED_STEP1}"
echo ""

for SL in 2 4 6 8 10; do
    echo ">>> AG News | split_layer=${SL}"
    python main_hdcbert.py \
        --config configs/hdc_agnews.yaml \
        --run_name "split_${SL}" --seed 42 \
        --split_layer ${SL} \
        --resume_step1_ckpt ${AGNEWS_SHARED_STEP1} \
        --output_root ./experiments_split/agnews

    rm -f "./experiments_split/agnews/split_${SL}/best_model_step1.pt" \
          "./experiments_split/agnews/split_${SL}/best_model_step2.pt" \
          "./experiments_split/agnews/split_${SL}/best_model_step3.pt"
    echo ""
done

rm -f "${AGNEWS_SHARED_STEP1}"

# ==================== IMDB ====================
echo "---------- IMDB ----------"

IMDB_SHARED_STEP1="./experiments_split/imdb/shared_step1.pt"

echo ">>> IMDB | Step1 (shared, run once)"
python main_hdcbert.py \
    --config configs/hdc_imdb.yaml \
    --run_name "shared_step1" --seed 42 \
    --split_layer 6 \
    --stage2_epochs 0 \
    --stage3_epochs 0 \
    --output_root ./experiments_split/imdb
cp "./experiments_split/imdb/shared_step1/best_model_step1.pt" \
   "${IMDB_SHARED_STEP1}"
rm -f "./experiments_split/imdb/shared_step1/best_model_step1.pt" \
      "./experiments_split/imdb/shared_step1/best_model_step2.pt" \
      "./experiments_split/imdb/shared_step1/best_model_step3.pt"
echo "Shared Step1 saved to ${IMDB_SHARED_STEP1}"
echo ""

for SL in 2 4 6 8 10; do
    echo ">>> IMDB | split_layer=${SL}"
    python main_hdcbert.py \
        --config configs/hdc_imdb.yaml \
        --run_name "split_${SL}" --seed 42 \
        --split_layer ${SL} \
        --resume_step1_ckpt ${IMDB_SHARED_STEP1} \
        --output_root ./experiments_split/imdb

    rm -f "./experiments_split/imdb/split_${SL}/best_model_step1.pt" \
          "./experiments_split/imdb/split_${SL}/best_model_step2.pt" \
          "./experiments_split/imdb/split_${SL}/best_model_step3.pt"
    echo ""
done

rm -f "${IMDB_SHARED_STEP1}"

echo "=========================================="
echo "  Split ablation complete."
echo "  Results in experiments_split/"
echo "=========================================="
