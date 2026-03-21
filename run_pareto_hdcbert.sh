#!/bin/bash
set -e

echo "=========================================="
echo "  HDC-BERT Pareto — AG News + IMDB"
echo "=========================================="

# ==================== AG News ====================
echo ""
echo "---------- AG News ----------"

AGNEWS_SHARED_STEP1="./experiments_pareto/hdcbert/agnews/shared_step1.pt"
AGNEWS_SHARED_STEP2="./experiments_pareto/hdcbert/agnews/shared_step2.pt"
AGNEWS_THRESHOLDS="0.05 0.2 0.5 0.9 1.38"

# Step1 只跑一次
echo ">>> AG News | Step1 (shared, run once)"
python main_hdcbert.py \
    --config configs/hdc_agnews.yaml \
    --run_name "shared_step1" --seed 42 \
    --target_keep_ratio 0.5 \
    --stage2_epochs 0 \
    --stage3_epochs 0 \
    --output_root ./experiments_pareto/hdcbert/agnews
cp "./experiments_pareto/hdcbert/agnews/shared_step1/best_model_step1.pt" \
   "${AGNEWS_SHARED_STEP1}"
rm -f "./experiments_pareto/hdcbert/agnews/shared_step1/best_model_step1.pt" \
      "./experiments_pareto/hdcbert/agnews/shared_step1/best_model_step2.pt" \
      "./experiments_pareto/hdcbert/agnews/shared_step1/best_model_step3.pt"
echo "Shared Step1 saved to ${AGNEWS_SHARED_STEP1}"
echo ""

# Step2 只跑一次（off-ramp 不依赖 target_keep_ratio）
echo ">>> AG News | Step2 (shared, run once)"
python main_hdcbert.py \
    --config configs/hdc_agnews.yaml \
    --run_name "shared_step2" --seed 42 \
    --target_keep_ratio 0.5 \
    --resume_step1_ckpt ${AGNEWS_SHARED_STEP1} \
    --stage3_epochs 0 \
    --output_root ./experiments_pareto/hdcbert/agnews
cp "./experiments_pareto/hdcbert/agnews/shared_step2/best_model_step2.pt" \
   "${AGNEWS_SHARED_STEP2}"
rm -f "./experiments_pareto/hdcbert/agnews/shared_step2/best_model_step1.pt" \
      "./experiments_pareto/hdcbert/agnews/shared_step2/best_model_step2.pt" \
      "./experiments_pareto/hdcbert/agnews/shared_step2/best_model_step3.pt"
echo "Shared Step2 saved to ${AGNEWS_SHARED_STEP2}"
echo ""

# Sweep keep_ratio，每次只训 Step3，eval 时 sweep entropy_threshold
for KR in 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    echo ">>> AG News | keep_ratio=${KR} | Step3 only"
    python main_hdcbert.py \
        --config configs/hdc_agnews.yaml \
        --run_name "keep_${KR}" --seed 42 \
        --target_keep_ratio ${KR} \
        --resume_step1_ckpt ${AGNEWS_SHARED_STEP1} \
        --resume_step2_ckpt ${AGNEWS_SHARED_STEP2} \
        --output_root ./experiments_pareto/hdcbert/agnews
    echo ""

    echo ">>> AG News | keep_ratio=${KR} | Eval (sweep thresholds: ${AGNEWS_THRESHOLDS})"
    python eval_pareto_hdcbert.py \
        --ckpt               "./experiments_pareto/hdcbert/agnews/keep_${KR}/best_model_step3.pt" \
        --keep_ratio         ${KR} \
        --dataset            ag_news \
        --output_root        ./experiments_pareto/hdcbert/agnews \
        --max_length         128 \
        --entropy_thresholds ${AGNEWS_THRESHOLDS}

    rm -f "./experiments_pareto/hdcbert/agnews/keep_${KR}/best_model_step1.pt" \
          "./experiments_pareto/hdcbert/agnews/keep_${KR}/best_model_step2.pt" \
          "./experiments_pareto/hdcbert/agnews/keep_${KR}/best_model_step3.pt"
    echo ""
done

rm -f "${AGNEWS_SHARED_STEP1}" "${AGNEWS_SHARED_STEP2}"

# ==================== IMDB ====================
echo "---------- IMDB ----------"

IMDB_SHARED_STEP1="./experiments_pareto/hdcbert/imdb/shared_step1.pt"
IMDB_SHARED_STEP2="./experiments_pareto/hdcbert/imdb/shared_step2.pt"
IMDB_THRESHOLDS="0.05 0.2 0.4 0.6 0.69"

echo ">>> IMDB | Step1 (shared, run once)"
python main_hdcbert.py \
    --config configs/hdc_imdb.yaml \
    --run_name "shared_step1" --seed 42 \
    --target_keep_ratio 0.5 \
    --stage2_epochs 0 \
    --stage3_epochs 0 \
    --output_root ./experiments_pareto/hdcbert/imdb
cp "./experiments_pareto/hdcbert/imdb/shared_step1/best_model_step1.pt" \
   "${IMDB_SHARED_STEP1}"
rm -f "./experiments_pareto/hdcbert/imdb/shared_step1/best_model_step1.pt" \
      "./experiments_pareto/hdcbert/imdb/shared_step1/best_model_step2.pt" \
      "./experiments_pareto/hdcbert/imdb/shared_step1/best_model_step3.pt"
echo "Shared Step1 saved to ${IMDB_SHARED_STEP1}"
echo ""

echo ">>> IMDB | Step2 (shared, run once)"
python main_hdcbert.py \
    --config configs/hdc_imdb.yaml \
    --run_name "shared_step2" --seed 42 \
    --target_keep_ratio 0.5 \
    --resume_step1_ckpt ${IMDB_SHARED_STEP1} \
    --stage3_epochs 0 \
    --output_root ./experiments_pareto/hdcbert/imdb
cp "./experiments_pareto/hdcbert/imdb/shared_step2/best_model_step2.pt" \
   "${IMDB_SHARED_STEP2}"
rm -f "./experiments_pareto/hdcbert/imdb/shared_step2/best_model_step1.pt" \
      "./experiments_pareto/hdcbert/imdb/shared_step2/best_model_step2.pt" \
      "./experiments_pareto/hdcbert/imdb/shared_step2/best_model_step3.pt"
echo "Shared Step2 saved to ${IMDB_SHARED_STEP2}"
echo ""

for KR in 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    echo ">>> IMDB | keep_ratio=${KR} | Step3 only"
    python main_hdcbert.py \
        --config configs/hdc_imdb.yaml \
        --run_name "keep_${KR}" --seed 42 \
        --target_keep_ratio ${KR} \
        --resume_step1_ckpt ${IMDB_SHARED_STEP1} \
        --resume_step2_ckpt ${IMDB_SHARED_STEP2} \
        --output_root ./experiments_pareto/hdcbert/imdb
    echo ""

    echo ">>> IMDB | keep_ratio=${KR} | Eval (sweep thresholds: ${IMDB_THRESHOLDS})"
    python eval_pareto_hdcbert.py \
        --ckpt               "./experiments_pareto/hdcbert/imdb/keep_${KR}/best_model_step3.pt" \
        --keep_ratio         ${KR} \
        --dataset            imdb \
        --output_root        ./experiments_pareto/hdcbert/imdb \
        --max_length         256 \
        --entropy_thresholds ${IMDB_THRESHOLDS}

    rm -f "./experiments_pareto/hdcbert/imdb/keep_${KR}/best_model_step1.pt" \
          "./experiments_pareto/hdcbert/imdb/keep_${KR}/best_model_step2.pt" \
          "./experiments_pareto/hdcbert/imdb/keep_${KR}/best_model_step3.pt"
    echo ""
done

rm -f "${IMDB_SHARED_STEP1}" "${IMDB_SHARED_STEP2}"

echo "=========================================="
echo "  Pareto experiment complete."
echo "=========================================="
