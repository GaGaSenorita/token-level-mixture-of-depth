#!/bin/bash
set -e

echo "=========================================="
echo "  RouterBERT Pareto — AG News + IMDB"
echo "=========================================="

# ==================== AG News ====================
echo ""
echo "---------- AG News ----------"

AGNEWS_SHARED_STEP1="./experiments_pareto/routerbert/agnews/shared_step1.pt"

# Step1 只跑一次（target_keep_ratio 不影响 Step1，随便传一个）
echo ">>> AG News | Step1 (shared, run once)"
python main_routerbert.py \
    --config configs/router_tuning_agnews.yaml \
    --run_name "shared_step1" --seed 42 \
    --target_keep_ratio 0.5 \
    --output_root ./experiments_pareto/routerbert/agnews
cp "./experiments_pareto/routerbert/agnews/shared_step1/best_model_step1.pt" \
   "${AGNEWS_SHARED_STEP1}"
rm -f "./experiments_pareto/routerbert/agnews/shared_step1/best_model_step1.pt" \
      "./experiments_pareto/routerbert/agnews/shared_step1/best_model_step2.pt"
echo "Shared Step1 checkpoint saved to ${AGNEWS_SHARED_STEP1}"
echo ""

# Sweep keep_ratio，每次只训 Step2
for KR in 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    echo ">>> AG News | keep_ratio=${KR} | seed=42 | Step2 only"
    python main_routerbert.py \
        --config configs/router_tuning_agnews.yaml \
        --run_name "keep_${KR}" --seed 42 \
        --target_keep_ratio ${KR} \
        --resume_step1_ckpt ${AGNEWS_SHARED_STEP1} \
        --output_root ./experiments_pareto/routerbert/agnews
    echo ""

    echo ">>> AG News | keep_ratio=${KR} | Eval"
    python eval_pareto_routerbert.py \
        --ckpt      "./experiments_pareto/routerbert/agnews/keep_${KR}/best_model_step2.pt" \
        --keep_ratio ${KR} \
        --dataset   ag_news \
        --output_root ./experiments_pareto/routerbert/agnews \
        --max_length  128

    rm -f "./experiments_pareto/routerbert/agnews/keep_${KR}/best_model_step1.pt" \
          "./experiments_pareto/routerbert/agnews/keep_${KR}/best_model_step2.pt"
    echo ""
done

rm -f "${AGNEWS_SHARED_STEP1}"

# ==================== IMDB ====================
echo "---------- IMDB ----------"

IMDB_SHARED_STEP1="./experiments_pareto/routerbert/imdb/shared_step1.pt"

echo ">>> IMDB | Step1 (shared, run once)"
python main_routerbert.py \
    --config configs/router_tuning_imdb.yaml \
    --run_name "shared_step1" --seed 42 \
    --target_keep_ratio 0.5 \
    --output_root ./experiments_pareto/routerbert/imdb
cp "./experiments_pareto/routerbert/imdb/shared_step1/best_model_step1.pt" \
   "${IMDB_SHARED_STEP1}"
rm -f "./experiments_pareto/routerbert/imdb/shared_step1/best_model_step1.pt" \
      "./experiments_pareto/routerbert/imdb/shared_step1/best_model_step2.pt"
echo "Shared Step1 checkpoint saved to ${IMDB_SHARED_STEP1}"
echo ""

for KR in 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    echo ">>> IMDB | keep_ratio=${KR} | seed=42 | Step2 only"
    python main_routerbert.py \
        --config configs/router_tuning_imdb.yaml \
        --run_name "keep_${KR}" --seed 42 \
        --target_keep_ratio ${KR} \
        --resume_step1_ckpt ${IMDB_SHARED_STEP1} \
        --output_root ./experiments_pareto/routerbert/imdb
    echo ""

    echo ">>> IMDB | keep_ratio=${KR} | Eval"
    python eval_pareto_routerbert.py \
        --ckpt      "./experiments_pareto/routerbert/imdb/keep_${KR}/best_model_step2.pt" \
        --keep_ratio ${KR} \
        --dataset   imdb \
        --output_root ./experiments_pareto/routerbert/imdb \
        --max_length  256

    rm -f "./experiments_pareto/routerbert/imdb/keep_${KR}/best_model_step1.pt" \
          "./experiments_pareto/routerbert/imdb/keep_${KR}/best_model_step2.pt"
    echo ""
done

rm -f "${IMDB_SHARED_STEP1}"

echo "=========================================="
echo "  Pareto experiment complete."
echo "=========================================="
