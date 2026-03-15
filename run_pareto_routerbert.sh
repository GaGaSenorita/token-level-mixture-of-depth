#!/bin/bash
set -e

KEEP_RATIOS="0.3,0.4,0.5,0.6,0.7,0.8,0.9"

echo "=========================================="
echo "  RouterBERT Pareto — AG News + IMDB"
echo "=========================================="

# ==================== AG News ====================
echo ""
echo "---------- AG News ----------"

for KR in 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    echo ">>> AG News | keep_ratio=${KR} | seed=42 | Stage1 + Stage2"
    python main_routerbert.py \
        --config configs/router_tuning_agnews.yaml \
        --run_name "keep_${KR}" --seed 42 \
        --target_keep_ratio ${KR} \
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

# ==================== IMDB ====================
echo "---------- IMDB ----------"

for KR in 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    echo ">>> IMDB | keep_ratio=${KR} | seed=42 | Stage1 + Stage2"
    python main_routerbert.py \
        --config configs/router_tuning_imdb.yaml \
        --run_name "keep_${KR}" --seed 42 \
        --target_keep_ratio ${KR} \
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

echo "=========================================="
echo "  Pareto experiment complete."
echo "=========================================="
