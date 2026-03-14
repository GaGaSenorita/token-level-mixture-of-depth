"""
主入口：负责调度 Router-Tuning 两阶段训练流程
职责：参数解析 → 初始化 → Step1(fine-tune) → load ckpt → Step2(train router) → 保存结果
不包含任何模型或训练细节
"""
import argparse
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from utils import set_seed, save_results, get_device, ensure_dir, setup_logger
from data import load_data
import sys


def load_yaml_config(config_path: str) -> dict:
    """read yaml config file and return as dict"""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "You used --config but PyYAML is not installed. Please run: pip install pyyaml"
        ) from e

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a dict, got: {type(cfg)}")
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Train Router-Tuning BERT (2-stage) with AGnews")
    parser.add_argument("--config", type=str, default=None, help="Path to yaml config")

    # ---- Experiment parameters ----
    parser.add_argument("--run_name", type=str, default="router_tuning_agnews")
    parser.add_argument("--model_key", type=str, default="router_tuning")

    # ---- Data parameters ----
    parser.add_argument("--dataset", type=str, default="ag_news", help="Dataset: ag_news or imdb")
    parser.add_argument("--batch_size", type=int, default=32, help="Train batch size")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")

    # ---- Model parameters ----
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Pretrained model name")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # ---- Train parameters: Step 1 (标准 fine-tune BERT) ----
    parser.add_argument("--stage1_learning_rate", type=float, default=2e-5, help="Step1 learning rate")
    parser.add_argument("--stage1_epochs", type=int, default=3, help="Step1 epochs")

    # ---- Train parameters: Step 2 (冻住 backbone，只训练 router) ----
    parser.add_argument("--stage2_learning_rate", type=float, default=1e-3, help="Step2 learning rate (router only)")
    parser.add_argument("--stage2_epochs", type=int, default=5, help="Step2 epochs")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Gradient clip norm (step2)")

    # ---- Router parameters ----
    parser.add_argument("--routing_mode", type=str, default="token", choices=["token", "sample"],
                        help="'token': 逐 token 决策; 'sample': 逐序列决策")
    parser.add_argument("--tau", type=float, default=0.5, help="二值化阈值（配合全零初始化）")
    parser.add_argument("--lambda_mod", type=float, default=1e-3, help="L_MoD 惩罚系数 λ")
    parser.add_argument("--target_keep_ratio", type=float, default=0.7, help="目标保留比例 s")
    parser.add_argument("--routed_layers", type=str, default=None,
                        help="哪些层做 routing，None 表示所有层，也可传 '0,1,2' 或 'last4'，我们默认所有层")

    # ---- Reproducibility ----
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # ---- IO parameters ----
    parser.add_argument("--output_root", type=str, default="./outputs")
    parser.add_argument("--resume_step1_ckpt", type=str, default=None, help="Optional: path to step1 best ckpt")

    args_pre, _ = parser.parse_known_args()
    if args_pre.config:
        cfg = load_yaml_config(args_pre.config)
        known_keys = {a.dest for a in parser._actions}
        cfg = {k: v for k, v in cfg.items() if k in known_keys}
        parser.set_defaults(**cfg)

    args = parser.parse_args()
    return args


def parse_routed_layers(spec, n_layers):
    if spec is None or str(spec).strip().lower() in ("none", "null"):
        return None  # None 表示所有层，交给模型 __init__ 处理

    spec = str(spec).strip().lower()
    if spec == "all":
        return list(range(n_layers))
    if spec.startswith("last"):
        k = int(spec.replace("last", ""))
        k = max(1, min(k, n_layers))
        return list(range(n_layers - k, n_layers))

    # 逗号分隔的层号
    indices = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            indices.append(int(part))
    return indices


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_root) / args.run_name
    ensure_dir(output_dir)
    args.output_dir = str(output_dir)
    logger = setup_logger(output_dir, name="training")
    logger.info("Final args:")
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")

    device = get_device()
    logger.info(f"Using device: {device}")

    # -------- Data --------
    logger.info("Loading data (train/test)...")
    train_loader, test_loader, num_labels = load_data(
        dataset=args.dataset,
        tokenizer_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    logger.info(f"Data loaded. num_labels={num_labels}")

    # -------- 解析 routed_layers --------
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model_name)
    routed_layers = parse_routed_layers(args.routed_layers, config.num_hidden_layers)
    logger.info(f"Routed layers: {routed_layers if routed_layers is not None else 'all'}")

    # -------- Model --------
    logger.info("Initializing Router-Tuning BERT model...")
    from models.router_tuning_bert import RouterTuningBERTClassifier

    model = RouterTuningBERTClassifier(
        model_name=args.model_name,
        num_labels=num_labels,
        dropout=args.dropout,
        routing_mode=args.routing_mode,
        tau=args.tau,
        target_keep_ratio=args.target_keep_ratio,
        routed_layers=routed_layers,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    router_params = sum(p.numel() for p in model.routers.parameters())
    backbone_params = total_params - router_params
    logger.info(f"Total parameters:    {total_params:,}")
    logger.info(f"Router parameters:   {router_params:,} ({router_params/total_params*100:.4f}%)")
    logger.info(f"Backbone parameters: {backbone_params:,}")

    # -------- Step 1 --------
    step1_best_path = Path(args.output_dir) / "best_model_step1.pt"
    if args.resume_step1_ckpt:  # 如果有指定 checkpoint，就直接加载
        logger.info(f"Loading Step1 checkpoint from: {args.resume_step1_ckpt}")
        model.load_state_dict(torch.load(args.resume_step1_ckpt, map_location=device))
        step1_best_path = Path(args.resume_step1_ckpt)
        history_step1 = None
    else:
        logger.info("Starting Step1 training")
        from train import train_step1_router_tuning
        history_step1, best_ckpt_path = train_step1_router_tuning(model, train_loader, test_loader, args, device)
        step1_best_path = Path(best_ckpt_path)

        logger.info(f"Loading best Step1 checkpoint: {step1_best_path}")
        model.load_state_dict(torch.load(step1_best_path, map_location=device))

    # -------- Step 2 --------
    logger.info("Starting Step2 training (freeze backbone，only training router)...")
    from train import train_step2_router_tuning

    history_step2 = train_step2_router_tuning(
        model,
        train_loader,
        test_loader,
        args,
        device,
    )

    # -------- Load best Step2 checkpoint（确保 final eval 用的是最优模型）--------
    best_step2_path = Path(args.output_dir) / "best_model_step2.pt"
    if best_step2_path.exists():
        logger.info(f"Loading best Step2 checkpoint: {best_step2_path}")
        model.load_state_dict(torch.load(best_step2_path, map_location=device))
    else:
        logger.warning("best_model_step2.pt not found, final eval uses last-epoch model state.")

    # -------- Final eval（推理路径，拿 eval-time keep_rates 用于 FLOPs 估算）--------
    logger.info("Running final evaluation for report metrics...")
    from eval import evaluate_router_full
    final_acc, final_eval_keep_rates, final_avg_keep_rate = evaluate_router_full(
        model, test_loader, device
    )
    logger.info(f"Final test accuracy:       {final_acc:.4f}")
    logger.info(f"Final avg keep rate (eval): {final_avg_keep_rate:.4f}" if final_avg_keep_rate else "N/A")
    if final_eval_keep_rates:
        for i, kr in enumerate(final_eval_keep_rates):
            if kr is not None:
                logger.info(f"  Layer {i:2d} keep_rate: {kr:.4f}")

    # -------- Save results --------
    results = {
        # ---- 实验参数 ----
        "args": vars(args),

        # ---- 模型结构（report 用）----
        "model_config": {
            "model_name":  args.model_name,
            "n_layers":    config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "n_heads":     config.num_attention_heads,
            "seq_len":     args.max_length,
            "routing_mode":      args.routing_mode,
            "target_keep_ratio": args.target_keep_ratio,
            "routed_layers":     sorted(list(model.routed_layers)),
        },

        # ---- 参数量统计（report 用）----
        "parameter_counts": {
            "total":          total_params,
            "router":         router_params,
            "backbone":       backbone_params,
            "router_ratio_pct": round(router_params / total_params * 100, 6),
        },

        # ---- 训练历史 ----
        "step1_best_ckpt": str(step1_best_path),
        "history_step1":   history_step1,
        "history_step2":   history_step2,

        # ---- 最终评估指标（report 核心数据）----
        "final_eval": {
            "accuracy":             final_acc,
            "per_layer_keep_rates": final_eval_keep_rates,   # List[float|None], len=n_layers
            "avg_keep_rate":        final_avg_keep_rate,      # 所有路由层平均
        },

        # ---- FLOPs 估算所需原始数据 ----
        # 计算公式（单层 attention）:
        #   baseline_attn_FLOPs = 2 * L * 3H^2 + 4 * L^2 * H + 2 * L * H^2
        #   routed_attn_FLOPs_i = 2 * K_i * 3H^2 + 4 * K_i^2 * H + 2 * K_i * H^2
        #   where K_i = keep_rate_i * L
        #   FLOPs_saved_i = baseline - routed  (for routed layers only)
        "flops_estimation_data": {
            "n_layers":    config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "n_heads":     config.num_attention_heads,
            "seq_len":     args.max_length,
            "routing_mode":          args.routing_mode,
            "routed_layers":         sorted(list(model.routed_layers)),
            "eval_keep_rates_per_layer": final_eval_keep_rates,  # 核心：每层实际保留率
            "avg_keep_rate":             final_avg_keep_rate,
            "target_keep_ratio":         args.target_keep_ratio,
        },
    }

    # 从 step2 history 里提取最佳指标（兼容旧逻辑）
    if history_step2 is not None and isinstance(history_step2, dict):
        if "step2_test_acc" in history_step2 and history_step2["step2_test_acc"]:
            results["best_test_acc"] = max(history_step2["step2_test_acc"])
        if "step2_eval_keep_rates" in history_step2 and history_step2["step2_eval_keep_rates"]:
            results["best_epoch_eval_keep_rates"] = history_step2["step2_eval_keep_rates"][
                history_step2["step2_test_acc"].index(max(history_step2["step2_test_acc"]))
            ]

    save_results(results, args.output_dir)
    logger.info("Router-Tuning experiment completed successfully!")


if __name__ == "__main__":
    main()

## 实验主线流程说明
'''
1. main_routerbert.py → 解析参数、设置随机种子
2. data.py → 加载 AG News、tokenization、构建 DataLoader
3. models/router_tuning_bert.py → 初始化 Router-Tuning BERT 模型
4. Step1: train.py → train_step1_router_tuning() → 标准 fine-tune BERT（和 baseline 一样）
5. Step2: train.py → train_step2_router_tuning() → 冻住 backbone，只训练 router
6. eval.py → evaluate_router_full() 每个 epoch 后评估准确率 + 推理路径 keep_rates
7. Final eval → evaluate_router_full() → 获取最终 per-layer keep_rates（用于 FLOPs 估算）
8. 保存最佳模型 checkpoint 和实验结果（JSON），results 包含：
   - model_config: 模型结构参数（n_layers, hidden_size, n_heads, seq_len）
   - parameter_counts: 总参数 / router 参数 / backbone 参数
   - history_step1/2: 每个 epoch 的 loss、acc、keep_rates
   - final_eval: 最终 accuracy + 每层 eval-time keep_rate + 平均 keep_rate
   - flops_estimation_data: 计算 FLOPs 节省所需的全部原始数据
'''
