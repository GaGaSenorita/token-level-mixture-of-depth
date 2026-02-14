"""
主入口：负责调度 Router-Tuning 两阶段训练流程
职责：参数解析 → 初始化 → Step1(fine-tune) → load ckpt → Step2(train router) → 保存结果
不包含任何模型或训练细节
"""
import argparse
from pathlib import Path

import torch
from utils import set_seed, save_results, get_device, ensure_dir, setup_logger
from data import load_ag_news_data
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
    train_loader, test_loader, num_labels = load_ag_news_data(
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
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Router parameters: {router_params:,} ({router_params/total_params*100:.4f}%)")

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

    # -------- Save results --------
    results = {
        "args": vars(args),
        "step1_best_ckpt": str(step1_best_path),
        "history_step1": history_step1,
        "history_step2": history_step2,
    }

    # 从 step2 history 里提取最佳指标
    if history_step2 is not None and isinstance(history_step2, dict):
        if "step2_test_acc" in history_step2 and history_step2["step2_test_acc"]:
            results["best_test_acc"] = max(history_step2["step2_test_acc"])
        if "step2_keep_rates" in history_step2 and history_step2["step2_keep_rates"]:
            results["final_keep_rates"] = history_step2["step2_keep_rates"][-1]

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
6. eval.py → evaluate_router() 每个 epoch 后评估准确率
7. 保存最佳模型 checkpoint 和实验结果（JSON）
'''
