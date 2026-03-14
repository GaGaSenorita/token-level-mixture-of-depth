"""
主入口：负责调度 DeeBERT 两阶段训练流程
职责：参数解析 → 初始化 → Step1 → load ckpt → Step2(+early-exit eval) → 保存结果
不包含任何模型或训练细节
"""
import argparse
from pathlib import Path

import torch
from utils import set_seed, save_results, get_device, ensure_dir, setup_logger
# from model import BERTClassifier
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
    parser = argparse.ArgumentParser(description="Train DeeBERT (2-stage) with AGnews")
    parser.add_argument("--config", type=str, default=None, help="Path to yaml config")

    # ---- Experiment parameters ----
    parser.add_argument("--run_name", type=str, default="deebert_training_agnews")
    parser.add_argument("--model_key", type=str, default="deebert")

    # ---- Data parameters ----
    parser.add_argument("--dataset", type=str, default="ag_news", help="Dataset: ag_news or imdb")
    parser.add_argument("--batch_size", type=int, default=32, help="Train batch size")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")

    # ---- Model parameters ----
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Pretrained model name")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # ---- Train parameters: Step 1 ----
    parser.add_argument("--stage1_learning_rate", type=float, default=2e-5, help="Step1 learning rate")
    parser.add_argument("--stage1_epochs", type=int, default=3, help="Step1 epochs")

    # ---- Train parameters: Step 2 ----
    parser.add_argument("--stage2_learning_rate", type=float, default=1e-3, help="Step2 learning rate (heads only)")
    parser.add_argument("--stage2_epochs", type=int, default=5, help="Step2 epochs")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Gradient clip norm (step2)")

    # ---- Early-exit (eval/inference) ----
    parser.add_argument("--entropy_threshold", type=float, default=0.2, help="Early-exit entropy threshold")
    parser.add_argument(
        "--early_exit_fn",
        type=str,
        default="forward_early_exit_batchwise",
        help="Name of early-exit forward method on model"
    )
    parser.add_argument(
        "--eval_early_exit",
        action="store_true",
        help="Whether to evaluate early-exit metrics during/after step2"
    )

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

    # early-exit eval loader: batch_size=1 (sample-level semantics)
    logger.info("Building early-exit test loader (batch_size=1)...")
    _, test_loader_ee, _ = load_data(
        dataset=args.dataset,
        tokenizer_name=args.model_name,
        batch_size=1,
        max_length=args.max_length
    )
    logger.info(f"Early-exit test loader batches: {len(test_loader_ee)}")

    # -------- Model --------
    logger.info("Initializing DeeBERT model...")
    from models import DeeBERTClassifier

    model = DeeBERTClassifier(
        model_name=args.model_name,
        num_labels=num_labels,
        dropout=args.dropout
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # -------- Step 1 --------
    step1_best_path = Path(args.output_dir) / "best_model_step1.pt"
    if args.resume_step1_ckpt: # 如果有指定 checkpoint，就直接加载
        logger.info(f"Loading Step1 checkpoint from: {args.resume_step1_ckpt}")
        model.load_state_dict(torch.load(args.resume_step1_ckpt, map_location=device))
        step1_best_path = Path(args.resume_step1_ckpt)
        history_step1 = None
    else:
        logger.info("Starting Step1 training (dense fine-tune)...")
        from train import train_step1_deebert
        history_step1, best_ckpt_path = train_step1_deebert(model, train_loader, test_loader, args, device)
        step1_best_path = Path(best_ckpt_path)

        logger.info(f"Loading best Step1 checkpoint: {step1_best_path}")
        print(torch.load(step1_best_path, map_location=device).keys())
        model.load_state_dict(torch.load(step1_best_path, map_location=device))

    # -------- Step 2 --------
    logger.info("Starting Step2 training (train off-ramps)...")
    from train import train_step2_deebert

    history_step2 = train_step2_deebert(
        model,
        train_loader,
        test_loader,
        test_loader_ee, # step2 的 early-exit eval 用 batch=1 的 loader 更对齐论文语义
        args,
        device,
        entropy_threshold=args.entropy_threshold,
        eval_early_exit=args.eval_early_exit
    )

    # -------- Save results --------
    results = {
        "args": vars(args),
        "step1_best_ckpt": str(step1_best_path),
        "history_step1": history_step1,
        "history_step2": history_step2,
    }

    # 你 train_step2_deebert 里如果记录了 early-exit acc，就顺手算个 best
    if history_step2 is not None and isinstance(history_step2, dict):
        if "step2_test_acc_ee" in history_step2 and history_step2["step2_test_acc_ee"]:
            results["best_ee_acc"] = max(history_step2["step2_test_acc_ee"])
        if "step2_avg_exit_layer" in history_step2 and history_step2["step2_avg_exit_layer"]:
            results["best_avg_exit_layer"] = min(history_step2["step2_avg_exit_layer"])

    save_results(results, args.output_dir)
    logger.info("DeeBERT experiment completed successfully!")


if __name__ == "__main__":
    main()
