"""
主入口：负责调度 HDC-BERT 三阶段训练流程
职责：参数解析 → 初始化 → Stage1(fine-tune) → Stage2(train off-ramps) → Stage3(train routers) → 保存结果
不包含任何模型或训练细节
"""
import argparse
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from utils import set_seed, save_results, get_device, ensure_dir, setup_logger
from data import load_data


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
    parser = argparse.ArgumentParser(description="Train HDC-BERT (3-stage) with AGnews")
    parser.add_argument("--config", type=str, default=None, help="Path to yaml config")

    # ---- Experiment parameters ----
    parser.add_argument("--run_name", type=str, default="hdc_bert_agnews")
    parser.add_argument("--model_key", type=str, default="hdc_bert")

    # ---- Data parameters ----
    parser.add_argument("--dataset", type=str, default="ag_news", help="Dataset: ag_news or imdb")
    parser.add_argument("--batch_size", type=int, default=32, help="Train batch size")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")

    # ---- Model parameters ----
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Pretrained model name")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # ---- HDC-specific parameters ----
    parser.add_argument("--split_layer", type=int, default=6,
                        help="分界层: [0..split_layer-1] = Stage A (early-exit), "
                             "[split_layer..n-1] = Stage B (token routing)")
    parser.add_argument("--tau", type=float, default=0.5, help="STE 二值化阈值")
    parser.add_argument("--target_keep_ratio", type=float, default=0.7,
                        help="Stage B token routing 保留目标")
    parser.add_argument("--lambda_mod", type=float, default=1e-3, help="Budget loss 系数 (Stage 3)")

    # ---- Train parameters: Stage 1 (标准 fine-tune BERT) ----
    parser.add_argument("--stage1_learning_rate", type=float, default=2e-5, help="Stage 1 learning rate")
    parser.add_argument("--stage1_epochs", type=int, default=3, help="Stage 1 epochs")

    # ---- Train parameters: Stage 2 (训练 off-ramp classifiers) ----
    parser.add_argument("--stage2_learning_rate", type=float, default=1e-3, help="Stage 2 learning rate")
    parser.add_argument("--stage2_epochs", type=int, default=5, help="Stage 2 epochs")

    # ---- Train parameters: Stage 3 (训练 token routers) ----
    parser.add_argument("--stage3_learning_rate", type=float, default=1e-3, help="Stage 3 learning rate")
    parser.add_argument("--stage3_epochs", type=int, default=5, help="Stage 3 epochs")

    # ---- Early-exit evaluation ----
    parser.add_argument("--entropy_threshold", type=float, default=0.2,
                        help="Stage A early-exit 熵阈值")
    parser.add_argument("--eval_early_exit", action="store_true",
                        help="训练中是否评估完整 HDC 推理")

    # ---- Reproducibility ----
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # ---- IO parameters ----
    parser.add_argument("--output_root", type=str, default="./outputs")
    parser.add_argument("--resume_step1_ckpt", type=str, default=None,
                        help="Optional: path to Stage 1 best ckpt")
    parser.add_argument("--resume_step2_ckpt", type=str, default=None,
                        help="Optional: path to Stage 2 best ckpt")

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
        logger.info(f"  {k}: {v}")

    device = get_device()
    logger.info(f"Using device: {device}")

    # -------- Data --------
    logger.info("Loading data (train/test)...")
    train_loader, test_loader, num_labels = load_data(
        dataset=args.dataset,
        tokenizer_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    # batch_size=1 loader for early-exit evaluation
    _, test_loader_ee, _ = load_data(
        dataset=args.dataset,
        tokenizer_name=args.model_name,
        batch_size=1,
        max_length=args.max_length,
    )
    logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    logger.info(f"Data loaded. num_labels={num_labels}")

    # -------- Model --------
    logger.info("Initializing HDC-BERT model...")
    from models.hdc_bert import HDCBERTClassifier

    model = HDCBERTClassifier(
        model_name=args.model_name,
        num_labels=num_labels,
        dropout=args.dropout,
        split_layer=args.split_layer,
        tau=args.tau,
        target_keep_ratio=args.target_keep_ratio,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    offramp_params = sum(p.numel() for p in model.offramp_classifiers.parameters())
    router_params = sum(p.numel() for p in model.routers.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Off-ramp parameters: {offramp_params:,} ({offramp_params/total_params*100:.4f}%)")
    logger.info(f"Router parameters: {router_params:,} ({router_params/total_params*100:.4f}%)")
    logger.info(f"Stage A layers: 0..{args.split_layer - 1} ({args.split_layer} layers)")
    logger.info(f"Stage B layers: {args.split_layer}..{model.n_layers - 1} ({model.n_layers - args.split_layer} layers)")

    # -------- Stage 1: Dense fine-tune --------
    step1_best_path = Path(args.output_dir) / "best_model_step1.pt"
    if args.resume_step1_ckpt:
        logger.info(f"Loading Stage 1 checkpoint from: {args.resume_step1_ckpt}")
        model.load_state_dict(torch.load(args.resume_step1_ckpt, map_location=device))
        step1_best_path = Path(args.resume_step1_ckpt)
        history_step1 = None
    else:
        logger.info("Starting Stage 1: Dense fine-tuning...")
        from train import train_step1_hdc
        history_step1, best_ckpt_path = train_step1_hdc(model, train_loader, test_loader, args, device)
        step1_best_path = Path(best_ckpt_path)

        logger.info(f"Loading best Stage 1 checkpoint: {step1_best_path}")
        model.load_state_dict(torch.load(step1_best_path, map_location=device))

    # -------- Stage 2: Off-ramp training --------
    step2_best_path = Path(args.output_dir) / "best_model_step2.pt"
    if args.resume_step2_ckpt:
        logger.info(f"Loading Stage 2 checkpoint from: {args.resume_step2_ckpt}")
        model.load_state_dict(torch.load(args.resume_step2_ckpt, map_location=device))
        step2_best_path = Path(args.resume_step2_ckpt)
        history_step2 = None
    else:
        logger.info("Starting Stage 2: Off-ramp training...")
        from train import train_step2_hdc
        history_step2, best_ckpt_path_2 = train_step2_hdc(
            model, train_loader, test_loader, test_loader_ee,
            args, device,
            entropy_threshold=args.entropy_threshold,
            eval_early_exit=args.eval_early_exit,
        )
        step2_best_path = Path(best_ckpt_path_2)

        logger.info(f"Loading best Stage 2 checkpoint: {step2_best_path}")
        model.load_state_dict(torch.load(step2_best_path, map_location=device))

    # -------- Stage 3: Router training --------
    logger.info("Starting Stage 3: Router training...")
    from train import train_step3_hdc
    history_step3, best_ckpt_path_3 = train_step3_hdc(model, train_loader, test_loader, args, device)
    step3_best_path = Path(best_ckpt_path_3)

    logger.info(f"Loading best Stage 3 checkpoint: {step3_best_path}")
    model.load_state_dict(torch.load(step3_best_path, map_location=device))

    # -------- Final evaluation: routing accuracy + per-layer keep_rates --------
    logger.info("Running final routing evaluation (forward_with_routing)...")
    from eval import evaluate_hdc_routing_full, evaluate_with_time
    final_routing_acc, final_eval_keep_rates, final_avg_keep_rate = evaluate_hdc_routing_full(
        model, test_loader, device
    )
    logger.info(f"Final routing accuracy:     {final_routing_acc:.4f}")
    logger.info(f"Final avg keep rate (B):    {final_avg_keep_rate:.4f}")
    if final_eval_keep_rates:
        for i, kr in enumerate(final_eval_keep_rates):
            logger.info(f"  Stage B layer {i}: keep_rate={kr:.4f}")

    # -------- Final HDC inference evaluation (early-exit + token routing) --------
    logger.info("Running final HDC inference evaluation...")
    from eval import evaluate_hdc_inference
    hdc_results = evaluate_hdc_inference(
        model, test_loader_ee, device,
        entropy_threshold=args.entropy_threshold,
    )
    logger.info(f"HDC inference accuracy:     {hdc_results['accuracy']:.4f}")
    logger.info(f"Stage A exit rate:          {hdc_results['stage_a_exit_rate']:.4f}")
    logger.info(f"Avg exit layer (Stage A):   {hdc_results['avg_exit_layer_a']:.2f}")
    logger.info(f"Avg keep rate (Stage B):    {hdc_results['avg_keep_rate_b']:.4f}")
    logger.info(f"Exit histogram:             {hdc_results['exit_histogram']}")

    # -------- Inference latency --------
    logger.info("Measuring inference latency...")
    final_acc_timed, ms_per_sample = evaluate_with_time(model, test_loader, device)
    logger.info(f"Inference speed: {ms_per_sample:.4f} ms/sample (acc={final_acc_timed:.4f})")

    # -------- Model config for report --------
    model_config = {
        "model_name": args.model_name,
        "num_labels": num_labels,
        "n_layers": model.n_layers,
        "hidden_size": model.bert.config.hidden_size,
        "num_attention_heads": model.bert.config.num_attention_heads,
        "intermediate_size": model.bert.config.intermediate_size,
        "split_layer": args.split_layer,
        "stage_a_layers": list(range(args.split_layer)),
        "stage_b_layers": list(range(args.split_layer, model.n_layers)),
        "tau": args.tau,
        "target_keep_ratio": args.target_keep_ratio,
        "lambda_mod": args.lambda_mod,
        "entropy_threshold": args.entropy_threshold,
        "dropout": args.dropout,
        "max_length": args.max_length,
    }

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    offramp_params = sum(p.numel() for p in model.offramp_classifiers.parameters())
    router_params = sum(p.numel() for p in model.routers.parameters())
    final_cls_params = sum(p.numel() for p in model.classifier.parameters())
    bert_params = sum(p.numel() for p in model.bert.parameters())
    parameter_counts = {
        "total": total_params,
        "bert_backbone": bert_params,
        "offramp_classifiers": offramp_params,
        "token_routers": router_params,
        "final_classifier": final_cls_params,
        "new_params_added": offramp_params + router_params,
        "new_params_pct": round((offramp_params + router_params) / total_params * 100, 4),
    }

    # -------- FLOPs estimation data --------
    # 提供给 post-hoc FLOPs 计算所需的所有数值
    flops_data = {
        "n_layers": model.n_layers,
        "split_layer": args.split_layer,
        "n_stage_a_layers": args.split_layer,
        "n_stage_b_layers": model.n_layers - args.split_layer,
        "hidden_size": model.bert.config.hidden_size,
        "num_attention_heads": model.bert.config.num_attention_heads,
        "intermediate_size": model.bert.config.intermediate_size,
        "max_length": args.max_length,
        # Stage A early-exit stats
        "stage_a_exit_rate": hdc_results["stage_a_exit_rate"],
        "avg_exit_layer_a": hdc_results["avg_exit_layer_a"],
        "exit_histogram": hdc_results["exit_histogram"],
        # Stage B routing stats (eval-time, hard mask)
        "stage_b_eval_keep_rates": final_eval_keep_rates,
        "stage_b_avg_keep_rate": final_avg_keep_rate,
    }

    # -------- Assemble full results --------
    # 从 step3 history 提取最佳指标
    best_step3_acc = None
    best_step3_epoch = None
    final_train_keep_rates = None
    final_eval_keep_rates_hist = None
    final_avg_keep_rate_hist = None
    if history_step3 and isinstance(history_step3, dict):
        accs = history_step3.get("hdc_step3_test_acc", [])
        if accs:
            best_step3_acc = max(accs)
            best_step3_epoch = int(accs.index(best_step3_acc)) + 1
        train_krs = history_step3.get("hdc_step3_keep_rates", [])
        if train_krs:
            final_train_keep_rates = train_krs[-1]
        eval_krs = history_step3.get("hdc_step3_eval_keep_rates", [])
        if eval_krs:
            final_eval_keep_rates_hist = eval_krs[-1]
        avg_krs = history_step3.get("hdc_step3_avg_keep_rate", [])
        if avg_krs:
            final_avg_keep_rate_hist = avg_krs[-1]

    results = {
        # ---- experiment config ----
        "args": vars(args),
        "model_config": model_config,
        "parameter_counts": parameter_counts,

        # ---- checkpoints ----
        "step1_best_ckpt": str(step1_best_path),
        "step2_best_ckpt": str(step2_best_path),
        "step3_best_ckpt": str(step3_best_path),

        # ---- training histories ----
        "history_step1": history_step1,
        "history_step2": history_step2,
        "history_step3": history_step3,

        # ---- Stage 3 summary ----
        "step3_best_routing_acc": best_step3_acc,
        "step3_best_epoch": best_step3_epoch,
        "step3_final_train_keep_rates": final_train_keep_rates,
        "step3_final_eval_keep_rates": final_eval_keep_rates_hist,
        "step3_final_avg_keep_rate": final_avg_keep_rate_hist,

        # ---- final evaluation (routing mode, best Step3 ckpt) ----
        "final_routing_acc": final_routing_acc,
        "final_eval_keep_rates": final_eval_keep_rates,
        "final_avg_keep_rate": final_avg_keep_rate,

        # ---- final HDC inference evaluation ----
        "hdc_inference": hdc_results,

        # ---- latency ----
        "inference_ms_per_sample": ms_per_sample,

        # ---- FLOPs estimation data ----
        "flops_estimation_data": flops_data,
    }

    save_results(results, args.output_dir)
    logger.info("HDC-BERT experiment completed successfully!")


if __name__ == "__main__":
    main()

## 实验主线流程说明
'''
1. main_hdcbert.py → 解析参数、设置随机种子
2. data.py → 加载 AG News、tokenization、构建 DataLoader (batch_size=32 + batch_size=1)
3. models/hdc_bert.py → 初始化 HDC-BERT 模型
4. Stage 1: train.py → train_step1_hdc() → 标准 fine-tune BERT (和 baseline 一样)
5. Stage 2: train.py → train_step2_hdc() → 冻住 backbone, 只训练 off-ramp classifiers
6. Stage 3: train.py → train_step3_hdc() → 冻住 backbone + off-ramps, 只训练 token routers
7. eval.py → evaluate_hdc_inference() 最终 HDC 推理评估
8. 保存最佳模型 checkpoint 和实验结果 (JSON)
'''
