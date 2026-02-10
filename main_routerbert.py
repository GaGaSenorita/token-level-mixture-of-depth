"""
Router-Tuning training entrypoint (Mixture-of-Depth style).
Does not modify existing training scripts.
"""
import argparse
from pathlib import Path
import os
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoModel
from tqdm import tqdm

from utils import set_seed, get_device, ensure_dir, setup_logger, save_results
from data import load_ag_news_data
from models import BERTClassifier
from models.router_tuning_bert import RouterTuningBERTClassifier
from train import train_router_tuning


def load_yaml_config(config_path: str) -> dict:
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


def parse_routed_layers(spec: str, n_layers: int) -> list[int]:
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(n_layers))
    if spec.startswith("last"):
        k = int(spec.replace("last", ""))
        k = max(1, min(k, n_layers))
        return list(range(n_layers - k, n_layers))
    # comma-separated indices
    indices = []
    for part in spec.split(","):
        part = part.strip()
        if part == "":
            continue
        indices.append(int(part))
    return indices


def parse_args():
    parser = argparse.ArgumentParser(description="Router-Tuning for BERT on AGNews")
    parser.add_argument("--config", type=str, default=None, help="Path to yaml config")

    # ---- Experiment ----
    parser.add_argument("--mode", type=str, default="router", choices=["baseline", "router"])
    parser.add_argument("--run_name", type=str, default="router_tuning_agnews")
    parser.add_argument("--model_key", type=str, default="router_tuning")

    # ---- Data ----
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)

    # ---- Model ----
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--dropout", type=float, default=0.1)

    # ---- Training ----
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    # ---- Router ----
    parser.add_argument("--routing_mode", type=str, default="token", choices=["token", "sample"])
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--lambda_mod", type=float, default=1e-3)
    parser.add_argument("--target_keep_ratio", type=float, default=0.7)
    parser.add_argument("--routed_layers", type=str, default="last4")

    # ---- IO ----
    parser.add_argument("--output_root", type=str, default="./outputs")

    args_pre, _ = parser.parse_known_args()
    if args_pre.config:
        cfg = load_yaml_config(args_pre.config)
        known_keys = {a.dest for a in parser._actions}
        cfg = {k: v for k, v in cfg.items() if k in known_keys}
        parser.set_defaults(**cfg)
    return parser.parse_args()


def build_scheduler(optimizer, steps_per_epoch, epochs, warmup_ratio=0.1):
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


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

    logger.info("Loading data...")
    train_loader, test_loader, num_labels = load_ag_news_data(
        tokenizer_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    if args.mode == "baseline":
        logger.info("Running baseline training (BERTClassifier).")
        model = BERTClassifier(
            model_name=args.model_name,
            num_labels=num_labels,
            dropout=args.dropout,
        ).to(device)

        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = build_scheduler(optimizer, len(train_loader), args.epochs, warmup_ratio=0.1)

        from eval import evaluate
        history = {"train_loss": [], "test_acc": []}
        best_acc = 0.0
        best_path = os.path.join(args.output_dir, "best_model.pt")

        for epoch in range(args.epochs):
            logger.info(f"===== Epoch {epoch + 1}/{args.epochs} =====")
            model.train()
            total_loss = 0.0
            loss_fn = torch.nn.CrossEntropyLoss()

            for batch in tqdm(train_loader, desc="Train", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                total_loss += loss.item()

            train_loss = total_loss / max(1, len(train_loader))
            test_acc = evaluate(model, test_loader, device)

            history["train_loss"].append(train_loss)
            history["test_acc"].append(test_acc)

            logger.info(f"Train loss: {train_loss:.4f}")
            logger.info(f"Test acc:    {test_acc:.4f}")

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), best_path)
                logger.info(f"Best model saved: {best_path} (acc={best_acc:.4f})")

        results = {
            "args": vars(args),
            "history": history,
            "best_test_acc": best_acc,
        }
        save_results(results, args.output_dir)
        logger.info("Baseline training completed.")
        return

    # ---- Router mode ----
    logger.info("Running router-tuning training.")
    # Need a temp model to get n_layers for routed_layers parsing
    temp = AutoModel.from_pretrained(args.model_name)
    n_layers = temp.config.num_hidden_layers
    del temp

    routed_layers = parse_routed_layers(args.routed_layers, n_layers)
    logger.info(f"Routed layers: {routed_layers}")

    model = RouterTuningBERTClassifier(
        model_name=args.model_name,
        num_labels=num_labels,
        dropout=args.dropout,
        routing_mode=args.routing_mode,
        tau=args.tau,
        target_keep_ratio=args.target_keep_ratio,
        routed_layers=routed_layers,
    ).to(device)

    history = train_router_tuning(model, train_loader, test_loader, args, device)

    best_acc = max(history["test_acc"]) if history["test_acc"] else 0.0
    results = {
        "args": vars(args),
        "history": history,
        "best_test_acc": best_acc,
    }
    save_results(results, args.output_dir)
    logger.info("Router-tuning training completed.")


if __name__ == "__main__":
    main()
