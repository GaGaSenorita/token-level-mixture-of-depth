"""
主入口：负责调度整个实验流程
职责：参数解析 → 初始化 → 调用训练/评估 → 保存结果
不包含任何模型或训练细节
"""
import argparse
from pathlib import Path
import time

from utils import set_seed, save_results, get_device, ensure_dir, setup_logger
from data import load_ag_news_data
from models import BERTClassifier
from train import train
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
    parser = argparse.ArgumentParser(description="Finetune Dense BERT with AGnews")
    parser.add_argument("--config", type=str, default=None, help="Path to yaml config")

    # ---- Experiment parameters ----
    parser.add_argument('--run_name', type=str, default='dense_bert_finetune_agnews')
    parser.add_argument('--model_key', type=str, default='finetuned_dense_bert')

    # ---- Data parameters ----
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length')
    
    # ---- Model parameters ----
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', 
                        help='Pretrained model name')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # ---- Train parameters ----
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # --- IO parameters ----
    parser.add_argument('--output_root', type=str, default='./outputs')

    args_pre, _ = parser.parse_known_args()
    # if config is provided, load it and override defaults
    if args_pre.config:
        cfg = load_yaml_config(args_pre.config)
        known_keys = {a.dest for a in parser._actions}
        # only keep keys that are known to the parser
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
    # logger.info("Final args:")
    # for k, v in vars(args).items():
    #     logger.info(f"{k}: {v}")

    device = get_device()
    logger.info(f"Using device: {device}")
    
    logger.info("Loading data...")
    train_loader, test_loader, num_labels = load_ag_news_data(
        tokenizer_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    logger.info(f"Data loaded. num_labels={num_labels}")

    logger.info("Initializing model...")
    model = BERTClassifier(
        model_name=args.model_name,
        num_labels=num_labels,
        dropout=args.dropout
    )
    model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    logger.info("Starting training...")
    history = train(model, train_loader, test_loader, args, device)
    

    results = {
        'args': vars(args),
        'history': history,
        'final_test_acc': history['test_acc'][-1],
        'best_test_acc': max(history['test_acc'])
    }
    
    save_results(results, args.output_dir)
    logger.info("Experiment completed successfully!")


if __name__ == '__main__':
    main()

## 实验主线流程说明
'''
1. main.py → 解析参数、设置随机种子
2. data.py → 加载 AG News、tokenization、构建 DataLoader
3. models/bert_baseline.py → 初始化 BERT 分类模型
4. train.py → 标准训练循环（AdamW + warmup）
5. eval.py → 每个 epoch 后评估准确率
6. 保存最佳模型 checkpoint 和实验结果（JSON）
'''