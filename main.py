"""
主入口：负责调度整个实验流程
职责：参数解析 → 初始化 → 调用训练/评估 → 保存结果
不包含任何模型或训练细节
"""
import argparse
from pathlib import Path

from utils import set_seed, save_results, get_device, ensure_dir, setup_logger
from data import load_ag_news_data
from models import BERTClassifier, BertSampleLevelForSequenceClassification
from train import train
import sys


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="BERT Baseline for AG News")
    
    # 数据参数
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', 
                        help='Pretrained model name')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # 训练参数
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./outputs/bert_baseline',
                        help='Output directory')
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    logger = setup_logger(args.output_dir, name="training")
    logger.info("Starting experiment:")

    device = get_device()
    logger.info(f"Using device: {device}")
    sys.exit("没得跑了")
    
    logger.info("Loading data...")
    train_loader, test_loader, num_labels = load_ag_news_data(
        tokenizer_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
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