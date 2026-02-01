"""
数据模块：负责加载 AG News 并进行 tokenization
与具体模型解耦，只提供标准的 DataLoader
"""
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import sys

def load_ag_news_data(tokenizer_name: str, batch_size: int = 32, max_length: int = 128, num_workers: int = 0):
    """
    加载 AG News 数据集并进行tokenization
    """
    dataset = load_dataset("ag_news")

    # 初始化 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
    
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )

    # set format for PyTorch
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    print(tokenized_datasets)
    
    train_loader = DataLoader(
        tokenized_datasets['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        tokenized_datasets['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    num_labels = len(set(dataset['train']['label']))
    
    return train_loader, test_loader, num_labels