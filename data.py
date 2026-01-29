"""
数据模块：负责加载 AG News 并进行 tokenization
与具体模型解耦，只提供标准的 DataLoader
"""
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def load_ag_news_data(tokenizer_name: str, batch_size: int = 32, max_length: int = 128, num_workers: int = 0):
    """
    加载 AG News 数据集并进行tokenization
    """
    dataset = load_dataset("ag_news")
    
    # 初始化 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def tokenize_function(examples):
        """Tokenization 函数"""
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
    
    # 应用 tokenization
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    # 设置格式为 PyTorch tensors
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # 创建 DataLoader
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
    
    print(f"Data loaded: {len(tokenized_datasets['train'])} train, {len(tokenized_datasets['test'])} test")
    print(f"Number of labels: {num_labels}")
    
    return train_loader, test_loader, num_labels