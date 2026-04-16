"""
Data module: responsible for loading datasets and applying tokenization.
It is decoupled from specific models and only provides standard DataLoaders.
Supported datasets: ag_news, imdb.
"""
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import sys

def load_ag_news_data(tokenizer_name: str, batch_size: int = 32, max_length: int = 128, num_workers: int = 0):
    """
    Load the AG News dataset and apply tokenization.
    """
    dataset = load_dataset("ag_news")

    # Initialize the tokenizer
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


def load_imdb_data(tokenizer_name: str, batch_size: int = 32, max_length: int = 256, num_workers: int = 0):
    """
    Load the IMDB dataset and apply tokenization.
    This is binary sentiment analysis (0=negative, 1=positive) with longer
    texts, so the default max_length is 256.
    """
    dataset = load_dataset("imdb")

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

    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

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


def load_data(dataset: str, tokenizer_name: str, batch_size: int = 32, max_length: int = 128, num_workers: int = 0):
    """
    Unified data-loading entry point that dispatches to the corresponding
    loader based on the dataset name.
    """
    if dataset == "ag_news":
        return load_ag_news_data(tokenizer_name, batch_size, max_length, num_workers)
    elif dataset == "imdb":
        return load_imdb_data(tokenizer_name, batch_size, max_length, num_workers)
    else:
        raise ValueError(f"Unknown dataset: '{dataset}'. Supported: ag_news, imdb")
