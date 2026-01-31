from datasets import load_dataset

def load_ag_news(split: str, seed: int = 10):
    """
    Load AG News dataset.

    split: 'train' | 'valid' | 'test'
    return: texts (List[str]), labels (List[int])
    """
    dataset = load_dataset("ag_news")

    if split == "train":
        data = dataset["train"]
      
    elif split == "valid":
        split_data = dataset["train"].train_test_split(test_size=0.1, seed=seed)
        data = split_data["test"]
      
    elif split == "test":
        data = dataset["test"]
      
    else:
        raise ValueError(f"Unknown split: {split}")

    texts = data["text"]
    labels = data["label"]

    return texts, labels
