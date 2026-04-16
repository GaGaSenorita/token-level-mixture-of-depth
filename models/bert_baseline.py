"""
BERT baseline model.
Later experiments only need to create new model files while keeping the same
interface.
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class BERTClassifier(nn.Module):

    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels) # hidden_size = 768 for bert-base
        
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            If labels are provided: (loss, logits)
            Otherwise: logits
        
        AG News text -> BERT encoding -> take CLS -> compute classification logits
        """
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ) # [batch_size, seq_len, hidden_size]
        
        # Use the output of the [CLS] token
        pooled_output = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size] After padding/truncation to seq_len, the first token output is used as the sentence representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_labels] 
        return logits
