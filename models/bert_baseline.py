"""
BERT Baseline 模型
后续实验只需要创建新的 model 文件，保持相同接口即可
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
            若提供 labels: (loss, logits)
            否则: logits
        
        把AG News的文本 → BERT 编码 → 取 CLS → 做分类logits
        """
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ) # [batch_size, seq_len, hidden_size]
        
        # 使用 [CLS] token 的输出
        pooled_output = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size] 我们有B条句子，每条句子背padding/truncation到seq_len长度，取每条句子的第0个token的输出作为句子表示
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_labels] 
        return logits