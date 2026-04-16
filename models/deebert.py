'''
1) Reuse the baseline skeleton.
Copy the structure from bert_baseline.py.
Keep self.bert = AutoModel.from_pretrained(model_name).
Add output_hidden_states=True to obtain hidden states from every layer.
Replace self.classifier with self.classifiers = ModuleList([...]).
forward() returns the logits from the last-layer head by default so the
existing train/eval pipeline does not need to change.

2)
forward_all_exits(): returns logits_list for Stage 2 training
forward_early_exit(): used by evaluation or standalone scripts
'''

# models/deebert.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1).clamp_min(1e-12) # [B, C]
    return -(probs * probs.log()).sum(dim=-1)  # [B]


class DeeBERTClassifier(nn.Module):
    """
    DeeBERT: one classifier head per Transformer layer (called off-ramps in paper).
    - Default forward(): return last-exit logits (drop-in replacement of BERTClassifier)
    - forward_all_exits(): return the list of logits for all layers (used in Stage 2 training)
    - forward_early_exit(): perform early exit based on the entropy threshold (evaluation/inference)
    """
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        self.n_layers = self.bert.config.num_hidden_layers
        hidden = self.bert.config.hidden_size

        self.classifiers = nn.ModuleList(
            [nn.Linear(hidden, num_labels) for _ in range(self.n_layers)]
        )

    def freeze_backbone_and_last_head(self): # freeze BERT and last classifier head for stage2 training
        for p in self.bert.parameters():
            p.requires_grad = False
        for p in self.classifiers[-1].parameters():
            p.requires_grad = False

    def forward_all_exits(self, input_ids, attention_mask):
        """
        Return logits for every layer exit:
        logits_list: length = n_layers, each [B, C]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = outputs.hidden_states # tuple of length n_layers+1, each [B, seq_len, H]

        logits_list = []
        for i in range(1, self.n_layers + 1): # hidden_states[0] is the embedding output
            cls = hidden_states[i][:, 0] # [B, H]
            cls = self.dropout(cls) # [B, H]
            logits = self.classifiers[i - 1](cls) # [B, C]
            logits_list.append(logits)
        return logits_list

    def forward(self, input_ids, attention_mask): # Return the logits from the last layer
        """
        Drop-in replacement for your current pipeline:
        return last layer logits (same behavior as BERTClassifier).
        """
        logits_list = self.forward_all_exits(input_ids, attention_mask)
        return logits_list[-1]

    @torch.no_grad()
    def forward_early_exit_batchwise(self, input_ids, attention_mask, entropy_threshold: float = 0.2):
        """
        True layer-by-layer early exit: computation stops at exit layer.
        Returns:
            logits: [B, C] from the exit layer
            exited_layer: int (1..n_layers)
        """
        self.eval()

        # Step 1: embedding layer
        hidden_states = self.bert.embeddings(input_ids=input_ids)

        # Step 2: extended attention mask (same as HuggingFace internal)
        extended_attention_mask = self.bert.get_extended_attention_mask(
            attention_mask, input_ids.shape
        )

        # Step 3: run transformer layers one by one, exit early if condition met
        for i, layer_module in enumerate(self.bert.encoder.layer):
            layer_outputs = layer_module(hidden_states, extended_attention_mask)
            hidden_states = layer_outputs[0]  # [B, seq_len, H]

            cls = self.dropout(hidden_states[:, 0])  # [B, H]
            logits = self.classifiers[i](cls)         # [B, C]
            ent = entropy_from_logits(logits)         # [B]

            if torch.all(ent < entropy_threshold):
                return logits, i + 1  # 1-indexed

        return logits, self.n_layers
