"""
Router-Tuning BERT 模型（Mixture-of-Depth 风格）
只对 Attention 子层做门控，MLP 仍然全量执行。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def ste_binarize(prob: torch.Tensor, tau: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Binarize probabilities with a hard threshold and STE gradient.

    Returns:
        mask: STE mask (float), same shape as prob
        mask_hard: hard mask (0/1) for stats/penalty
    """
    mask_hard = (prob >= tau).to(prob.dtype)
    mask = mask_hard.detach() - prob.detach() + prob
    return mask, mask_hard


class TokenRouter(nn.Module):
    """Token-level router: R_i = sigmoid(W x_i)."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, L, H]
        return torch.sigmoid(self.proj(hidden_states))  # [B, L, 1]


class SampleRouter(nn.Module):
    """Sample-level router: R = sigmoid(W mean_pool(x))."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # hidden_states: [B, L, H]
        if attention_mask is None:
            pooled = hidden_states.mean(dim=1)
        else:
            # attention_mask: [B, L]
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = (hidden_states * mask).sum(dim=1) / denom
        return torch.sigmoid(self.proj(pooled))  # [B, 1]


class NoRouter(nn.Module):
    """Placeholder for non-routed layers."""
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return None


class RouterTuningBERTClassifier(nn.Module):
    """
    BERT classifier with MoD routing on attention outputs.
    forward 返回: logits, router_stats, l_mod
    """
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        routing_mode: str = "token",
        tau: float = 0.5,
        target_keep_ratio: float = 0.7,
        routed_layers: list[int] | None = None,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        self.n_layers = self.bert.config.num_hidden_layers
        self.hidden = self.bert.config.hidden_size

        self.routing_mode = routing_mode
        self.tau = tau
        self.target_keep_ratio = target_keep_ratio

        if routed_layers is None:
            routed_layers = list(range(self.n_layers))
        self.routed_layers = set(routed_layers)

        routers = []
        for i in range(self.n_layers):
            if i in self.routed_layers:
                if routing_mode == "token":
                    routers.append(TokenRouter(self.hidden))
                elif routing_mode == "sample":
                    routers.append(SampleRouter(self.hidden))
                else:
                    raise ValueError(f"Unknown routing_mode: {routing_mode}")
            else:
                routers.append(NoRouter())
        self.routers = nn.ModuleList(routers)

    def _gate_attention(self, layer_module, hidden_states, attention_mask, raw_attention_mask, router):
        """
        gate attention output with STE mask, then run MLP as usual.
        """
        attn_outputs = layer_module.attention.self(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=None,
            output_attentions=False,
        )
        context_layer = attn_outputs[0]

        attn_out = layer_module.attention.output.dense(context_layer)
        attn_out = layer_module.attention.output.dropout(attn_out)

        keep_rate = None
        l_mod = hidden_states.new_tensor(0.0)

        if not isinstance(router, NoRouter):
            if self.routing_mode == "token":
                prob = router(hidden_states)  # [B, L, 1]
                mask, mask_hard = ste_binarize(prob, self.tau)
                attn_out = attn_out * mask

                token_mask = raw_attention_mask.to(attn_out.dtype).unsqueeze(-1)
                m_sum = (mask_hard * token_mask).sum()
                denom = token_mask.sum().clamp_min(1.0)
                keep_rate = (m_sum / denom).item()
                budget = self.target_keep_ratio * denom
                l_mod = F.relu(m_sum - budget)

            elif self.routing_mode == "sample":
                prob_sample = router(hidden_states, raw_attention_mask)  # [B, 1]
                mask_sample, mask_hard_sample = ste_binarize(prob_sample, self.tau)
                mask = mask_sample.unsqueeze(1)  # [B, 1, 1] -> broadcast
                attn_out = attn_out * mask

                m_sum = mask_hard_sample.sum()
                denom = mask_hard_sample.new_tensor(mask_hard_sample.size(0)).clamp_min(1.0)
                keep_rate = (m_sum / denom).item()
                budget = self.target_keep_ratio * denom
                l_mod = F.relu(m_sum - budget)

        attn_out = layer_module.attention.output.LayerNorm(attn_out + hidden_states)
        intermediate_output = layer_module.intermediate(attn_out)
        layer_output = layer_module.output(intermediate_output, attn_out)
        return layer_output, keep_rate, l_mod

    def forward(self, input_ids, attention_mask):
        device = input_ids.device
        input_shape = input_ids.size()

        embedding_output = self.bert.embeddings(
            input_ids=input_ids,
            token_type_ids=None,
        )

        extended_attention_mask = self.bert.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        hidden_states = embedding_output
        keep_rates = [None for _ in range(self.n_layers)]
        l_mod_total = hidden_states.new_tensor(0.0)

        for i, layer_module in enumerate(self.bert.encoder.layer):
            router = self.routers[i]
            hidden_states, keep_rate, l_mod = self._gate_attention(
                layer_module,
                hidden_states,
                extended_attention_mask,
                attention_mask,
                router,
            )
            if keep_rate is not None:
                keep_rates[i] = keep_rate
                l_mod_total = l_mod_total + l_mod

        pooled_output = hidden_states[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        router_stats = {
            "keep_rates": keep_rates,
            "routed_layers": sorted(self.routed_layers),
        }
        return logits, router_stats, l_mod_total
