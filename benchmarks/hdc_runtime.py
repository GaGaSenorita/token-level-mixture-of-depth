"""Inference implementations and execution-shape matmul accounting for HDC.

``query`` preserves the existing training forward's hard query-gate semantics:
retained queries attend to ALL padded K/V positions (padding keys are masked).
``legacy`` retains the historical compact K-by-K attention for comparison.
Neither runtime computes the unused BERT pooler, but its weights remain present
so historical HDC state dictionaries can be loaded strictly.

FLOPs below count one multiply-add as TWO operations. They count actual matrix
shapes, including padding, routers and classification heads. Embedding lookups,
biases, LayerNorm, activations/softmax, residuals, gather/scatter and control flow
are excluded. Consequently these are matmul FLOPs, not total GPU instructions
or a latency prediction. Collect statistics in a separate untimed traversal.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, BertConfig

from models.deebert import entropy_from_logits
from models.hdc_bert import HDCBERTClassifier
from models.router_tuning_bert import TokenRouter


def dense_matmul_flops(
    batch_size: int,
    padded_length: int,
    hidden_size: int,
    intermediate_size: int,
    num_labels: int,
    n_layers: int = 12,
) -> int:
    """Total batch matmul FLOPs for full dense layers and the final head."""
    b, t, h, i, c = map(int, (
        batch_size, padded_length, hidden_size, intermediate_size, num_labels
    ))
    per_layer = 8 * b * t * h * h + 4 * b * t * t * h + 4 * b * t * h * i
    return int(n_layers) * per_layer + 2 * b * h * c


def matmul_flops_from_stats(
    stats: dict[str, Any],
    hidden_size: int,
    intermediate_size: int,
    num_labels: int,
) -> int:
    """Count executed shapes; do not square or multiply aggregate keep ratios.

For one sample with padded length T and K selected queries, query attention is
4*(K+T)*H^2 + 4*K*T*H. Legacy compact attention is 8*K*H^2 + 4*K^2*H.
Both are zero when K=0 because the implementation skips all attention work.
Every executed Stage-B layer still runs its router and its full padded FFN.
"""
    b, t = int(stats["batch_size"]), int(stats["padded_length"])
    h, i, c = int(hidden_size), int(intermediate_size), int(num_labels)
    a = int(stats["stage_a_layers"])
    layer = 8 * b * t * h * h + 4 * b * t * t * h + 4 * b * t * h * i
    total = a * (layer + 2 * b * h * c)  # one off-ramp per visited A layer
    routing = stats["routing"]
    if routing not in ("query", "legacy"):
        raise ValueError(f"Unknown routing mode: {routing}")
    for entry in stats["stage_b_layers"]:
        counts = entry["k_per_sample"]
        if len(counts) != b:
            raise ValueError("Every Stage-B layer must contain one K per sample")
        total += 2 * b * t * h + 4 * b * t * h * i  # router + full FFN
        for kept in counts:
            k = int(kept)
            if not 0 <= k <= t:
                raise ValueError("K must be between zero and padded length")
            if k == 0:
                continue
            if routing == "query":
                total += 4 * (k + t) * h * h + 4 * k * t * h
            else:
                total += 8 * k * h * h + 4 * k * k * h
    if stats["final_head"]:
        total += 2 * b * h * c
    return int(total)


class HDCBertRuntime(HDCBERTClassifier):
    """State-compatible HDC with an eager, query-sparse inference path.

Pass ``config=BertConfig(...)`` without model_name for random offline tests.
Pass model_name for the normal pretrained loading flow. Training methods are
inherited; callers must explicitly set ``eval()`` for both inference methods.
"""

    def __init__(
        self,
        model_name: str | None = None,
        num_labels: int = 2,
        dropout: float = 0.1,
        split_layer: int = 6,
        tau: float = 0.5,
        target_keep_ratio: float = 0.7,
        *,
        config: BertConfig | None = None,
    ) -> None:
        nn.Module.__init__(self)
        if config is not None and model_name is not None:
            raise ValueError("Pass config OR model_name, not both")
        if config is not None:
            self.bert = AutoModel.from_config(
                copy.deepcopy(config), attn_implementation="eager"
            )
        elif model_name is not None:
            self.bert = AutoModel.from_pretrained(
                model_name, attn_implementation="eager"
            )
        else:
            raise ValueError("A model_name or an offline BertConfig is required")
        if self.bert.config.model_type != "bert":
            raise ValueError("This runtime supports BERT encoders only")
        if getattr(self.bert.config, "position_embedding_type", "absolute") != "absolute":
            raise ValueError("Query-sparse runtime requires absolute positions")
        if getattr(self.bert.config, "is_decoder", False):
            raise ValueError("Decoder/cross-attention BERT is unsupported")
        self.num_labels = int(num_labels)
        self.dropout = nn.Dropout(dropout)
        self.n_layers = self.bert.config.num_hidden_layers
        hidden = self.bert.config.hidden_size
        self.split_layer = int(split_layer)
        if not 0 <= self.split_layer <= self.n_layers:
            raise ValueError("split_layer must lie between 0 and n_layers")
        self.tau = float(tau)
        self.target_keep_ratio = float(target_keep_ratio)
        self.offramp_classifiers = nn.ModuleList(
            [nn.Linear(hidden, self.num_labels) for _ in range(self.split_layer)]
        )
        self.routers = nn.ModuleList(
            [TokenRouter(hidden) for _ in range(self.n_layers - self.split_layer)]
        )
        self.classifier = nn.Linear(hidden, self.num_labels)

    def _prepare(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        if self.training:
            raise RuntimeError("Call model.eval() before runtime inference")
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must share shape [B,T]")
        if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("Empty batches/sequences are unsupported")
        hidden = self.bert.embeddings(input_ids=input_ids, token_type_ids=None)
        extended = self.bert.get_extended_attention_mask(
            attention_mask, input_ids.shape
        )
        return hidden, extended

    @torch.inference_mode()
    def forward_dense(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Full encoder + final head, without the unused BERT pooler."""
        hidden, extended = self._prepare(input_ids, attention_mask)
        hidden = self.bert.encoder(
            hidden, attention_mask=extended, return_dict=False
        )[0]
        return self.classifier(self.dropout(hidden[:, 0]))

    def _query_attention(
        self,
        layer: nn.Module,
        hidden: torch.Tensor,
        extended_mask: torch.Tensor,
        keep: torch.Tensor,
    ) -> torch.Tensor:
        """Compute only selected query rows against FULL padded K/V length.

The zero-query case skips Q/K/V and output projection entirely. In particular,
do not use the compact-subsequence helper here: it also removes K/V tokens.
"""
        attention = layer.attention.self
        output = torch.zeros_like(hidden)
        heads = attention.num_attention_heads
        head_size = attention.attention_head_size
        for b in range(hidden.shape[0]):
            idx = torch.nonzero(keep[b], as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            full = hidden[b:b + 1]
            selected = full[:, idx, :]

            def as_heads(x):
                return x.reshape(1, -1, heads, head_size).transpose(1, 2)

            query = as_heads(attention.query(selected))
            key = as_heads(attention.key(full))
            value = as_heads(attention.value(full))
            scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_size)
            probs = torch.softmax(scores + extended_mask[b:b + 1], dim=-1)
            probs = attention.dropout(probs)
            context = torch.matmul(probs, value).transpose(1, 2).contiguous()
            context = context.reshape(1, idx.numel(), attention.all_head_size)
            selected_output = layer.attention.output.dropout(
                layer.attention.output.dense(context)
            )
            # Under autocast, hidden/residual can be FP32 while projections are
            # BF16. Advanced-index assignment requires identical dtypes; this
            # cast also matches dense-gate multiplication followed by residual.
            output[b, idx, :] = selected_output[0].to(output.dtype)
        return output

    def _token_attn_skip_inference(self, layer_module, hidden_states, attention_mask, mask_hard):
        """Legacy K-by-K semantics with an autocast-safe scatter destination."""
        keep = (mask_hard.squeeze(-1) > 0) & attention_mask.bool()
        output = torch.zeros_like(hidden_states)
        for b in range(hidden_states.shape[0]):
            idx = torch.nonzero(keep[b], as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            selected = hidden_states[b:b + 1, idx, :]
            sub_mask = attention_mask.new_ones((1, idx.numel()))
            extended = self.bert.get_extended_attention_mask(sub_mask, sub_mask.shape)
            selected_output = self._run_attention(layer_module, selected, extended)
            output[b, idx, :] = selected_output[0].to(output.dtype)
        return output

    @torch.inference_mode()
    def forward_dynamic(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        entropy_threshold: float,
        routing: str = "query",
        collect_stats: bool = False,
        force_keep_all: bool = False,
    ):
        """Return (logits, stats), preserving historical whole-batch early exit.

``entropy_threshold=-1`` disables entropy-based exits. ``force_keep_all`` is
a diagnostic override for valid tokens; routers still execute and are counted.
When collect_stats=False, stats is empty and no keep-count .item/.cpu is added.
Early-exit decisions and dynamic nonzero indices still inherently synchronize.
"""
        if routing not in ("query", "legacy"):
            raise ValueError("routing must be 'query' or 'legacy'")
        hidden, extended = self._prepare(input_ids, attention_mask)
        stats: dict[str, Any] = {}
        if collect_stats:
            stats = {
                "batch_size": input_ids.shape[0],
                "padded_length": input_ids.shape[1],
                "routing": routing,
                "exit_layer": self.n_layers,
                "exited_early": False,
                "stage_a_layers": 0,
                "stage_b_layers": [],
                "final_head": True,
            }
        for index, layer in enumerate(self.bert.encoder.layer):
            if index < self.split_layer:
                hidden = layer(hidden, extended)[0]
                logits = self.offramp_classifiers[index](self.dropout(hidden[:, 0]))
                if collect_stats:
                    stats["stage_a_layers"] += 1
                # A negative threshold disables exits but still computes the
                # heads/entropy, keeping the actual runtime path explicit.
                entropy = entropy_from_logits(logits)
                if entropy_threshold >= 0 and bool(torch.all(entropy < entropy_threshold)):
                    if collect_stats:
                        stats.update(exit_layer=index + 1, exited_early=True, final_head=False)
                        self._finish_stats(stats)
                    return logits, stats
            else:
                probability = self.routers[index - self.split_layer](hidden)
                keep = (probability.squeeze(-1) >= self.tau) & attention_mask.bool()
                if force_keep_all:
                    keep = attention_mask.bool()
                if collect_stats:
                    stats["stage_b_layers"].append({
                        "layer": index + 1,
                        "k_per_sample": keep.sum(dim=1).detach().cpu().tolist(),
                    })
                if routing == "query":
                    attn_output = self._query_attention(layer, hidden, extended, keep)
                else:
                    attn_output = self._token_attn_skip_inference(
                        layer, hidden, attention_mask, keep.unsqueeze(-1).to(hidden.dtype)
                    )
                hidden = self._run_post_attention(layer, hidden, attn_output)
        logits = self.classifier(self.dropout(hidden[:, 0]))
        if collect_stats:
            self._finish_stats(stats)
        return logits, stats

    def _finish_stats(self, stats: dict[str, Any]) -> None:
        stats["matmul_flops"] = matmul_flops_from_stats(
            stats, self.bert.config.hidden_size,
            self.bert.config.intermediate_size, self.num_labels,
        )


# An explicit alias is convenient for callers using the benchmark terminology.
HDCBERTBenchmark = HDCBertRuntime
