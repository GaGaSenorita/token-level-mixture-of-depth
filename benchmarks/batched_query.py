"""Batched inference implementations of the existing hard query gate.

Both paths preserve ALL key/value positions and mask padding keys. They do not
implement the historical K-by-K token pruning operation. Only selected query
outputs survive the gate; residual/LayerNorm/FFN remain the caller's job.

``batched_query`` packs queries to the largest kept count in the batch. It has
one device-to-host Kmax synchronization per layer (rather than one ``nonzero``
per sample), but still pays sorting, gather/scatter and query-padding overhead.
Zero-query samples still execute projections when another sample has queries.
``dense_gate`` runs the full attention then gates outputs, saving NO attention
matrix FLOPs; it is useful for isolating sparse-execution overhead.

These eager implementations support absolute-position encoder BERT in eval
mode. They are NOT training/STE implementations, optimized SDPA, or a promise
of speedup. Collect optional statistics outside timed execution. FLOPs count
matrix multiplies only, with multiply-add = two operations.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


def attention_matmul_flops(
    batch_size: int,
    padded_length: int,
    query_length: int,
    hidden_size: int,
    attention_size: int | None = None,
) -> int:
    """Executed Q/K/V/output projections + QK/AV for a padded batched call.

``query_length=0`` means every sample skips all attention projections. For a
mixed batch the caller must pass Kmax, not the mean or sum of retained counts.
"""
    b, t, k, h = map(int, (batch_size, padded_length, query_length, hidden_size))
    a = h if attention_size is None else int(attention_size)
    if min(b, t, h, a) <= 0 or not 0 <= k <= t:
        raise ValueError("Positive B/T/H/A and 0 <= query_length <= T are required")
    if k == 0:
        return 0
    return 4 * b * (k + t) * h * a + 4 * b * k * t * a


def _validate(layer: nn.Module, hidden, extended_mask, keep) -> None:
    if layer.training:
        raise RuntimeError("Batched query backends are inference-only; call eval()")
    attention = layer.attention.self
    if getattr(attention, "position_embedding_type", "absolute") != "absolute":
        raise ValueError("Only absolute-position BERT attention is supported")
    if getattr(attention, "is_decoder", False):
        raise ValueError("Decoder/cross-attention BERT is unsupported")
    if hidden.ndim != 3 or keep.shape != hidden.shape[:2] or keep.dtype != torch.bool:
        raise ValueError("hidden must be [B,T,H] and keep must be boolean [B,T]")
    if hidden.shape[0] == 0 or hidden.shape[1] == 0:
        raise ValueError("Empty batches or sequences are unsupported")
    if extended_mask.shape != (hidden.shape[0], 1, 1, hidden.shape[1]):
        raise ValueError("The additive padding-key mask must have shape [B,1,1,T]")


def _effective_keep(keep: torch.Tensor, extended_mask: torch.Tensor) -> torch.Tensor:
    # Hugging Face's additive padding mask is zero at valid keys and negative
    # at padding. Mask queries too, matching the existing runtime's contract.
    return keep & extended_mask[:, 0, 0, :].eq(0)


def _stats(layer, hidden, keep, packed_queries, backend) -> dict[str, Any]:
    b, t, h = hidden.shape
    return {
        "backend": backend,
        "batch_size": b,
        "padded_length": t,
        "packed_query_length": int(packed_queries),
        "k_per_sample": keep.sum(dim=1).detach().cpu().tolist(),
        "attention_matmul_flops": attention_matmul_flops(
            b, t, packed_queries, h, layer.attention.self.all_head_size
        ),
    }


def _attention_output(layer, hidden, queries, extended_mask):
    attention = layer.attention.self
    b, k, _ = queries.shape
    heads = attention.num_attention_heads
    head_size = attention.attention_head_size

    def as_heads(x):
        return x.reshape(b, -1, heads, head_size).transpose(1, 2)

    query = as_heads(attention.query(queries))
    key = as_heads(attention.key(hidden))
    value = as_heads(attention.value(hidden))
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_size)
    probabilities = attention.dropout(torch.softmax(scores + extended_mask, dim=-1))
    context = torch.matmul(probabilities, value).transpose(1, 2).contiguous()
    context = context.reshape(b, k, attention.all_head_size)
    return layer.attention.output.dropout(layer.attention.output.dense(context))


@torch.inference_mode()
def batched_query_attention(
    layer: nn.Module,
    hidden: torch.Tensor,
    extended_mask: torch.Tensor,
    keep: torch.Tensor,
    *,
    collect_stats: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return gated attention output [B,T,H] and optional untimed statistics.

Sort keys are unique, so dummy queries never collide with real query scatter
destinations. Dummy output rows are zeroed *after* projection (including bias).
This is semantically equivalent to dense attention followed by a hard query
gate, to floating-point tolerance; reduced-precision rounding can differ.
"""
    _validate(layer, hidden, extended_mask, keep)
    keep = _effective_keep(keep, extended_mask)
    b, t, h = hidden.shape
    kmax = int(keep.sum(dim=1).max().item())
    output = torch.zeros_like(hidden)
    if kmax:
        positions = torch.arange(t, device=hidden.device).expand(b, t)
        order = torch.argsort(positions + (~keep).to(torch.long) * t, dim=1)[:, :kmax]
        indices = order.unsqueeze(-1).expand(b, kmax, h)
        queries = torch.gather(hidden, 1, indices)
        selected_output = _attention_output(layer, hidden, queries, extended_mask)
        real_queries = torch.gather(keep, 1, order).unsqueeze(-1)
        selected_output = selected_output.to(output.dtype).masked_fill(~real_queries, 0)
        output.scatter_(1, indices, selected_output)
    stats = _stats(layer, hidden, keep, kmax, "batched_query") if collect_stats else {}
    return output, stats


@torch.inference_mode()
def dense_gated_attention(
    layer: nn.Module,
    hidden: torch.Tensor,
    extended_mask: torch.Tensor,
    keep: torch.Tensor,
    *,
    collect_stats: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Full batched attention plus query gate; performs full attention FLOPs."""
    _validate(layer, hidden, extended_mask, keep)
    keep = _effective_keep(keep, extended_mask)
    output = _attention_output(layer, hidden, hidden, extended_mask)
    output = output.to(hidden.dtype).masked_fill(~keep.unsqueeze(-1), 0)
    stats = _stats(layer, hidden, keep, hidden.shape[1], "dense_gate") if collect_stats else {}
    return output, stats
