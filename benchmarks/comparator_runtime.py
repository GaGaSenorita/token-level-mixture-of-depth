"""Explicit standalone comparators and same-weight routing implementations.

These are validation-selected adaptations of this repository's models, not a
claim to reproduce the decoder/sequence-router design of the Router-Tuning paper.
The original model sources and November 8 benchmark remain unchanged.
"""
from __future__ import annotations

import torch

from benchmarks.hdc_runtime import HDCBertRuntime
from models.deebert import entropy_from_logits


class ComparatorRuntime(HDCBertRuntime):
    def __init__(self, *args, backend="query_loop", **kwargs):
        super().__init__(*args, **kwargs)
        if backend not in {"query_loop", "query_batched", "dense_gate", "legacy", "no_router"}:
            raise ValueError(f"Unknown backend {backend}")
        self.backend = backend

    @torch.inference_mode()
    def forward_dynamic(self, input_ids, attention_mask, entropy_threshold,
                        routing=None, collect_stats=False, force_keep_all=False):
        """Backend is fixed at construction; routing is the harness variant ID.

The shared measurement harness passes names such as ``router_batched`` via
``routing``. It does not select this class's execution backend; specs() records
that mapping explicitly. ``no_router`` is an HDC early-exit-only ablation.
"""
        hidden, extended = self._prepare(input_ids, attention_mask)
        b, t = input_ids.shape
        h, f = self.bert.config.hidden_size, self.bert.config.intermediate_size
        c = self.num_labels
        dense_layer = 8*b*t*h*h + 4*b*t*t*h + 4*b*t*h*f
        stats = {}
        total = 0
        if collect_stats:
            stats = {"batch_size": b, "padded_length": t, "backend": self.backend,
                     "exit_layer": self.n_layers, "exited_early": False,
                     "stage_a_layers": 0, "stage_b_layers": [], "final_head": False}
        for index, layer in enumerate(self.bert.encoder.layer):
            if index < self.split_layer:
                hidden = layer(hidden, extended)[0]
                logits = self.offramp_classifiers[index](self.dropout(hidden[:, 0]))
                if collect_stats:
                    total += dense_layer + 2*b*h*c
                    stats["stage_a_layers"] += 1
                entropy = entropy_from_logits(logits)
                exit_now = entropy_threshold >= 0 and bool(torch.all(entropy < entropy_threshold))
                # In standalone DeeBERT, off-ramp 12 IS the frozen dense head.
                # Never compute the identical final classifier a second time.
                if exit_now or index == self.n_layers - 1:
                    if collect_stats:
                        stats.update(exit_layer=index+1, exited_early=index+1 < self.n_layers,
                                     matmul_flops=int(total))
                    return logits, stats
            else:
                if self.backend == "no_router":
                    hidden = layer(hidden, extended)[0]
                    if collect_stats:
                        total += dense_layer
                    continue
                probability = self.routers[index-self.split_layer](hidden)
                keep = (probability.squeeze(-1) >= self.tau) & attention_mask.bool()
                if force_keep_all:
                    keep = attention_mask.bool()
                detail = {}
                if self.backend == "query_loop":
                    output = self._query_attention(layer, hidden, extended, keep)
                elif self.backend == "legacy":
                    output = self._token_attn_skip_inference(
                        layer, hidden, attention_mask, keep.unsqueeze(-1).to(hidden.dtype))
                else:
                    from benchmarks.batched_query import batched_query_attention, dense_gated_attention
                    helper = batched_query_attention if self.backend == "query_batched" else dense_gated_attention
                    output, detail = helper(layer, hidden, extended, keep, collect_stats=collect_stats)
                hidden = self._run_post_attention(layer, hidden, output)
                if collect_stats:
                    counts = keep.sum(dim=1).cpu().tolist()
                    entry = {"layer": index+1, "k_per_sample": counts, **detail}
                    stats["stage_b_layers"].append(entry)
                    total += 2*b*t*h + 4*b*t*h*f  # router and full padded FFN
                    if self.backend in {"query_batched", "dense_gate"}:
                        total += int(detail["attention_matmul_flops"])
                    else:
                        for k in counts:
                            if k:
                                total += (4*(k+t)*h*h + 4*k*t*h) if self.backend == "query_loop" else (8*k*h*h + 4*k*k*h)
        logits = self.classifier(self.dropout(hidden[:, 0]))
        if collect_stats:
            total += 2*b*h*c
            stats.update(final_head=True, matmul_flops=int(total))
        return logits, stats
