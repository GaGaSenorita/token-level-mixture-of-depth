"""Offline CPU tests for runtime semantics and executed-shape FLOPs."""

import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import AutoModel, BertConfig

from benchmarks.hdc_runtime import (
    HDCBertRuntime, dense_matmul_flops, matmul_flops_from_stats,
)
from models.hdc_bert import HDCBERTClassifier


class FixedRouter(nn.Module):
    def __init__(self, keep):
        super().__init__()
        self.register_buffer("keep", keep.float())

    def forward(self, hidden):
        return self.keep.to(hidden.dtype).unsqueeze(-1)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.config = BertConfig(
            vocab_size=41, hidden_size=16, intermediate_size=32,
            num_attention_heads=4, num_hidden_layers=3,
            max_position_embeddings=32, hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        self.model = HDCBertRuntime(
            config=self.config, num_labels=3, dropout=0.0, split_layer=1
        ).eval()
        self.ids = torch.tensor([[2, 5, 7, 9, 11, 0], [2, 4, 6, 0, 0, 0]])
        self.mask = self.ids.ne(0).long()

    def dense_gated_reference(self, model):
        hidden, extended = model._prepare(self.ids, self.mask)
        for index, layer in enumerate(model.bert.encoder.layer):
            if index < model.split_layer:
                hidden = layer(hidden, extended)[0]
            else:
                prob = model.routers[index - model.split_layer](hidden)
                keep = (prob >= model.tau) & self.mask.bool().unsqueeze(-1)
                attn = model._run_attention(layer, hidden, extended) * keep
                hidden = model._run_post_attention(layer, hidden, attn)
        return model.classifier(hidden[:, 0])

    def test_dense_matches_original_and_strict_state_load(self):
        # Patch only pretrained loading; this test never accesses the network.
        with patch("models.hdc_bert.AutoModel.from_pretrained", side_effect=lambda *_a, **_k:
                   AutoModel.from_config(self.config, attn_implementation="eager")):
            old = HDCBERTClassifier("offline", 3, dropout=0.0, split_layer=1).eval()
        old.load_state_dict(self.model.state_dict(), strict=True)
        self.model.load_state_dict(old.state_dict(), strict=True)
        with torch.inference_mode():
            torch.testing.assert_close(
                self.model.forward_dense(self.ids, self.mask), old(self.ids, self.mask),
                rtol=1e-5, atol=1e-6,
            )

    def test_all_kept_matches_dense_with_and_without_padding(self):
        for mask in (self.mask, torch.ones_like(self.mask)):
            dense = self.model.forward_dense(self.ids, mask)
            for routing in ("query", "legacy"):
                actual, stats = self.model.forward_dynamic(
                    self.ids, mask, -1, routing, collect_stats=True, force_keep_all=True
                )
                torch.testing.assert_close(actual, dense, rtol=1e-5, atol=1e-6)
                self.assertEqual(stats["stage_b_layers"][0]["k_per_sample"],
                                 mask.sum(1).tolist())

    def test_query_matches_dense_gate_half_none_and_mixed(self):
        cases = [
            torch.tensor([[1, 0, 1, 0, 1, 1], [1, 0, 1, 1, 1, 1]]),
            torch.zeros_like(self.mask),
            torch.tensor([[0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]]),
        ]
        for keep in cases:
            self.model.routers = nn.ModuleList([FixedRouter(keep), FixedRouter(keep.flip(1))])
            with torch.inference_mode():
                expected = self.dense_gated_reference(self.model)
            actual, stats = self.model.forward_dynamic(self.ids, self.mask, -1, collect_stats=True)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            self.assertTrue(torch.isfinite(actual).all())
            self.assertEqual(stats["stage_b_layers"][0]["k_per_sample"],
                             (keep.bool() & self.mask.bool()).sum(1).tolist())

    def test_collect_stats_does_not_change_predictions(self):
        a, empty = self.model.forward_dynamic(self.ids, self.mask, -1)
        b, full = self.model.forward_dynamic(self.ids, self.mask, -1, collect_stats=True)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(empty, {})
        self.assertEqual(full["exit_layer"], 3)
        self.assertFalse(full["exited_early"])

    def test_batch_early_exit_and_head_accounting(self):
        # Infinite threshold makes the whole batch exit after the first layer.
        logits, stats = self.model.forward_dynamic(
            self.ids, self.mask, float("inf"), collect_stats=True
        )
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(stats["exit_layer"], 1)
        self.assertTrue(stats["exited_early"])
        self.assertEqual(stats["stage_b_layers"], [])
        self.assertFalse(stats["final_head"])
        self.assertEqual(stats["matmul_flops"], dense_matmul_flops(2, 6, 16, 32, 3, 1))
        with patch("benchmarks.hdc_runtime.entropy_from_logits", return_value=torch.tensor([0.0, 2.0])):
            _, grouped = self.model.forward_dynamic(self.ids, self.mask, 1.0, collect_stats=True)
        self.assertFalse(grouped["exited_early"])
        self.assertEqual(len(grouped["stage_b_layers"]), 2)

    def test_flops_zero_queries_counts_full_padded_ffn(self):
        self.model.routers = nn.ModuleList([FixedRouter(torch.zeros_like(self.mask)) for _ in range(2)])
        for routing in ("query", "legacy"):
            _, stats = self.model.forward_dynamic(self.ids, self.mask, -1, routing, collect_stats=True)
            first = dense_matmul_flops(2, 6, 16, 32, 3, 1)
            router_ffn = 2 * 2 * 6 * 16 + 4 * 2 * 6 * 16 * 32
            final_head = 2 * 2 * 16 * 3
            self.assertEqual(stats["matmul_flops"], first + 2 * router_ffn + final_head)

    def test_flops_uses_individual_k_squared(self):
        stats = dict(batch_size=2, padded_length=6, routing="legacy", stage_a_layers=0,
                     stage_b_layers=[dict(layer=1, k_per_sample=[1, 3])], final_head=True)
        expected = (2 * 2 * 6 * 16 + 4 * 2 * 6 * 16 * 32 +
                    8 * 4 * 16**2 + 4 * (1**2 + 3**2) * 16 + 2 * 2 * 16 * 3)
        self.assertEqual(matmul_flops_from_stats(stats, 16, 32, 3), expected)
        stats["routing"] = "query"
        expected_query = (2 * 2 * 6 * 16 + 4 * 2 * 6 * 16 * 32 +
                          4 * (1 + 6 + 3 + 6) * 16**2 +
                          4 * (1 + 3) * 6 * 16 + 2 * 2 * 16 * 3)
        self.assertEqual(matmul_flops_from_stats(stats, 16, 32, 3), expected_query)

    def test_runtime_requires_eval(self):
        self.model.train()
        with self.assertRaises(RuntimeError):
            self.model.forward_dynamic(self.ids, self.mask, -1)

    def test_cpu_bfloat16_autocast_scatter(self):
        keep = torch.tensor([[1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 0, 0]])
        self.model.routers = nn.ModuleList([FixedRouter(keep), FixedRouter(keep)])
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            expected = self.dense_gated_reference(self.model)
            actual, _ = self.model.forward_dynamic(self.ids, self.mask, -1)
            legacy, _ = self.model.forward_dynamic(self.ids, self.mask, -1, routing="legacy")
        self.assertTrue(torch.isfinite(legacy).all())
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.003)

    def test_stage3_task_loss_reaches_routers_with_frozen_backbone(self):
        self.model.freeze_for_stage3()
        self.model.train()
        self.model.bert.eval()
        self.model.dropout.eval()
        logits, _, _ = self.model.forward_with_routing(self.ids, self.mask)
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        self.assertTrue(all(p.grad is None for p in self.model.bert.parameters()))
        for router in self.model.routers:
            self.assertIsNotNone(router.proj.weight.grad)
            self.assertGreater(router.proj.weight.grad.abs().sum().item(), 0.0)

    def test_cpu_profiler_independently_matches_matmul_counts(self):
        # Execute the REAL linear routers, then override only their probabilities
        # using hooks so route shapes are known. Replacing routers by constants
        # would incorrectly omit their actual matmul cost from this crosscheck.
        keeps = [
            torch.tensor([[1, 0, 1, 0, 1, 1], [1, 0, 0, 1, 1, 1]]),
            torch.tensor([[1, 1, 0, 0, 0, 1], [1, 1, 0, 1, 1, 1]]),
        ]
        handles = [
            router.register_forward_hook(
                lambda _m, _a, output, keep=keep: keep.to(output.dtype).unsqueeze(-1)
            ) for router, keep in zip(self.model.routers, keeps)
        ]
        # B=2,T=6,H=16,I=32,C=3,N=3,A=1. The valid masks turn
        # the two Stage-B selections into K=[3,1] and K=[2,2].
        # Independently expanded 2-op/MAC totals, including routers and heads:
        expected = {"dense": 161472, "query": 139904, "legacy": 121600}
        try:
            for mode in expected:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU],
                    with_flops=True, profile_memory=False,
                ) as profile:
                    if mode == "dense":
                        self.model.forward_dense(self.ids, self.mask)
                        calculated = dense_matmul_flops(2, 6, 16, 32, 3, 3)
                    else:
                        _, stats = self.model.forward_dynamic(
                            self.ids, self.mask, -1, mode, collect_stats=True
                        )
                        calculated = stats["matmul_flops"]
                events = profile.key_averages()
                # Do not include profiler-counted aten::add/mul: this protocol
                # explicitly reports matmul-only FLOPs, not elementwise work.
                measured = sum(e.flops for e in events
                               if e.key in {"aten::mm", "aten::addmm", "aten::bmm"})
                self.assertGreater(measured, 0, "Profiler did not cover matmul operators")
                self.assertEqual(calculated, expected[mode], mode)
                self.assertEqual(measured, expected[mode], mode)
        finally:
            for handle in handles:
                handle.remove()


if __name__ == "__main__":
    unittest.main()
