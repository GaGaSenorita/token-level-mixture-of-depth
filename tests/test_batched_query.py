"""CPU correctness and independent matrix-FLOPs checks for packed queries."""

import unittest
from unittest.mock import patch

import torch
from transformers import BertConfig

from benchmarks.batched_query import (
    attention_matmul_flops, batched_query_attention, dense_gated_attention,
)
from benchmarks.hdc_runtime import HDCBertRuntime


class BatchedQueryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(142)
        torch.set_num_threads(1)
        self.model = HDCBertRuntime(config=BertConfig(
            vocab_size=41, hidden_size=16, intermediate_size=32,
            num_attention_heads=4, num_hidden_layers=3,
            max_position_embeddings=32, hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
        ), num_labels=3, dropout=0, split_layer=1).eval()
        self.ids = torch.tensor([[2, 5, 7, 9, 11, 0], [2, 4, 6, 0, 0, 0],
                                 [2, 5, 6, 7, 8, 9]])
        self.mask = self.ids.ne(0).long()
        self.hidden, self.extended = self.model._prepare(self.ids, self.mask)
        self.layer = self.model.bert.encoder.layer[1]

    def cases(self):
        return [
            torch.ones_like(self.mask).bool(),
            torch.zeros_like(self.mask).bool(),
            torch.tensor([[0, 0, 0, 0, 0, 0], [1, 0, 1, 1, 1, 1],
                          [0, 1, 0, 1, 0, 1]]).bool(),
            torch.tensor([[1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0]]).bool(),
        ]

    def test_all_none_and_mixed_with_padding_match_loop_and_dense(self):
        for keep in self.cases():
            valid = keep & self.mask.bool()
            with torch.inference_mode():
                expected = self.model._query_attention(self.layer, self.hidden, self.extended, valid)
                reference = self.model._run_attention(self.layer, self.hidden, self.extended) * valid.unsqueeze(-1)
            for backend in (batched_query_attention, dense_gated_attention):
                actual, stats = backend(self.layer, self.hidden, self.extended, keep, collect_stats=True)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
                torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-7)
                self.assertEqual(actual[~valid].abs().sum().item(), 0)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertEqual(stats["k_per_sample"], valid.sum(1).tolist())

    def test_single_sample_and_length_one(self):
        for length in (1, 6):
            hidden = self.hidden[:1, :length]
            extended = self.extended[:1, :, :, :length]
            for enabled in (False, True):
                keep = torch.full((1, length), enabled, dtype=torch.bool)
                effective = keep & extended[:, 0, 0].eq(0)
                expected = self.model._query_attention(self.layer, hidden, extended, effective)
                actual, _ = batched_query_attention(self.layer, hidden, extended, keep)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)

    def test_no_statistics_does_not_change_output(self):
        keep = self.cases()[2]
        for backend in (batched_query_attention, dense_gated_attention):
            actual, empty = backend(self.layer, self.hidden, self.extended, keep)
            collected, stats = backend(self.layer, self.hidden, self.extended, keep, collect_stats=True)
            torch.testing.assert_close(actual, collected, rtol=0, atol=0)
            self.assertEqual(empty, {})
            self.assertIn("attention_matmul_flops", stats)

    def test_zero_query_fast_path_skips_projections(self):
        with patch.object(self.layer.attention.self.query, "forward", side_effect=AssertionError("query executed")):
            actual, stats = batched_query_attention(
                self.layer, self.hidden, self.extended, self.cases()[1], collect_stats=True
            )
        self.assertEqual(actual.abs().sum().item(), 0)
        self.assertEqual(stats["attention_matmul_flops"], 0)

    def test_dense_gate_still_counts_full_flops_when_no_query_is_kept(self):
        _, stats = dense_gated_attention(
            self.layer, self.hidden, self.extended, self.cases()[1], collect_stats=True
        )
        self.assertEqual(stats["packed_query_length"], 6)
        self.assertEqual(stats["attention_matmul_flops"], 8 * 3 * 6 * 16**2 + 4 * 3 * 6**2 * 16)

    def test_independent_profiler_counts_include_dummy_queries_and_zero_samples(self):
        keep = self.cases()[2]  # K=[0,2,3]; packed kernel executes B=3,Kmax=3
        for backend in (batched_query_attention, dense_gated_attention):
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True
            ) as profile:
                _, stats = backend(self.layer, self.hidden, self.extended, keep, collect_stats=True)
            actual = sum(event.flops for event in profile.key_averages()
                         if event.key in {"aten::mm", "aten::addmm", "aten::bmm"})
            k = 3 if backend is batched_query_attention else 6
            expected = 4 * 3 * (k + 6) * 16**2 + 4 * 3 * k * 6 * 16
            self.assertEqual(actual, expected)
            self.assertEqual(stats["attention_matmul_flops"], expected)

    def test_bfloat16_autocast_scatter_keeps_residual_dtype(self):
        keep = self.cases()[2]
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            expected = self.model._query_attention(self.layer, self.hidden, self.extended, keep & self.mask.bool())
            for backend in (batched_query_attention, dense_gated_attention):
                actual, _ = backend(self.layer, self.hidden, self.extended, keep)
                self.assertEqual(actual.dtype, self.hidden.dtype)
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.0002)

    def test_end_to_end_logits_and_gates_match_looped_runtime(self):
        for router in self.model.routers:
            torch.nn.init.normal_(router.proj.weight, std=0.2)
        expected, _ = self.model.forward_dynamic(self.ids, self.mask, -1)
        for backend in (batched_query_attention, dense_gated_attention):
            with patch.object(self.model, "_query_attention", side_effect=lambda layer, hidden, ext, keep:
                              backend(layer, hidden, ext, keep)[0]):
                actual, _ = self.model.forward_dynamic(self.ids, self.mask, -1)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_reject_unsupported_attention_or_training(self):
        self.layer.train()
        with self.assertRaises(RuntimeError):
            batched_query_attention(self.layer, self.hidden, self.extended, self.cases()[0])
        self.layer.eval()
        self.layer.attention.self.position_embedding_type = "relative_key"
        with self.assertRaises(ValueError):
            batched_query_attention(self.layer, self.hidden, self.extended, self.cases()[0])

    def test_flop_helper_validates_shapes(self):
        with self.assertRaises(ValueError):
            attention_matmul_flops(3, 6, 7, 16)


if __name__ == "__main__":
    unittest.main()
