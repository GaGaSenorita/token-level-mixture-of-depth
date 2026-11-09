"""Independent CPU checks for full comparator paths and executed matrix FLOPs."""

import unittest
from unittest.mock import patch

import torch
from transformers import BertConfig

from benchmarks.comparator_runtime import ComparatorRuntime
from benchmarks.hdc_runtime import HDCBertRuntime, dense_matmul_flops


class ComparatorRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(1)
        self.config = BertConfig(
            vocab_size=41, hidden_size=16, intermediate_size=32,
            num_attention_heads=4, num_hidden_layers=3,
            max_position_embeddings=32, hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        self.ids = torch.tensor([[2, 5, 7, 9, 11, 0], [2, 4, 6, 0, 0, 0]])
        self.mask = self.ids.ne(0).long()

    def model(self, split=0, backend="query_loop"):
        return ComparatorRuntime(config=self.config, num_labels=3,
                                 dropout=0.0, split_layer=split, backend=backend).eval()

    def test_hdc_wrapper_preserves_legacy_benchmark_logits_and_counts(self):
        # Includes no exit, first off-ramp exit, and legacy K-by-K diagnostic.
        old = HDCBertRuntime(config=self.config, num_labels=3, dropout=0.0, split_layer=1).eval()
        new = self.model(split=1)
        new.load_state_dict(old.state_dict(), strict=True)
        for threshold in (-1.0, float("inf")):
            for old_backend, new_backend in (("query", "query_loop"), ("legacy", "legacy")):
                new.backend = new_backend
                expected, prior = old.forward_dynamic(
                    self.ids, self.mask, threshold, old_backend, collect_stats=True)
                actual, stats = new.forward_dynamic(
                    self.ids, self.mask, threshold, collect_stats=True)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(stats["matmul_flops"], prior["matmul_flops"])
                self.assertEqual(stats["exit_layer"], prior["exit_layer"])
                self.assertEqual(stats["exited_early"], prior["exited_early"])

    def test_twelve_layer_hdc_matches_at_every_first_six_exit(self):
        config = BertConfig.from_dict(self.config.to_dict())
        config.num_hidden_layers = 12
        old = HDCBertRuntime(config=config, num_labels=3, dropout=0.0, split_layer=6).eval()
        new = ComparatorRuntime(config=config, num_labels=3, dropout=0.0,
                                split_layer=6, backend="query_batched").eval()
        new.load_state_dict(old.state_dict(), strict=True)
        for exit_layer in range(1, 7):
            entropies = [torch.tensor([0.0, 2.0]) for _ in range(exit_layer-1)]
            entropies.append(torch.tensor([0.0, 0.0]))
            with patch("benchmarks.hdc_runtime.entropy_from_logits", side_effect=entropies):
                expected, prior = old.forward_dynamic(self.ids, self.mask, 1, collect_stats=True)
            with patch("benchmarks.comparator_runtime.entropy_from_logits", side_effect=entropies):
                actual, stats = new.forward_dynamic(self.ids, self.mask, 1, collect_stats=True)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(stats["exit_layer"], exit_layer)
            self.assertEqual(stats["matmul_flops"], prior["matmul_flops"])

    def test_three_router_backends_agree_for_mixed_zero_and_padded_routes(self):
        model = self.model()
        # Real router projections execute; hooks override only the gates.
        keeps = [torch.tensor([[1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0]]),
                 torch.tensor([[1, 0, 0, 0, 0, 0], [1, 1, 0, 1, 1, 1]]),
                 torch.zeros_like(self.mask)]
        handles = [router.register_forward_hook(
            lambda _m, _a, output, keep=keep: keep.to(output.dtype).unsqueeze(-1))
            for router, keep in zip(model.routers, keeps)]
        try:
            expected = None
            for backend in ("query_loop", "query_batched", "dense_gate"):
                model.backend = backend
                actual, stats = model.forward_dynamic(self.ids, self.mask, -1, collect_stats=True)
                quiet, empty = model.forward_dynamic(self.ids, self.mask, -1)
                self.assertEqual(empty, {})
                torch.testing.assert_close(actual, quiet, rtol=0, atol=0)
                if expected is None:
                    expected = actual
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                self.assertEqual([e["k_per_sample"] for e in stats["stage_b_layers"]],
                                 [[3, 0], [1, 2], [0, 0]])
                if backend == "query_batched":
                    self.assertEqual([e["packed_query_length"] for e in stats["stage_b_layers"]],
                                     [3, 2, 0])
        finally:
            for handle in handles:
                handle.remove()

    def test_independent_profiler_covers_actual_kmax_padded_work(self):
        model = self.model()
        keeps = [torch.tensor([[1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0]]),
                 torch.tensor([[1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]]),
                 torch.zeros_like(self.mask)]
        handles = [router.register_forward_hook(
            lambda _m, _a, output, keep=keep: keep.to(output.dtype).unsqueeze(-1))
            for router, keep in zip(model.routers, keeps)]
        # Independent arithmetic, B2/T6/H16/I32/C3: router+FFN=24960/layer,
        # final head=192. Loop attention=10368+16512+0. Batched attention
        # executes Kmax even on empty rows: 20736+17920+0. Dense=29184*3.
        expected = {"query_loop": 101952, "query_batched": 113728, "dense_gate": 162624}
        try:
            for backend, flops in expected.items():
                model.backend = backend
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                            with_flops=True) as profiler:
                    _, stats = model.forward_dynamic(self.ids, self.mask, -1, collect_stats=True)
                measured = sum(e.flops for e in profiler.key_averages()
                               if e.key in {"aten::mm", "aten::addmm", "aten::bmm"})
                self.assertEqual(stats["matmul_flops"], flops, backend)
                self.assertEqual(measured, flops, backend)
        finally:
            for handle in handles:
                handle.remove()

    def test_deebert_final_exit_is_dense_head_once_and_whole_batch_semantics(self):
        model = self.model(split=3)
        # This mapping is required by the comparator checkpoint contract.
        model.offramp_classifiers[-1].load_state_dict(model.classifier.state_dict())
        final_calls = []
        handle = model.classifier.register_forward_hook(lambda *_a: final_calls.append(1))
        try:
            actual, stats = model.forward_dynamic(self.ids, self.mask, -1, collect_stats=True)
            self.assertEqual(final_calls, [], "Dee path must not run final head twice")
            expected = model.forward_dense(self.ids, self.mask)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            self.assertFalse(stats["exited_early"])
            self.assertEqual(stats["stage_a_layers"], 3)
            # Dense formula includes one head; Dee also executed two extra heads.
            self.assertEqual(stats["matmul_flops"],
                             dense_matmul_flops(2, 6, 16, 32, 3, 3) + 2 * (2*2*16*3))
            with patch("benchmarks.comparator_runtime.entropy_from_logits",
                       return_value=torch.tensor([0.0, 2.0])):
                _, mixed = model.forward_dynamic(self.ids, self.mask, 1, collect_stats=True)
            self.assertFalse(mixed["exited_early"])
            _, early = model.forward_dynamic(self.ids, self.mask, float("inf"), collect_stats=True)
            self.assertEqual(early["exit_layer"], 1)
            self.assertEqual(early["matmul_flops"], dense_matmul_flops(2, 6, 16, 32, 3, 1))
        finally:
            handle.remove()


if __name__ == "__main__":
    unittest.main()
