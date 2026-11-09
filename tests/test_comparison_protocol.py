"""Offline orchestration checks; no training, downloads, or GPU operations."""

from contextlib import ExitStack
import copy
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import BertConfig

from benchmarks import compare_baselines as comparison
from benchmarks.comparator_runtime import ComparatorRuntime
from benchmarks.hdc_runtime import dense_matmul_flops
from benchmarks.measure_latency import quality_pass as actual_quality_pass
from benchmarks.measure_latency import read_json, sample_indices, write_json


class ComparisonProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source, self.out = self.root / "source", self.root / "out"
        self.source.mkdir()
        self.out.mkdir()
        write_json(self.source / "selection.json", {"entropy_threshold": 0.3})
        write_json(self.out / "deebert_selection.json", {"entropy_threshold": 0.2})
        self.config = {
            "experiment_id": "offline_protocol_test", "seed": 42,
            "benchmark_batches": [1, 8], "benchmark_samples": 16,
            "benchmark_sampling_seed": 982, "benchmark_repeats": 3,
            "warmup_batches": 2, "benchmark_min_calls": 5,
        }
        self.data = {"test": {"labels": list(range(32))}}
        self.quality_calls, self.latency_calls = [], []
        self.file_revision = "stable"

    def tearDown(self):
        self.temp.cleanup()

    def fake_quality(self, torch, model, data, name, size, threshold, device, output):
        self.quality_calls.append((name, size, threshold))
        return {"variant": name, "batch_size": size}

    def fake_latency(self, torch, model, data, name, size, threshold, device, output,
                     indices, repeat, warmup, calls):
        self.latency_calls.append((name, size, threshold, list(indices), repeat, warmup, calls))
        return {"variant": name, "batch_size": size, "repeat": repeat}

    def mocked_execution(self):
        stack = ExitStack()
        for target, kwargs in (
            ("make_model", {"side_effect": lambda *_: object()}),
            ("shared_weight_identity", {"return_value": {"sha256": "same", "tensor_count": 201}}),
            ("file_hash", {"side_effect": lambda path: self.file_revision + ":" + str(path)}),
            ("quality_pass", {"side_effect": self.fake_quality}),
            ("latency_pass", {"side_effect": self.fake_latency}),
            ("aggregate_results", {"return_value": []}),
        ):
            stack.enter_context(patch.object(comparison, target, **kwargs))
        stack.enter_context(patch.object(comparison.torch.cuda, "get_device_name", return_value="offline"))
        stack.enter_context(patch.object(comparison.torch.cuda, "empty_cache"))
        stack.enter_context(patch.object(comparison, "log"))
        return stack

    def run_measurements(self, diagnostic=False):
        return comparison.run_measurements(
            self.config, {}, self.source, self.out, self.data, diagnostic=diagnostic
        )

    def test_primary_model_calls_share_exact_inputs_and_repeat_protocol(self):
        with self.mocked_execution():
            self.run_measurements()
        names = {"dense", "deebert", "router_loop", "router_batched", "hdc_loop", "hdc_batched", "hdc_exitonly"}
        expected_indices = sample_indices(32, 16, 982)
        self.assertEqual(len(self.quality_calls), 7 * 2)
        self.assertEqual(len(self.latency_calls), 7 * 2 * 3)
        self.assertEqual({tuple(x[3]) for x in self.latency_calls}, {tuple(expected_indices)})
        for size in self.config["benchmark_batches"]:
            for repeat in range(3):
                matching = [x for x in self.latency_calls if x[1] == size and x[4] == repeat]
                self.assertEqual({x[0] for x in matching}, names)
                self.assertEqual(len(matching), 7)
        for name, _, threshold, _, _, warmup, calls in self.latency_calls:
            self.assertEqual(threshold, 0.3 if name.startswith("hdc") else 0.2 if name == "deebert" else -1)
            self.assertEqual((warmup, calls), (2, 5))
        manifest = read_json(self.out / "benchmark" / "manifest.json")
        self.assertEqual(manifest["sample_indices"], expected_indices)
        self.assertFalse(manifest["diagnostic"])

    def test_changed_configuration_rejects_resume_before_new_calls(self):
        with self.mocked_execution():
            self.run_measurements()
            before = len(self.latency_calls)
            self.config["benchmark_min_calls"] += 1
            with self.assertRaisesRegex(RuntimeError, "protocol"):
                self.run_measurements()
            self.assertEqual(len(self.latency_calls), before)

    def test_changed_checkpoint_or_source_hash_rejects_resume(self):
        with self.mocked_execution():
            self.run_measurements()
            self.file_revision = "changed"
            with self.assertRaisesRegex(RuntimeError, "protocol"):
                self.run_measurements()

    def test_changed_threshold_rejects_resume(self):
        with self.mocked_execution():
            self.run_measurements()
            write_json(self.out / "deebert_selection.json", {"entropy_threshold": 0.4})
            with self.assertRaisesRegex(RuntimeError, "protocol"):
                self.run_measurements()

    def test_diagnostics_separate_same_weight_backends_from_primary_output(self):
        with self.mocked_execution():
            self.run_measurements(diagnostic=True)
        manifest = read_json(self.out / "diagnostic_benchmark" / "manifest.json")
        self.assertTrue(manifest["diagnostic"])
        self.assertEqual(manifest["batch_sizes"], [1, 32])
        self.assertEqual(set(manifest["models"]),
                         {"dense", "router_loop", "router_legacy", "router_densegate", "router_batched"})
        self.assertFalse((self.out / "benchmark").exists())
        configured = comparison.specs(self.source, self.out)
        for name in ("router_loop", "router_batched", "router_legacy", "router_densegate"):
            self.assertEqual(configured[name][0], self.out / "router_best.pt")
            self.assertEqual(configured[name][1], 0)

    def test_stale_calibration_rejected_before_loading_a_model_or_touching_gpu(self):
        with patch.object(comparison, "calibration_identity", return_value="current"), \
             patch.object(comparison, "make_model", side_effect=AssertionError("model load must not happen")):
            with self.assertRaisesRegex(RuntimeError, "calibration identity"):
                comparison.select_deebert(self.config, {}, self.source, self.out, {})
            matching = {"entropy_threshold": 0.2, "calibration_protocol_sha256": "current"}
            write_json(self.out / "deebert_selection.json", matching)
            self.assertEqual(comparison.select_deebert(self.config, {}, self.source, self.out, {}), matching)

    def test_calibration_hash_covers_config_checkpoint_data_and_code(self):
        revisions = {}
        def controlled_hash(path):
            return revisions.get(str(path), "stable:" + str(path))
        with patch.object(comparison, "file_hash", side_effect=controlled_hash):
            original = comparison.calibration_identity(self.config, self.source, self.out)
            self.assertEqual(original, comparison.calibration_identity(self.config, self.source, self.out))
            for path in (self.out / "deebert_best.pt", self.source / "data.pt",
                         comparison.ROOT / "models/deebert.py"):
                revisions[str(path)] = "changed"
                self.assertNotEqual(original, comparison.calibration_identity(self.config, self.source, self.out))
                revisions.clear()
            changed_config = {**self.config, "maximum_validation_accuracy_drop": 0.02}
            self.assertNotEqual(original, comparison.calibration_identity(changed_config, self.source, self.out))

    def test_model_creation_rejects_a_different_deebert_final_exit(self):
        classifier = torch.nn.Linear(3, 2)
        head = copy.deepcopy(classifier)
        model = SimpleNamespace(classifier=classifier, offramp_classifiers=[head])
        config = {"num_labels": 2, "dropout": 0, "tau": 0.5, "router_target_keep_ratio": 0.7}
        with patch.object(comparison, "ComparatorRuntime", return_value=model), \
             patch.object(comparison, "load_weights"):
            actual = comparison.make_model("deebert", config, {"model_name": "offline"}, self.source, self.out)
            self.assertIs(actual, model)
            with torch.no_grad():
                head.bias[0] += 1
            with self.assertRaisesRegex(RuntimeError, "final off-ramp"):
                comparison.make_model("deebert", config, {"model_name": "offline"}, self.source, self.out)

    def _calibrate_tiny_cpu_model(self, corrupt_record=False):
        (self.out / "deebert_selection.json").unlink()
        torch.manual_seed(84)
        torch.set_num_threads(1)
        model = ComparatorRuntime(config=BertConfig(
            vocab_size=21, hidden_size=8, intermediate_size=16,
            num_attention_heads=2, num_hidden_layers=3, max_position_embeddings=16,
            hidden_dropout_prob=0, attention_probs_dropout_prob=0,
        ), num_labels=2, dropout=0, split_layer=3).eval()
        model.offramp_classifiers[-1].load_state_dict(model.classifier.state_dict())
        ids = torch.tensor([[2, 3, 4, 0], [2, 4, 5, 6], [2, 6, 0, 0]])
        mask = ids.ne(0).long()
        labels = model.forward_dynamic(ids, mask, -1)[0].argmax(-1)
        config = {**self.config, "max_length": 4, "threshold_candidates": [0.0],
                  "maximum_validation_accuracy_drop": 0.01}
        data = {"validation": {"input_ids": ids, "attention_mask": mask, "labels": labels}}
        def run_cpu_quality(torch_module, actual_model, dataset, name, size, threshold, device, output):
            result = actual_quality_pass(torch_module, actual_model, dataset, name, size, threshold,
                                         torch.device("cpu"), output)
            if corrupt_record:
                path = output / result["predictions_file"]
                with path.open(newline="", encoding="utf-8") as stream:
                    reader = csv.DictReader(stream)
                    fields, records = reader.fieldnames, list(reader)
                # Aggregate result intentionally stays unchanged: the regression
                # check must inspect exact per-sample output, not just totals.
                records[0]["prediction"] = str(1 - int(records[0]["prediction"]))
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(records)
            return result
        with patch.object(comparison, "calibration_identity", return_value="tiny_identity"), \
             patch.object(comparison, "make_model", return_value=model), \
             patch.object(comparison, "file_hash", return_value="tiny_checkpoint"), \
             patch.object(comparison, "quality_pass", side_effect=run_cpu_quality), \
             patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self), \
             patch.object(torch.cuda, "empty_cache"), patch.object(comparison, "log"):
            return comparison.select_deebert(config, {}, self.source, self.out, data)

    def test_actual_cpu_calibration_uses_identity_directory_and_frozen_dense_head(self):
        result = self._calibrate_tiny_cpu_model()
        self.assertEqual(result["dense_validation_accuracy"], 1)
        self.assertEqual(result["accuracy"], 1)
        self.assertTrue(result["actual_runtime_verified"])
        self.assertEqual(result["calibration_protocol_sha256"], "tiny_identity")
        self.assertTrue((self.out / "validation_checks_tiny_identity").is_dir())

    def test_calibration_rejects_per_sample_disagreement_even_with_matching_totals(self):
        with self.assertRaisesRegex(RuntimeError, "per-sample"):
            self._calibrate_tiny_cpu_model(corrupt_record=True)
        self.assertFalse((self.out / "deebert_selection.json").exists())

    def semantic_patches(self, fault=None):
        self.semantic_data = {"test": {"labels": torch.tensor([0, 1, 2, 0, 1, 2])}}
        predictions = {"router_loop": [0, 0, 2, 1, 0, 0],
                       "router_legacy": [0, 1, 1, 2, 0, 2]}
        def quality(torch_module, model, data, name, size, threshold, device, output):
            self.assertEqual((size, threshold), (32, -1))
            labels = data["labels"].tolist()
            rows = [{"sample_index": i, "label": label, "prediction": pred, "batch_exit_layer": 12}
                    for i, (label, pred) in enumerate(zip(labels, predictions[name]))]
            correct = sum(row["prediction"] == row["label"] for row in rows)
            if fault == "unpaired" and name == "router_legacy":
                rows[0], rows[1] = rows[1], rows[0]
            if fault == "both_truncated":
                rows = rows[:-1]
            if fault == "both_wrong_labels":
                rows[0]["label"] = 2
            path = output / (name + "_predictions.csv")
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["sample_index", "label", "prediction", "batch_exit_layer"])
                writer.writeheader()
                writer.writerows(rows)
            return {"variant": name, "batch_size": size, "samples": len(labels), "correct": correct,
                    "accuracy": correct / len(labels), "predictions_file": path.name}
        stack = ExitStack()
        stack.enter_context(patch.object(comparison, "make_model", return_value=object()))
        stack.enter_context(patch.object(comparison, "quality_pass", side_effect=quality))
        stack.enter_context(patch.object(comparison, "file_hash", side_effect=lambda p: self.file_revision + ":" + str(p)))
        stack.enter_context(patch.object(comparison.torch.cuda, "empty_cache"))
        stack.enter_context(patch.object(comparison, "log"))
        return stack

    def run_semantic(self):
        comparison.semantic_diagnostic(self.config, {}, self.source, self.out, self.semantic_data)
        return read_json(self.out / "semantic_diagnostic" / "summary.json")

    def test_semantic_disagreement_counts_include_both_wrong_different_predictions(self):
        with self.semantic_patches():
            result = self.run_semantic()
        self.assertEqual(result["samples"], 6)
        self.assertEqual(result["prediction_disagreements"], 4)
        self.assertEqual(result["query_correct_legacy_wrong"], 1)
        self.assertEqual(result["legacy_correct_query_wrong"], 2)
        # Four disagreements need not equal the three cases changing accuracy:
        # two different wrong labels contribute a disagreement, but no benefit.
        self.assertEqual(result["rows"][0]["correct"] - result["rows"][1]["correct"], -1)

    def test_semantic_resume_rejects_changed_identity_before_model_loading(self):
        with self.semantic_patches():
            self.run_semantic()
            self.file_revision = "changed"
            with patch.object(comparison, "make_model", side_effect=AssertionError("must reject first")):
                with self.assertRaisesRegex(RuntimeError, "identity changed"):
                    self.run_semantic()

    def test_semantic_rejects_unpaired_or_truncated_or_mislabelled_predictions(self):
        for fault in ("unpaired", "both_truncated", "both_wrong_labels"):
            with self.subTest(fault=fault), self.semantic_patches(fault=fault):
                with self.assertRaises(RuntimeError):
                    self.run_semantic()

    def test_no_router_ablation_matches_dense_and_counts_extra_exit_head_only(self):
        torch.manual_seed(80)
        torch.set_num_threads(1)
        model = ComparatorRuntime(config=BertConfig(
            vocab_size=21, hidden_size=8, intermediate_size=16,
            num_attention_heads=2, num_hidden_layers=3, max_position_embeddings=16,
            hidden_dropout_prob=0, attention_probs_dropout_prob=0,
        ), num_labels=2, dropout=0, split_layer=1, backend="no_router").eval()
        ids = torch.tensor([[2, 3, 4, 0], [2, 4, 5, 6]])
        mask = ids.ne(0).long()
        dense = model.forward_dense(ids, mask)
        expected = dense_matmul_flops(2, 4, 8, 16, 2, 3) + 2 * 2 * 8 * 2
        with ExitStack() as stack:
            for router in model.routers:
                stack.enter_context(patch.object(router, "forward", side_effect=AssertionError("router executed")))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as profiler:
                actual, stats = model.forward_dynamic(ids, mask, -1, collect_stats=True)
            torch.testing.assert_close(actual, dense, rtol=1e-5, atol=1e-6)
            measured = sum(event.flops for event in profiler.key_averages()
                           if event.key in {"aten::mm", "aten::addmm", "aten::bmm"})
            self.assertEqual(stats["matmul_flops"], expected)
            self.assertEqual(measured, expected)
            self.assertEqual(stats["stage_b_layers"], [])
            _, early = model.forward_dynamic(ids, mask, float("inf"), collect_stats=True)
            self.assertEqual(early["exit_layer"], 1)
            self.assertEqual(early["matmul_flops"], dense_matmul_flops(2, 4, 8, 16, 2, 1))


if __name__ == "__main__":
    unittest.main()
