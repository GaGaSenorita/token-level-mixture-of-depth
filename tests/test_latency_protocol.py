"""CPU-only protocol tests; no downloads, GPU calls, or trained checkpoints."""
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from benchmarks.measure_latency import (aggregate_results, make_batches, percentile,
                                        sample_indices, summarize_ms, variant_order)


class LatencyProtocolTests(unittest.TestCase):
    def test_unique_reproducible_sample_and_batch_alignment(self):
        indices = sample_indices(7600, 1000, 42)
        self.assertEqual(indices, sample_indices(7600, 1000, 42))
        self.assertEqual(len(set(indices)), 1000)
        batches = make_batches(indices, 32, drop_last=True)
        self.assertEqual(len(batches), 31)
        self.assertEqual([x for b in batches for x in b], indices[:992])
        full = make_batches(indices, 32, drop_last=False)
        self.assertEqual(len(full[-1]), 8)

    def test_latency_quantiles_and_units(self):
        result = summarize_ms([1, 2, 3, 4, 5])
        self.assertEqual(result["p50_ms"], 3)
        self.assertAlmostEqual(result["p95_ms"], 4.8)
        self.assertEqual(result["calls"], 5)
        for invalid in ([], [-1], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                summarize_ms(invalid)

    def test_path_order_alternates(self):
        variants = ["dense", "query", "legacy"]
        self.assertEqual(variant_order(variants, 0), variants)
        self.assertEqual(variant_order(variants, 1), variants[::-1])
        self.assertEqual(variant_order(variants, 2), variants)

    def test_ratios_use_same_batch_baseline(self):
        qualities, timings = [], []
        for batch, flops, latency in ((1, 100, 10), (8, 150, 30)):
            for variant, multiplier in (("dense", 1), ("query", .5)):
                qualities.append({"variant": variant, "batch_size": batch, "accuracy": .9 if variant == "dense" else .89,
                                  "matmul_flops_per_sample": flops * multiplier})
                timings.append({"variant": variant, "batch_size": batch, "latency": {"calls": 1024, "p50_ms": latency * multiplier, "p95_ms": latency * 2 * multiplier},
                                "throughput_samples_per_second": 100 / multiplier, "memory_peak_allocated_bytes": 10,
                                "timed_inputs_matmul_flops_per_sample": flops * multiplier * 1.1, "timed_inputs_accuracy": .9})
        result = aggregate_results(qualities, timings)
        for row in result:
            if row["variant"] == "query":
                self.assertEqual(row["matmul_flops_saving_percent"], 50)
                self.assertEqual(row["latency_p50_speedup"], 2)
                self.assertEqual(row["timed_inputs_matmul_flops_saving_percent"], 50)
                self.assertAlmostEqual(row["accuracy_delta_percentage_points"], -1)

    def test_cpu_smoke_artifacts_and_resume(self):
        """Run the entire harness offline against a tiny random model, never a GPU."""
        import torch
        from transformers import BertConfig
        from benchmarks.hdc_runtime import HDCBertRuntime
        from benchmarks.measure_latency import main, shared_weight_identity
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            local_model = run / "local_model"
            architecture = BertConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=16)
            model = HDCBertRuntime(config=architecture, num_labels=4, split_layer=1)
            model.bert.save_pretrained(local_model)
            torch.save(model.state_dict(), run / "stage1_best.pt")
            with torch.no_grad():
                model.offramp_classifiers[0].weight.add_(.01)
            torch.save(model.state_dict(), run / "stage3_best.pt")
            data = {"input_ids": torch.randint(0, 32, (8, 8)), "attention_mask": torch.ones(8, 8, dtype=torch.long),
                    "labels": torch.arange(8) % 4}
            torch.save({"test": data}, run / "data.pt")
            (run / "selection.json").write_text(json.dumps({"entropy_threshold": .2}))
            config = {"model_name": str(local_model), "num_labels": 4, "split_layer": 1, "max_length": 8}
            config_path = run / "config.json"
            config_path.write_text(json.dumps(config))
            argv = ["measure_latency.py", "--config", str(config_path), "--run-dir", str(run), "--device", "cpu", "--smoke"]
            with patch("sys.argv", argv):
                main()
            summary = json.loads((run / "benchmark_smoke" / "summary.json").read_text())
            self.assertEqual(len(summary["rows"]), 5)
            self.assertTrue(summary["smoke"])
            manifest = json.loads((run / "benchmark_smoke" / "manifest.json").read_text())
            self.assertEqual(manifest["shared_backbone_identity"]["verified_identical"], ["stage1", "stage3"])
            for row in summary["rows"]:
                self.assertEqual(row["samples"], 8)
                self.assertEqual(row["timed_calls_per_repeat"], [8])
            # Re-entry consumes exactly the same quality and timing artifacts.
            with patch("sys.argv", argv):
                main()
            after = json.loads((run / "benchmark_smoke" / "summary.json").read_text())
            self.assertEqual(after["rows"], summary["rows"])
            with torch.no_grad():
                model.classifier.weight.add_(.1)
            torch.save(model.state_dict(), run / "stage3_best.pt")
            with self.assertRaisesRegex(RuntimeError, "differ"):
                shared_weight_identity(torch, {"stage1": run / "stage1_best.pt", "stage3": run / "stage3_best.pt"})


if __name__ == "__main__":
    unittest.main()
