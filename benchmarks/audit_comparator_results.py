"""Independently audit completed comparator raw artifacts, without GPU use.

Run AFTER primary and semantic summaries exist. This module imports none of the
measured benchmark/runtime helpers. It independently recomputes CSV percentiles,
route-shape matrix FLOPs, paired labels/counts, and aggregate metrics. Source,
data and checkpoint hashing is intentionally deferred until completion to avoid
I/O contention during timed GPU measurements. Output is a local-only JSON audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("dense", "deebert", "router_loop", "router_batched", "hdc_loop", "hdc_batched", "hdc_exitonly")
BACKENDS = {"deebert": "query_loop", "router_loop": "query_loop", "router_batched": "query_batched",
            "hdc_loop": "query_loop", "hdc_batched": "query_batched", "hdc_exitonly": "no_router",
            "router_legacy": "legacy"}


def require(condition, description):
    if not condition:
        raise AssertionError(description)


def close(actual, expected, description):
    require(math.isfinite(float(actual)) and math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-8),
            f"{description}: recorded {actual!r}, recomputed {expected!r}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def jsonl_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def verify_protocol_hash(manifest):
    payload = {key: value for key, value in manifest.items() if key != "protocol_sha256"}
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    require(actual == manifest["protocol_sha256"], "Manifest protocol hash does not match its payload")


def quantile(values, fraction):
    values = sorted(values)
    require(bool(values), "Empty timing CSV")
    pos = fraction * (len(values) - 1)
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    return values[low] * (1 - (pos - low)) + values[high] * (pos - low)


def matmul_cost(rows, inputs, outputs):
    return 2 * rows * inputs * outputs


def attention_cost(b, t, k, h):
    """Q + K + V + attention scores + attention values + output projection."""
    if k == 0:
        return 0
    return (matmul_cost(b*k, h, h) + 2*matmul_cost(b*t, h, h)
            + 2*matmul_cost(b*k, t, h) + matmul_cost(b*k, h, h))


def route_flops(row, name, indices, valid_lengths, shape):
    b, t, h, f, c, layers = len(indices), shape["t"], shape["h"], shape["f"], shape["c"], shape["layers"]
    ff = matmul_cost(b*t, h, f) + matmul_cost(b*t, f, h)
    head = matmul_cost(b, h, c)
    dense_layer = attention_cost(b, t, t, h) + ff
    if name == "dense":
        require(row["exit_layer"] == layers, "Dense route exits early")
        return layers*dense_layer + head
    require(row["backend"] == BACKENDS[name], f"Wrong execution backend for {name}")
    require(row["batch_size"] == b and row["padded_length"] == t, f"Shape mismatch for {name}")
    a = row["stage_a_layers"]
    expected_a = min(row["exit_layer"], 6) if name.startswith("hdc") else row["exit_layer"] if name == "deebert" else 0
    require(a == expected_a, f"Wrong off-ramp layer count for {name}")
    total = a*(dense_layer + head)
    routed = row["stage_b_layers"]
    if row["exited_early"]:
        require(row["exit_layer"] < layers and not routed and not row["final_head"], "Invalid early-exit route")
        return total
    require(row["exit_layer"] == layers, "Full route has wrong exit layer")
    if name == "deebert":
        require(not row["final_head"] and not routed, "DeeBERT double-counted final head or routing")
        return total
    require(row["final_head"], "Full route missing final head")
    if name == "hdc_exitonly":
        require(not routed, "Exit-only ablation executed routers")
        return total + (layers-a)*dense_layer + head
    require([entry["layer"] for entry in routed] == list(range(a+1, layers+1)), "Wrong routed layer sequence")
    for entry in routed:
        counts = entry["k_per_sample"]
        require(len(counts) == b, "Wrong retained-count batch shape")
        require(all(isinstance(k, int) and 0 <= k <= valid_lengths[i] for k, i in zip(counts, indices)),
                "Retained-token count exceeds valid input tokens")
        total += matmul_cost(b*t, h, 1) + ff
        if BACKENDS[name] == "query_batched":
            kmax = max(counts)
            require(entry["backend"] == "batched_query" and entry["packed_query_length"] == kmax,
                    "Batched query shape/count mismatch")
            attention = attention_cost(b, t, kmax, h)
            require(entry["attention_matmul_flops"] == attention, "Packed attention FLOPs mismatch")
        elif BACKENDS[name] == "legacy":
            attention = sum(attention_cost(1, k, k, h) for k in counts)
        else:
            attention = sum(attention_cost(1, t, k, h) for k in counts)
        total += attention
    return total + head


def quality_audit(directory, name, batch, labels, valid_lengths, shape):
    record = read_json(directory / f"quality_{name}_b{batch}.json")
    require((record["variant"], record["batch_size"]) == (name, batch), "Quality identity mismatch")
    predictions = csv_rows(directory / record["predictions_file"])
    n = len(labels)
    require(record["samples"] == len(predictions) == n, f"Incomplete full-test predictions: {name} b{batch}")
    for i, row in enumerate(predictions):
        require(int(row["sample_index"]) == i and int(row["label"]) == labels[i], "Prediction labels/order mismatch")
        require(0 <= int(row["prediction"]) < shape["c"], "Invalid predicted class")
    correct = sum(int(row["prediction"]) == labels[i] for i, row in enumerate(predictions))
    require(record["correct"] == correct, "Quality accuracy count mismatch")
    close(record["accuracy"], correct/n, "Full-test accuracy")
    flop_sum, offset = 0, 0
    exits, kept, valid, routed_samples = Counter(), Counter(), Counter(), Counter()
    for route in jsonl_rows(directory / record["routes_file"]):
        indices = list(range(offset, min(offset+batch, n)))
        require(route["indices"] == indices and bool(indices), "Quality route grouping mismatch")
        route_correct = sum(int(predictions[i]["prediction"]) == labels[i] for i in indices)
        require(route["correct"] == route_correct, "Route prediction count mismatch")
        require(all(int(predictions[i]["batch_exit_layer"]) == route["exit_layer"] for i in indices),
                "Prediction exit layers differ from routes")
        flops = route_flops(route, name, indices, valid_lengths, shape)
        require(flops == route["matmul_flops"], "Quality executed-shape FLOPs mismatch")
        flop_sum += flops
        exits[str(route["exit_layer"])] += len(indices)
        for entry in route.get("stage_b_layers", []):
            key = str(entry["layer"])
            kept[key] += sum(entry["k_per_sample"])
            valid[key] += sum(valid_lengths[i] for i in indices)
            routed_samples[key] += len(indices)
        offset += len(indices)
    require(offset == n, "Incomplete quality routes")
    require(record["matmul_flops_total"] == flop_sum and record["exit_histogram"] == dict(exits), "Quality totals mismatch")
    close(record["matmul_flops_per_sample"], flop_sum/n, "Quality FLOPs/sample")
    require(record["routed_samples_by_layer"] == dict(routed_samples), "Routed sample totals mismatch")
    require(set(record["routed_token_keep_rate_by_layer"]) == set(kept), "Missing kept-token rates")
    for key in kept:
        close(record["routed_token_keep_rate_by_layer"][key], kept[key]/valid[key], "Kept-token rate")
    return record, predictions


def latency_audit(directory, name, batch, repeat, config, indices, valid_lengths, shape):
    stem = f"latency_{name}_b{batch}_r{repeat}"
    record = read_json(directory / (stem + ".json"))
    require((record["variant"], record["batch_size"], record["repeat"]) == (name, batch, repeat), "Timing identity mismatch")
    groups = [indices[i:i+batch] for i in range(0, len(indices)-batch+1, batch)]
    calls = max(len(groups), config["benchmark_min_calls"])
    timings = csv_rows(directory / (stem + ".csv"))
    require(len(timings) == calls, f"Incomplete timing CSV {stem}")
    durations = []
    for i, row in enumerate(timings):
        require((row["variant"], int(row["batch_size"]), int(row["repeat"]), int(row["batch_index"]))
                == (name, batch, repeat, i), "Raw timing row identity mismatch")
        require(json.loads(row["sample_indices"]) == groups[i % len(groups)], "Unpaired timing input groups")
        duration = float(row["latency_ms"])
        require(math.isfinite(duration) and duration >= 0, "Invalid latency")
        durations.append(duration)
    require(record["latency"]["calls"] == calls, "Recorded timing call count mismatch")
    expected_quantiles = {"p50_ms": quantile(durations, .5), "p95_ms": quantile(durations, .95),
                          "mean_ms": statistics.mean(durations), "min_ms": min(durations), "max_ms": max(durations)}
    for key, value in expected_quantiles.items():
        close(record["latency"][key], value, f"{stem} {key}")
    for key, value in {"samples": calls*batch, "sample_executions": calls*batch, "unique_samples": len(groups)*batch,
                       "unique_batches": len(groups), "min_timed_calls": config["benchmark_min_calls"],
                       "warmup_batches": config["warmup_batches"]}.items():
        require(record[key] == value, f"{stem} {key} mismatch")
    require(record["throughput_window_ms"] > 0, "Invalid throughput window")
    close(record["throughput_samples_per_second"], calls*batch*1000/record["throughput_window_ms"], "Throughput arithmetic")
    uses = Counter(i % len(groups) for i in range(calls))
    route_rows = list(jsonl_rows(directory / (stem + "_routes.jsonl")))
    require(len(route_rows) == len(groups), "Incomplete timed route groups")
    flop_sum, exits = 0, Counter()
    for j, (route, group) in enumerate(zip(route_rows, groups)):
        require(route["sample_indices"] == group and route["executions"] == uses[j], "Timed routes not paired or weighted correctly")
        flops = route_flops(route, name, group, valid_lengths, shape)
        require(route["matmul_flops"] == flops, "Timed executed-shape FLOPs mismatch")
        flop_sum += flops*uses[j]
        exits[str(route["exit_layer"])] += batch*uses[j]
    close(record["timed_inputs_matmul_flops_per_sample"], flop_sum/(calls*batch), "Weighted timed FLOPs")
    require(record["timed_inputs_exit_histogram"] == dict(exits), "Timed exit histogram mismatch")
    if record["memory_peak_allocated_bytes"] is not None:
        require(record["memory_peak_increment_bytes"] == record["memory_peak_allocated_bytes"]-record["memory_baseline_allocated_bytes"],
                "Memory increment mismatch")
    return record, route_rows


def aggregate_audit(summary, qualities, latencies):
    rows = {(row["variant"], row["batch_size"]): row for row in summary["rows"]}
    require(len(rows) == len(summary["rows"]) == len(qualities), "Incomplete or duplicate summary rows")
    calculated = {}
    for key, quality in qualities.items():
        row, measurements = rows[key], latencies[key]
        for field, value in quality.items():
            require(row[field] == value, f"Summary quality field {field} differs from raw quality JSON")
        means = {"latency_p50_ms_mean_of_repeats": statistics.mean(x["latency"]["p50_ms"] for x in measurements),
                 "latency_p95_ms_mean_of_repeats": statistics.mean(x["latency"]["p95_ms"] for x in measurements),
                 "throughput_samples_per_second_mean": statistics.mean(x["throughput_samples_per_second"] for x in measurements),
                 "timed_inputs_matmul_flops_per_sample": statistics.mean(x["timed_inputs_matmul_flops_per_sample"] for x in measurements),
                 "timed_inputs_accuracy": statistics.mean(x["timed_inputs_accuracy"] for x in measurements)}
        for field, value in means.items():
            close(row[field], value, f"Summary {key} {field}")
        require(row["latency_p50_ms_by_repeat"] == [x["latency"]["p50_ms"] for x in measurements], "Summary repeat P50 mismatch")
        require(row["throughput_by_repeat"] == [x["throughput_samples_per_second"] for x in measurements], "Summary repeat throughput mismatch")
        require(row["timed_calls_per_repeat"] == [x["latency"]["calls"] for x in measurements], "Summary repeat calls mismatch")
        peaks = [x["memory_peak_allocated_bytes"] for x in measurements if x["memory_peak_allocated_bytes"] is not None]
        require(row["memory_peak_allocated_bytes"] == (max(peaks) if peaks else None), "Summary memory peak mismatch")
        calculated[key] = {**quality, **means}
    for key, row in rows.items():
        actual, dense = calculated[key], calculated[("dense", key[1])]
        fields = {"accuracy_delta_percentage_points": 100*(actual["accuracy"]-dense["accuracy"]),
                  "matmul_flops_saving_percent": 100*(1-actual["matmul_flops_per_sample"]/dense["matmul_flops_per_sample"]),
                  "timed_inputs_matmul_flops_saving_percent": 100*(1-actual["timed_inputs_matmul_flops_per_sample"]/dense["timed_inputs_matmul_flops_per_sample"]),
                  "latency_p50_speedup": dense["latency_p50_ms_mean_of_repeats"]/actual["latency_p50_ms_mean_of_repeats"],
                  "latency_p50_reduction_percent": 100*(1-actual["latency_p50_ms_mean_of_repeats"]/dense["latency_p50_ms_mean_of_repeats"])}
        for field, value in fields.items():
            close(row[field], value, f"Summary reduction {key} {field}")
    return rows


def audit(run_dir, expected_test_samples=7600):
    benchmark, semantic = run_dir/"benchmark", run_dir/"semantic_diagnostic"
    require((benchmark/"summary.json").exists() and (semantic/"summary.json").exists(),
            "Both completed summaries are required; do not audit during GPU timing")
    manifest, summary = read_json(benchmark/"manifest.json"), read_json(benchmark/"summary.json")
    verify_protocol_hash(manifest)
    require(summary["protocol_sha256"] == manifest["protocol_sha256"], "Summary belongs to another protocol")
    config = manifest["config"]
    require(tuple(manifest["models"]) == MODELS and manifest["batch_sizes"] == [1, 8, 32], "Unexpected primary model/batch mapping")
    require(config["benchmark_repeats"] == 3 and not manifest["diagnostic"], "Unexpected repeat/diagnostic protocol")
    source = ROOT/config["source_run"]
    hash_cache = {}
    def verify_file(path, expected):
        path = Path(path).resolve()
        if path not in hash_cache:
            hash_cache[path] = digest(path)
        require(hash_cache[path] == expected, f"Artifact SHA256 mismatch: {path.name}")
    for relative, sha in manifest["source_sha256"].items():
        verify_file(ROOT/relative, sha)
    verify_file(source/"data.pt", manifest["dataset_sha256"])
    paths = {"dense": source/"stage1_best.pt", "deebert": run_dir/"deebert_best.pt",
             "router_loop": run_dir/"router_best.pt", "router_batched": run_dir/"router_best.pt",
             "hdc_loop": source/"stage3_best.pt", "hdc_batched": source/"stage3_best.pt", "hdc_exitonly": source/"stage3_best.pt"}
    for name, path in paths.items():
        verify_file(path, manifest["checkpoints"][name])
    # CPU-only mmap: never construct a model or initialize CUDA.
    import torch
    torch.set_num_threads(1)
    data = torch.load(source/"data.pt", map_location="cpu", weights_only=False, mmap=True)
    labels = data["test"]["labels"].tolist()
    valid_lengths = data["test"]["attention_mask"].sum(-1).tolist()
    require(len(labels) == expected_test_samples, "Unexpected official test size")
    resolved = read_json(source/"config_resolved.json")
    model_config = read_json(Path(resolved["model_name"])/"config.json")
    shape = {"t": config["max_length"], "h": model_config["hidden_size"], "f": model_config["intermediate_size"],
             "c": config["num_labels"], "layers": model_config["num_hidden_layers"]}
    require(shape["layers"] == 12 and data["test"]["input_ids"].shape[1] == shape["t"], "Unexpected BERT/input shape")
    expected_indices = random.Random(config["benchmark_sampling_seed"]).sample(range(len(labels)), min(len(labels), config["benchmark_samples"]))
    require(manifest["sample_indices"] == expected_indices, "Sampling identity mismatch")
    shared = manifest["shared_backbone_identity"]
    for path in dict.fromkeys(paths.values()):
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        keys = sorted(key for key in state if key.startswith(("bert.", "classifier.")))
        sha = hashlib.sha256()
        for key in keys:
            value = state[key].contiguous()
            sha.update(key.encode()); sha.update(str(value.dtype).encode()); sha.update(str(tuple(value.shape)).encode())
            sha.update(value.view(torch.uint8).numpy().tobytes())
        require(sha.hexdigest() == shared["sha256"] and len(keys) == shared["tensor_count"], "Shared backbone identity mismatch")
        if path.name == "deebert_best.pt":
            for suffix in ("weight", "bias"):
                require(torch.equal(state[f"classifier.{suffix}"], state[f"offramp_classifiers.11.{suffix}"]), "DeeBERT final head mapping mismatch")
        del state
    require(set(shared["verified_identical"]) == set(MODELS), "Incomplete shared-weight identity list")
    dee_selection = read_json(run_dir/"deebert_selection.json")
    hdc_selection = read_json(source/"selection.json")
    for name in MODELS:
        threshold = dee_selection["entropy_threshold"] if name == "deebert" else hdc_selection["entropy_threshold"] if name.startswith("hdc") else -1
        require(manifest["thresholds"][name] == threshold, "Wrong selected threshold in measurement")
    qualities, latencies, prediction_sets, differences = {}, {}, {}, []
    timing_calls = 0
    for name in MODELS:
        for batch in manifest["batch_sizes"]:
            key = (name, batch)
            quality, predictions = quality_audit(benchmark, name, batch, labels, valid_lengths, shape)
            qualities[key], prediction_sets[key] = quality, [int(row["prediction"]) for row in predictions]
            latencies[key] = []
            reference_routes = None
            for repeat in range(config["benchmark_repeats"]):
                record, routes = latency_audit(benchmark, name, batch, repeat, config, expected_indices, valid_lengths, shape)
                latencies[key].append(record)
                timing_calls += record["latency"]["calls"]
                if reference_routes is None:
                    reference_routes = routes
                require(routes == reference_routes, "Routing/counting changed across timing repeats")
    for family in ("router", "hdc"):
        for batch in manifest["batch_sizes"]:
            left, right = prediction_sets[(family+"_loop", batch)], prediction_sets[(family+"_batched", batch)]
            differences.append({"family": family, "batch_size": batch,
                                "full_test_prediction_disagreements": sum(a != b for a, b in zip(left, right))})
    rows = aggregate_audit(summary, qualities, latencies)
    table = csv_rows(benchmark/"summary.csv")
    require(len(table) == len(rows), "Summary CSV row count differs")
    for row in table:
        reference = rows[(row["variant"], int(row["batch_size"]))]
        for key, value in row.items():
            if key != "variant":
                close(float(value), reference[key], f"Summary CSV {key}")
    semantic_manifest, semantic_summary = read_json(semantic/"manifest.json"), read_json(semantic/"summary.json")
    verify_protocol_hash(semantic_manifest)
    require(semantic_manifest["config"] == config and semantic_manifest["batch_size"] == 32, "Semantic protocol differs")
    verify_file(run_dir/"router_best.pt", semantic_manifest["checkpoint_sha256"])
    verify_file(source/"data.pt", semantic_manifest["data_sha256"])
    for relative, sha in semantic_manifest["source_sha256"].items():
        verify_file(ROOT/relative, sha)
    semantic_rows, semantic_predictions = [], []
    for name in ("router_loop", "router_legacy"):
        row, predictions = quality_audit(semantic, name, 32, labels, valid_lengths, shape)
        semantic_rows.append(row)
        semantic_predictions.append([int(record["prediction"]) for record in predictions])
    require(semantic_predictions[0] == prediction_sets[("router_loop", 32)], "Semantic query predictions differ from primary")
    left, right = semantic_predictions
    counts = {"samples": len(labels), "prediction_disagreements": sum(a != b for a, b in zip(left, right)),
              "query_correct_legacy_wrong": sum(a == y and b != y for a, b, y in zip(left, right, labels)),
              "legacy_correct_query_wrong": sum(b == y and a != y for a, b, y in zip(left, right, labels))}
    require(semantic_summary["rows"] == semantic_rows and semantic_summary["protocol_sha256"] == semantic_manifest["protocol_sha256"],
            "Semantic summary identity/rows mismatch")
    for key, value in counts.items():
        require(semantic_summary[key] == value, f"Semantic {key} mismatch")
    return {"passed": True, "protocol_sha256": manifest["protocol_sha256"],
            "primary_quality_evaluations": len(qualities), "test_samples_per_evaluation": len(labels),
            "timing_json_csv_pairs": sum(map(len, latencies.values())), "raw_timed_calls_verified": timing_calls,
            "source_data_checkpoint_files_hashed": len(hash_cache), "shared_backbone_tensors_verified": shared["tensor_count"],
            "independently_recomputed": ["full-test labels, predictions and accuracy", "quality and timing executed-shape matrix FLOPs",
                "timing groups, repetitions and weighted FLOPs", "raw latency mean/P50/P95/min/max", "summary means and reductions",
                "throughput arithmetic and memory arithmetic", "checkpoint/backbone identity", "semantic paired disagreements"],
            "loop_batched_prediction_differences": differences, "semantic_counts": counts,
            "limitations": ["Timed routes omit predictions/correct counts; timed_inputs_accuracy is checked only as an aggregate of timing JSONs.",
                "Throughput arithmetic is verified from its saved elapsed window; the physical elapsed window cannot be independently replayed from artifacts.",
                "This is an artifact audit, not a new latency measurement or uncertainty estimate."],
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def self_test():
    close(quantile([1, 3, 5, 9], .5), 4, "Median self-test")
    close(quantile([1, 3, 5, 9], .95), 8.4, "P95 self-test")
    require(attention_cost(3, 6, 3, 16) == 31104, "Independent attention arithmetic self-test")
    try:
        close(1, 2, "Deliberate mismatch")
    except AssertionError:
        pass
    else:
        raise AssertionError("Audit failed to reject a numerical mismatch")
    print(json.dumps({"self_test": "passed"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT/"outputs/comparator_audit_20261109")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    result = audit(args.run_dir.resolve())
    output = args.output or args.run_dir/"independent_audit.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("passed", "primary_quality_evaluations", "timing_json_csv_pairs",
                                                  "raw_timed_calls_verified", "semantic_counts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
