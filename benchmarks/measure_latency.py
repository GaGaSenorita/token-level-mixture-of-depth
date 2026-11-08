"""Quality, executed-path analytical FLOPs, and synchronized latency benchmark.

Run after train/select has finished. No test labels are used to select settings.
Latency excludes statistics collection, checkpoint loading, and data transfer.
The supplementary text-to-label measurement explicitly includes tokenization.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def make_batches(indices, batch_size, drop_last=False):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return [list(indices[i:i + batch_size]) for i in range(0, len(indices), batch_size)
            if not drop_last or len(indices[i:i + batch_size]) == batch_size]


def sample_indices(size, requested, seed):
    if size < 1 or requested < 1:
        raise ValueError("sample sizes must be positive")
    return random.Random(seed).sample(range(size), min(size, requested))


def percentile(values, q):
    if not values or not 0 <= q <= 1:
        raise ValueError("nonempty values and quantile in [0, 1] required")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def summarize_ms(values):
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("durations must be finite and nonnegative")
    return {"calls": len(values), "mean_ms": statistics.mean(values),
            "p50_ms": percentile(values, .5), "p95_ms": percentile(values, .95),
            "min_ms": min(values), "max_ms": max(values)}


def variant_order(variants, repetition):
    return list(variants) if repetition % 2 == 0 else list(reversed(variants))


def telemetry():
    try:
        command = ["nvidia-smi", "--query-gpu=name,driver_version,pstate,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory,utilization.gpu,memory.used", "--format=csv,noheader,nounits"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return {"timestamp": time.time(), "columns": command[1].split("=", 1)[1],
                "value": result.stdout.strip(), "error": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"timestamp": time.time(), "error": str(exc)}


def synchronize(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_weights(torch, model, path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    del state
    model.to(device=device, dtype=torch.float32)
    model.eval()
    synchronize(torch, device)


def shared_weight_identity(torch, paths):
    """Verify that only added off-ramps/routers differ between checkpoints."""
    hashes, key_lists = {}, {}
    for name, path in paths.items():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "state_dict" in state:
            state = state["state_dict"]
        keys = sorted(k for k in state if k.startswith(("bert.", "classifier.")))
        if not keys or not any(k.startswith("classifier.") for k in keys):
            raise ValueError(f"Missing dense backbone/final classifier parameters in {path}")
        digest = hashlib.sha256()
        for key in keys:
            value = state[key].detach().cpu().contiguous()
            digest.update(key.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            # uint8 view supports bfloat16 too, without casting values for hashing.
            digest.update(value.view(torch.uint8).numpy().tobytes())
        hashes[name] = digest.hexdigest()
        key_lists[name] = keys
        del state
    if len(set(hashes.values())) != 1 or len({tuple(v) for v in key_lists.values()}) != 1:
        raise RuntimeError("Stage 1 and Stage 3 dense backbone/final classifier differ; cannot claim matched shared weights")
    return {"sha256": next(iter(hashes.values())), "tensor_count": len(next(iter(key_lists.values()))),
            "verified_identical": list(paths), "scope": "bert.* and classifier.* tensors, including unused pooler"}


def invoke(model, variant, ids, mask, threshold, collect_stats=False):
    if variant == "dense":
        return model.forward_dense(ids, mask), {}
    return model.forward_dynamic(ids, mask, entropy_threshold=threshold,
                                 routing=variant, collect_stats=collect_stats)


def quality_pass(torch, model, dataset, variant, batch_size, threshold, device,
                 output_dir, limit=None):
    from benchmarks.hdc_runtime import dense_matmul_flops
    path = output_dir / f"quality_{variant}_b{batch_size}.json"
    if path.exists():
        log(f"Resume completed quality {variant} batch={batch_size}")
        return read_json(path)
    size = len(dataset["labels"]) if limit is None else min(limit, len(dataset["labels"]))
    correct, total_flops, count, all_exits = 0, 0, 0, Counter()
    total_kept, total_valid, total_routed_samples = Counter(), Counter(), Counter()
    details_path = output_dir / f"quality_{variant}_b{batch_size}_routes.jsonl"
    predictions_path = output_dir / f"quality_{variant}_b{batch_size}_predictions.csv"
    config = model.bert.config
    start = time.perf_counter()
    with torch.inference_mode(), details_path.open("w", encoding="utf-8") as details, predictions_path.open("w", encoding="utf-8", newline="") as predictions:
        writer = csv.writer(predictions)
        writer.writerow(["sample_index", "label", "prediction", "batch_exit_layer"])
        for batch_number, indices in enumerate(make_batches(list(range(size)), batch_size)):
            ids = dataset["input_ids"][indices].to(device)
            mask = dataset["attention_mask"][indices].to(device)
            labels = dataset["labels"][indices]
            logits, stats = invoke(model, variant, ids, mask, threshold, collect_stats=True)
            prediction = logits.argmax(dim=-1).cpu()
            batch_correct = int((prediction == labels).sum())
            if variant == "dense":
                stats = {"exit_layer": model.n_layers, "exited_early": False,
                         "matmul_flops": dense_matmul_flops(len(indices), ids.shape[1], config.hidden_size,
                             config.intermediate_size, model.num_labels, n_layers=model.n_layers)}
            batch_flops = int(stats["matmul_flops"])
            correct += batch_correct
            total_flops += batch_flops
            count += len(indices)
            all_exits[str(stats["exit_layer"])] += len(indices)
            valid_counts = mask.sum(dim=-1).cpu().tolist()
            for layer in stats.get("stage_b_layers", []):
                key = str(layer["layer"])
                total_kept[key] += sum(layer["k_per_sample"])
                total_valid[key] += sum(valid_counts)
                total_routed_samples[key] += len(indices)
            details.write(json.dumps({"indices": indices, "correct": batch_correct, **stats}) + "\n")
            writer.writerows(zip(indices, labels.tolist(), prediction.tolist(), [stats["exit_layer"]] * len(indices)))
            if batch_number % max(1, 1000 // batch_size) == 0:
                log(f"quality {variant} b{batch_size}: {count}/{size}, acc={correct/count:.4f}")
    result = {"variant": variant, "batch_size": batch_size, "samples": count, "correct": correct,
              "accuracy": correct / count, "matmul_flops_total": total_flops,
              "matmul_flops_per_sample": total_flops / count, "exit_histogram": dict(all_exits),
              "routed_token_keep_rate_by_layer": {k: total_kept[k] / total_valid[k] for k in total_kept},
              "routed_samples_by_layer": dict(total_routed_samples),
              "wall_seconds_including_statistics": time.perf_counter() - start,
              "flops_convention": "Analytical executed matmul FLOPs, MAC=2; includes padded projections/FFN, routers and classifiers; excludes elementwise, indexing, softmax, LayerNorm and embedding lookup.",
              "predictions_file": predictions_path.name, "routes_file": details_path.name}
    write_json(path, result)
    return result


def latency_pass(torch, model, dataset, variant, batch_size, threshold, device,
                 output_dir, indices, repetition, warmup_batches, min_timed_calls=1024):
    stem = f"latency_{variant}_b{batch_size}_r{repetition}"
    path = output_dir / (stem + ".json")
    if path.exists():
        log(f"Resume completed latency {stem}")
        return read_json(path)
    batches = make_batches(indices, batch_size, drop_last=True)
    if not batches:
        raise ValueError("Not enough sampled inputs for the requested batch size")
    # Identical complete input tensors are resident for every variant and batch.
    resident_ids = dataset["input_ids"][indices].to(device)
    resident_mask = dataset["attention_mask"][indices].to(device)
    gpu_batches = [(resident_ids[i:i + batch_size], resident_mask[i:i + batch_size])
                   for i in range(0, len(batches) * batch_size, batch_size)]
    call_indices = [i % len(gpu_batches) for i in range(max(len(gpu_batches), min_timed_calls))]
    executions_per_batch = Counter(call_indices)
    durations = []
    before = telemetry() if device.type == "cuda" else {}
    with torch.inference_mode():
        if device.type == "cuda":
            # Clear leftovers before warming, never discard the warmed allocator.
            torch.cuda.empty_cache()
        for i in range(warmup_batches):
            ids, mask = gpu_batches[i % len(gpu_batches)]
            invoke(model, variant, ids, mask, threshold)
        synchronize(torch, device)
        if device.type == "cuda":
            memory_baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            memory_baseline = None
        for batch_index in call_indices:
            ids, mask = gpu_batches[batch_index]
            synchronize(torch, device)
            started = time.perf_counter_ns()
            logits, _ = invoke(model, variant, ids, mask, threshold)
            synchronize(torch, device)
            durations.append((time.perf_counter_ns() - started) / 1e6)
            del logits
        memory_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        # Throughput is a separate continuous stream, without harness per-call synchronization.
        # The dynamic model's own control-flow synchronization remains part of the measured code.
        synchronize(torch, device)
        started = time.perf_counter_ns()
        for batch_index in call_indices:
            ids, mask = gpu_batches[batch_index]
            logits, _ = invoke(model, variant, ids, mask, threshold)
            del logits
        synchronize(torch, device)
        throughput_ms = (time.perf_counter_ns() - started) / 1e6
    after = telemetry() if device.type == "cuda" else {}
    # Separate, untimed counting of exactly the groups used above. Whole-batch
    # exit makes random measurement groups different from full-test order.
    # Repeated identical batches use the same observed deterministic route.
    from benchmarks.hdc_runtime import dense_matmul_flops
    cfg = model.bert.config
    timed_flops, timed_correct, timed_exits = 0, 0, Counter()
    with torch.inference_mode(), (output_dir / (stem + "_routes.jsonl")).open("w", encoding="utf-8") as stream:
        for batch_index, (ids, mask) in enumerate(gpu_batches):
            logits, stats = invoke(model, variant, ids, mask, threshold, collect_stats=True)
            if variant == "dense":
                stats = {"exit_layer": model.n_layers, "matmul_flops": dense_matmul_flops(
                    batch_size, ids.shape[1], cfg.hidden_size, cfg.intermediate_size, model.num_labels, n_layers=model.n_layers)}
            uses = executions_per_batch[batch_index]
            timed_flops += uses * stats["matmul_flops"]
            timed_exits[str(stats["exit_layer"])] += uses * batch_size
            timed_correct += uses * int((logits.argmax(-1).cpu() == dataset["labels"][batches[batch_index]]).sum())
            stream.write(json.dumps({"sample_indices": batches[batch_index], "executions": uses, **stats}) + "\n")
    sample_executions = len(call_indices) * batch_size
    with (output_dir / (stem + ".csv")).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "batch_size", "repeat", "batch_index", "sample_indices", "latency_ms"])
        for i, duration in enumerate(durations):
            writer.writerow([variant, batch_size, repetition, i, json.dumps(batches[call_indices[i]]), duration])
    result = {"variant": variant, "batch_size": batch_size, "repeat": repetition,
              "samples": sample_executions, "unique_samples": len(batches) * batch_size,
              "sample_executions": sample_executions, "unique_batches": len(batches), "min_timed_calls": min_timed_calls,
              "warmup_batches": warmup_batches,
              "latency": summarize_ms(durations), "throughput_samples_per_second": sample_executions / (throughput_ms / 1000),
              "throughput_window_ms": throughput_ms, "memory_baseline_allocated_bytes": memory_baseline,
              "memory_peak_allocated_bytes": memory_peak,
              "memory_peak_increment_bytes": None if memory_peak is None else memory_peak - memory_baseline,
              "timed_inputs_matmul_flops_per_sample": timed_flops / sample_executions,
              "timed_inputs_accuracy": timed_correct / sample_executions, "timed_inputs_exit_histogram": dict(timed_exits),
              "telemetry_before": before, "telemetry_after": after,
              "timing_scope": "FP32 eager model only; input IDs/mask already on device; no statistics; no tokenization or copies; no serving/network/queue; batch latency is not per-request latency for batch>1."}
    write_json(path, result)
    log(f"{stem}: median {result['latency']['p50_ms']:.3f} ms/batch, throughput {result['throughput_samples_per_second']:.1f} samples/s")
    return result


def text_to_label_pass(torch, model, tokenizer, dataset, variant, threshold, device,
                       output_dir, indices, max_length):
    path = output_dir / f"text_to_label_{variant}.json"
    if path.exists():
        return read_json(path)
    texts = dataset.get("texts")
    if texts is None:
        return {"variant": variant, "skipped": "No raw text in data.pt"}
    durations = []
    with torch.inference_mode():
        for index in indices[:30]:
            encoded = tokenizer(texts[index], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            invoke(model, variant, encoded["input_ids"].to(device), encoded["attention_mask"].to(device), threshold)[0].argmax(-1).item()
        for index in indices:
            synchronize(torch, device)
            started = time.perf_counter_ns()
            encoded = tokenizer(texts[index], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            result, _ = invoke(model, variant, encoded["input_ids"].to(device), encoded["attention_mask"].to(device), threshold)
            prediction = result.argmax(-1).item()
            synchronize(torch, device)
            durations.append((time.perf_counter_ns() - started) / 1e6)
    result = {"variant": variant, "latency": summarize_ms(durations), "batch_size": 1,
              "scope": "Single-process text -> tokenization -> H2D -> FP32 eager inference -> CPU integer label; excludes network/queue/serving; one supplementary pass."}
    write_json(path, result)
    with (output_dir / f"text_to_label_{variant}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_index", "latency_ms"])
        writer.writerows(zip(indices, durations))
    return result


def aggregate_results(qualities, latencies):
    rows = []
    for quality in qualities:
        variant, batch_size = quality["variant"], quality["batch_size"]
        matching = [x for x in latencies if x["variant"] == variant and x["batch_size"] == batch_size]
        if not matching:
            continue
        row = {**quality,
               "latency_p50_ms_mean_of_repeats": statistics.mean(x["latency"]["p50_ms"] for x in matching),
               "latency_p95_ms_mean_of_repeats": statistics.mean(x["latency"]["p95_ms"] for x in matching),
               "throughput_samples_per_second_mean": statistics.mean(x["throughput_samples_per_second"] for x in matching),
               "latency_p50_ms_by_repeat": [x["latency"]["p50_ms"] for x in matching],
               "throughput_by_repeat": [x["throughput_samples_per_second"] for x in matching],
               "timed_inputs_matmul_flops_per_sample": statistics.mean(x["timed_inputs_matmul_flops_per_sample"] for x in matching),
               "timed_inputs_accuracy": statistics.mean(x["timed_inputs_accuracy"] for x in matching),
               "timed_calls_per_repeat": [x["latency"]["calls"] for x in matching],
               "memory_peak_allocated_bytes": max((x["memory_peak_allocated_bytes"] for x in matching if x["memory_peak_allocated_bytes"] is not None), default=None)}
        rows.append(row)
    for row in rows:
        baseline = next(r for r in rows if r["variant"] == "dense" and r["batch_size"] == row["batch_size"])
        row["accuracy_delta_percentage_points"] = 100 * (row["accuracy"] - baseline["accuracy"])
        row["matmul_flops_saving_percent"] = 100 * (1 - row["matmul_flops_per_sample"] / baseline["matmul_flops_per_sample"])
        row["timed_inputs_matmul_flops_saving_percent"] = 100 * (1 - row["timed_inputs_matmul_flops_per_sample"] / baseline["timed_inputs_matmul_flops_per_sample"])
        row["latency_p50_speedup"] = baseline["latency_p50_ms_mean_of_repeats"] / row["latency_p50_ms_mean_of_repeats"]
        row["latency_p50_reduction_percent"] = 100 * (1 - row["latency_p50_ms_mean_of_repeats"] / baseline["latency_p50_ms_mean_of_repeats"])
    return rows


def write_report(path, manifest, rows, text_results):
    lines = ["# HDC-BERT supplementary local benchmark", "",
             f"Hardware: {manifest['environment'].get('device_name', 'CPU smoke')}; FP32 eager attention; fixed padding {manifest['max_length']}.",
             f"Selected entropy threshold: {manifest['entropy_threshold']}. Checkpoints and source hashes: manifest.json.", "",
             "This is a new single-seed reproduction, not a timing remeasurement of unavailable historical weights.",
             "Dense and HDC use the same frozen fine-tuned backbone and final classifier. Query routing preserves the trained full-context attention semantics; legacy routing is a diagnostic with different attention semantics.", "",
             "FLOPs in the table use the same grouped inputs as latency. Complete-test FLOPs are separately retained in summary.json.", "",
             "| Path | Batch | Full-test accuracy | Δ accuracy (pp) | Timed-input GFLOPs/text | Timed-input FLOPs saved | P50 ms/batch | P95 ms/batch | P50 speedup | Texts/s | Peak allocated GiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        memory = "n/a" if row["memory_peak_allocated_bytes"] is None else f"{row['memory_peak_allocated_bytes']/2**30:.3f}"
        lines.append(f"| {row['variant']} | {row['batch_size']} | {row['accuracy']:.2%} | {row['accuracy_delta_percentage_points']:+.2f} | {row['timed_inputs_matmul_flops_per_sample']/1e9:.3f} | {row['timed_inputs_matmul_flops_saving_percent']:.2f}% | {row['latency_p50_ms_mean_of_repeats']:.3f} | {row['latency_p95_ms_mean_of_repeats']:.3f} | {row['latency_p50_speedup']:.3f}× | {row['throughput_samples_per_second_mean']:.1f} | {memory} |")
    lines += ["", "Interpretation and limits:", "",
              "- P50/P95 are means of the separate repeat percentiles. Repeat values and every timed call remain available in JSON/CSV.",
              "- Fixed unique inputs are cycled to reach the registered minimum timed call count at every batch size. Repeated inputs improve timing sampling but do not add independent quality examples. FLOPs for those exact input groups are counted in an additional untimed pass, separate from the full test evaluation.",
              "- Batch 1 latency describes a single local model call. For larger batches, milliseconds are per batch; dividing by batch size is amortized cost, not user waiting time.",
              "- Throughput uses a separate continuous window. Dynamic control-flow synchronization remains included, but the harness does not synchronize after every call in that window.",
              "- FLOPs are analytical counts of the matrix multiplications actually executed for each observed route (1 MAC = 2 FLOPs). They are not hardware-instruction measurements. Counts include all padded FFN/projection work, routers and classifier heads; exclude softmax, LayerNorm, activation, embedding lookup and gather/scatter overhead. These exclusions still affect measured latency.",
              "- Whole-batch early exit is retained. Consequently accuracy and route/FLOPs distributions are re-evaluated for every batch size; batch 1 quality is not reused for batch 32.",
              "- Both paths use eager attention and FP32 without TF32. This isolates the project implementation; an optimized SDPA dense baseline may be faster and has not been evaluated by this comparison.",
              "- Peak allocated CUDA memory includes the sole resident model, resident measurement inputs and temporary tensors. It excludes CUDA context/driver memory. The allocated baseline and increment are also recorded.",
              "- Data preprocessing, model load and tokenization are excluded from model-only timing. There is no network, serving stack, concurrency or deployment claim.",
              "- This benchmark uses one trained seed and one laptop GPU. Repeating inference measures timing variation, not training-seed uncertainty."]
    if text_results:
        lines += ["", "Supplementary local raw-text-to-label timing (one pass, batch 1):", ""]
        for item in text_results:
            if "latency" in item:
                lines.append(f"- {item['variant']}: P50 {item['latency']['p50_ms']:.3f} ms, P95 {item['latency']['p95_ms']:.3f} ms; {item['latency']['calls']} texts.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="tiny protocol check; writes separate benchmark_smoke outputs")
    args = parser.parse_args()
    import torch
    import transformers
    from transformers import AutoTokenizer
    from benchmarks.hdc_runtime import HDCBertRuntime
    config = read_json(args.config)
    settings = config.get("benchmark", {})
    if config.get("benchmark_precision", "float32") != "float32":
        raise ValueError("This registered benchmark supports float32 only; use a separately registered protocol for other precision")
    torch.set_num_threads(settings.get("cpu_threads", config.get("cpu_threads", 4)))
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing to silently replace the GPU benchmark with CPU")
    output_dir = args.run_dir / ("benchmark_smoke" if args.smoke else "benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.run_dir / "selection.json"
    selection = read_json(selection_path)
    threshold = float(selection["entropy_threshold"])
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Selected entropy threshold must be finite and nonnegative")
    checkpoint_paths = {"dense": args.run_dir / "stage1_best.pt", "query": args.run_dir / "stage3_best.pt", "legacy": args.run_dir / "stage3_best.pt"}
    data_path = args.run_dir / "data.pt"
    # Local trusted experiment artifact, includes raw text strings as well as tensors.
    dataset = torch.load(data_path, map_location="cpu", weights_only=False)["test"]
    batch_sizes = [1, 4] if args.smoke else settings.get("batch_sizes", config.get("benchmark_batches", [1, 8, 32]))
    requested = 8 if args.smoke else settings.get("sample_count", config.get("benchmark_samples", 1024))
    repeats = 1 if args.smoke else settings.get("repeats", config.get("benchmark_repeats", 3))
    warmup = 2 if args.smoke else settings.get("warmup_batches", config.get("warmup_batches", 30))
    min_calls = 8 if args.smoke else settings.get("min_timed_calls", config.get("benchmark_min_calls", 1024))
    seed = settings.get("sampling_seed", 20261108)
    indices = sample_indices(len(dataset["labels"]), requested, seed)
    environment = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "transformers": transformers.__version__,
                   "cuda_runtime": torch.version.cuda, "cpu_threads": torch.get_num_threads(), "device": str(device), "tf32": False,
                   "attention": "eager", "dtype": "float32", "telemetry_initial": telemetry() if device.type == "cuda" else {}}
    if device.type == "cuda":
        environment.update(device_name=torch.cuda.get_device_name(device), device_capability=list(torch.cuda.get_device_capability(device)))
    shared_identity = shared_weight_identity(torch, {"stage1": checkpoint_paths["dense"], "stage3": checkpoint_paths["query"]})
    manifest = {"config": config, "selection": selection, "entropy_threshold": threshold, "max_length": config.get("max_length", 128),
                "shared_backbone_identity": shared_identity,
                "dataset_sha256": file_hash(data_path), "stage1_sha256": file_hash(checkpoint_paths["dense"]), "stage3_sha256": file_hash(checkpoint_paths["query"]),
                "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in [Path(__file__), ROOT / "benchmarks" / "hdc_runtime.py"]},
                "sample_indices": indices, "batch_sizes": batch_sizes, "repeats": repeats, "warmup": warmup, "min_timed_calls": min_calls, "smoke": args.smoke,
                "environment": environment}
    # Environment timestamps/telemetry are intentionally excluded from the resume identity.
    identity = {k: v for k, v in manifest.items() if k != "environment"}
    identity["device_identity"] = {k: v for k, v in environment.items() if k not in {"telemetry_initial"}}
    manifest["protocol_sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path)["protocol_sha256"] != manifest["protocol_sha256"]:
        raise RuntimeError(f"Benchmark identity changed; preserve {output_dir} and choose a new run directory rather than mixing results")
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    model = HDCBertRuntime(model_name=config["model_name"], num_labels=config.get("num_labels", 4), dropout=config.get("dropout", .1),
                           split_layer=config.get("split_layer", 6), tau=config.get("tau", .5), target_keep_ratio=config.get("target_keep_ratio", .5)).to(device).eval()
    qualities, latencies, text_results = [], [], []
    for variant in ("dense", "query", "legacy"):
        load_weights(torch, model, checkpoint_paths[variant], device)
        for batch_size in ([1] if variant == "legacy" else batch_sizes):
            qualities.append(quality_pass(torch, model, dataset, variant, batch_size, threshold, device, output_dir, limit=8 if args.smoke else None))
    for batch_size in batch_sizes:
        variants = ["dense", "query"] + (["legacy"] if batch_size == 1 else [])
        for repetition in range(repeats):
            for variant in variant_order(variants, repetition):
                load_weights(torch, model, checkpoint_paths[variant], device)
                latencies.append(latency_pass(torch, model, dataset, variant, batch_size, threshold, device, output_dir, indices, repetition, warmup, min_calls))
    text_count = 4 if args.smoke else settings.get("end_to_end_samples", 200)
    if text_count and dataset.get("texts") is not None:
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"], local_files_only=True)
        for variant in ("dense", "query"):
            load_weights(torch, model, checkpoint_paths[variant], device)
            text_results.append(text_to_label_pass(torch, model, tokenizer, dataset, variant, threshold, device, output_dir,
                                                   indices[:text_count], config.get("max_length", 128)))
    rows = aggregate_results(qualities, latencies)
    write_json(output_dir / "summary.json", {"protocol_sha256": manifest["protocol_sha256"], "rows": rows, "text_to_label": text_results,
                                              "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke": args.smoke})
    write_report(output_dir / "report.md", manifest, rows, text_results)
    log(f"Benchmark complete: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
