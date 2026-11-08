"""Post-benchmark operator diagnosis; never a replacement for primary latency.

Refuses to initialize CUDA until the primary benchmark's completion artifact
and every expected timing repeat have been verified. Outputs go into a separate
diagnostics directory; primary config, weights, routes and results are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_completed_benchmark(run_dir, batch_size):
    """All checks here use only the standard library, before torch/CUDA import."""
    benchmark = Path(run_dir) / "benchmark"
    if not (benchmark / "summary.json").is_file():
        raise RuntimeError("Primary benchmark/summary.json is absent; profiling refused to avoid GPU contention")
    summary = read_json(benchmark / "summary.json")
    manifest = read_json(benchmark / "manifest.json")
    config = read_json(Path(run_dir) / "config_resolved.json")
    selection = read_json(Path(run_dir) / "selection.json")
    if summary.get("smoke") or manifest.get("smoke") or not summary.get("completed_at"):
        raise RuntimeError("A completed non-smoke primary benchmark is required")
    protocol = manifest.get("protocol_sha256")
    if not protocol or summary.get("protocol_sha256") != protocol:
        raise RuntimeError("Primary summary and manifest identities disagree")
    if config != manifest.get("config") or selection != manifest.get("selection"):
        raise RuntimeError("Resolved config/selection changed after the primary benchmark")
    if batch_size not in manifest.get("batch_sizes", []):
        raise ValueError(f"Batch {batch_size} was not measured in the primary benchmark")
    repeats = int(manifest["repeats"])
    expected = {(variant, b) for b in manifest["batch_sizes"] for variant in ("dense", "query")}
    if 1 in manifest["batch_sizes"]:
        expected.add(("legacy", 1))
    rows = {(r["variant"], r["batch_size"]): r for r in summary.get("rows", [])}
    if repeats < 1 or not expected.issubset(rows):
        raise RuntimeError("Primary summary is missing measured configurations")
    for variant, b in expected:
        values = rows[(variant, b)].get("latency_p50_ms_by_repeat", [])
        if len(values) != repeats or any(not math.isfinite(x) or x <= 0 for x in values):
            raise RuntimeError(f"Primary repeat summaries incomplete: {variant}, batch={b}")
        for repeat in range(repeats):
            path = benchmark / f"latency_{variant}_b{b}_r{repeat}.json"
            if not path.is_file() or read_json(path).get("latency", {}).get("calls", 0) < 1:
                raise RuntimeError(f"Primary timing repeat incomplete: {path.name}")
    return config, selection, manifest, summary


def select_modal_batch(routes_path, batch_size):
    """Choose the first unique batch at the modal exit; ties follow file order."""
    records = []
    with Path(routes_path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            indices = row.get("sample_indices", [])
            if len(indices) != batch_size or row.get("batch_size", batch_size) != batch_size:
                raise ValueError("Routes file contains a batch with unexpected size")
            if any(not isinstance(i, int) or i < 0 for i in indices):
                raise ValueError("Invalid original test sample indices")
            if int(row.get("executions", 1)) < 1:
                continue
            records.append(row)
    if not records:
        raise RuntimeError("No executed batches in the primary routes file")
    histogram = Counter(str(row["exit_layer"]) for row in records)
    modal_exit = max(histogram, key=histogram.get)
    selected = next(row for row in records if str(row["exit_layer"]) == modal_exit)
    weighted = Counter()
    for row in records:
        weighted[str(row["exit_layer"])] += int(row.get("executions", 1))
    return selected, {
        "rule": "First unique batch at the modal exit_layer; ties resolved by first appearance",
        "unique_batch_exit_histogram": dict(histogram),
        "execution_weighted_exit_histogram": dict(weighted),
        "modal_exit_layer": modal_exit,
    }


def event_number(event, *names):
    for name in names:
        if hasattr(event, name):
            return float(getattr(event, name))
    return None


def event_rows(profile):
    rows = []
    for event in profile.key_averages():
        rows.append({
            "operator": event.key, "count": int(event.count),
            "cpu_self_us": float(event.self_cpu_time_total),
            "cpu_total_us": float(event.cpu_time_total),
            "device_self_us": event_number(event, "self_device_time_total", "self_cuda_time_total"),
            "device_total_us": event_number(event, "device_time_total", "cuda_time_total"),
            "profiler_flops_if_available": int(event.flops or 0),
        })
    return rows


def write_csv(path, rows):
    columns = ["operator", "count", "cpu_self_us", "cpu_total_us", "device_self_us",
               "device_total_us", "profiler_flops_if_available"]
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def key_operator(name):
    exact = {"aten::linear", "aten::mm", "aten::addmm", "aten::bmm", "aten::matmul",
             "aten::nonzero", "aten::item", "aten::_local_scalar_dense", "aten::copy_",
             "aten::to", "aten::_to_copy", "cudaDeviceSynchronize", "cudaStreamSynchronize"}
    return name in exact or name.startswith(("aten::index", "aten::_index", "aten::scatter", "cudaMemcpy"))


def profile_variant(torch, model, ids, mask, variant, threshold, device, out, warmup, calls, top):
    def invoke():
        if variant == "dense":
            return model.forward_dense(ids, mask)
        return model.forward_dynamic(ids, mask, threshold, routing="query", collect_stats=False)[0]

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.inference_mode():
        for _ in range(warmup):
            invoke()
        sync()
        started = time.perf_counter()
        with torch.profiler.profile(
            activities=activities, record_shapes=True, profile_memory=False,
            with_stack=False, with_flops=True, acc_events=True,
        ) as profile:
            for _ in range(calls):
                with torch.profiler.record_function(f"diagnostic_{variant}_call"):
                    logits = invoke()
                    sync()
                    del logits
        elapsed = time.perf_counter() - started
    rows = event_rows(profile)
    cpu_top = sorted(rows, key=lambda row: row["cpu_self_us"], reverse=True)[:top]
    device_top = sorted(rows, key=lambda row: row["device_self_us"] or 0, reverse=True)[:top]
    key_rows = [row for row in rows if key_operator(row["operator"])]
    names = {
        "operators": f"{variant}_operators.csv", "top_cpu": f"{variant}_top_cpu.csv",
        "top_device": f"{variant}_top_device.csv", "key_operators": f"{variant}_key_operators.csv",
    }
    for key, data in (("operators", rows), ("top_cpu", cpu_top),
                      ("top_device", device_top), ("key_operators", key_rows)):
        write_csv(out / names[key], data)
    return {
        "variant": variant, "warmup_calls": warmup, "profiled_calls": calls,
        "profile_block_wall_seconds_including_profiler_overhead": elapsed,
        "key_operators": key_rows, "files": names,
        "activities_requested": ["CPU", "CUDA"] if device.type == "cuda" else ["CPU"],
        "device_timing_captured": device.type == "cuda" and any(
            (row["device_self_us"] or 0) > 0 for row in rows
        ),
        "device_timing_note": (
            "No positive CUDA device timings captured; CPU counts remain valid but cannot establish a CUDA bottleneck."
            if device.type == "cuda" and not any((row["device_self_us"] or 0) > 0 for row in rows)
            else "Device fields describe captured events; CPU-only runs have no CUDA timing evidence."
        ),
        "operator_time_unit": "microseconds, aggregated across all profiled calls",
        "top_sort": "self CPU/device time respectively; inclusive totals are never summed",
        "warning": "Profiling adds overhead and synchronization; these timings DO NOT replace primary latency results.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs" / "latency_20261108")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cuda", help="cuda for the original GPU; cpu for separate diagnostics")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if min(args.batch, args.warmup, args.calls, args.top) < 1:
        parser.error("batch/warmup/calls/top must be positive")
    run = args.run_dir.resolve()
    cfg, selection, manifest, summary = require_completed_benchmark(run, args.batch)
    routes_path = run / "benchmark" / f"latency_query_b{args.batch}_r0_routes.jsonl"
    selected, choice = select_modal_batch(routes_path, args.batch)
    threshold = float(selection["entropy_threshold"])
    if threshold != float(manifest["entropy_threshold"]):
        raise RuntimeError("Entropy threshold differs from the primary benchmark")
    checkpoints = {"dense": run / "stage1_best.pt", "query": run / "stage3_best.pt"}
    expected_files = [(checkpoints["dense"], manifest["stage1_sha256"]),
                      (checkpoints["query"], manifest["stage3_sha256"]),
                      (run / "data.pt", manifest["dataset_sha256"])]
    for path, expected in expected_files:
        if file_hash(path) != expected:
            raise RuntimeError(f"Artifact changed after primary benchmark: {path}")
    runtime_path = ROOT / "benchmarks" / "hdc_runtime.py"
    runtime_expected = next((v for k, v in manifest["source_sha256"].items()
                             if k.replace("\\", "/") == "benchmarks/hdc_runtime.py"), None)
    if not runtime_expected or file_hash(runtime_path) != runtime_expected:
        raise RuntimeError("Runtime source differs from the primary benchmark")
    if not Path(cfg["model_name"]).is_dir():
        raise RuntimeError("Resolved local model snapshot required; diagnostic never downloads models")

    # No GPU initialization occurs before ALL primary-completion/identity checks.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from benchmarks.hdc_runtime import HDCBertRuntime
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only cpu/cuda diagnostics are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU substitution")
    torch.set_num_threads(int(manifest["environment"]["cpu_threads"]))
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU diagnostic"
    if device.type == "cuda" and device_name != manifest["environment"].get("device_name"):
        raise RuntimeError("CUDA device name differs from primary benchmark hardware")
    dataset = torch.load(run / "data.pt", map_location="cpu", weights_only=False)["test"]
    indices = selected["sample_indices"]
    if max(indices) >= len(dataset["labels"]):
        raise ValueError("Selected test indices are out of range")
    ids = dataset["input_ids"][indices].to(device)
    mask = dataset["attention_mask"][indices].to(device)
    model = HDCBertRuntime(**{k: cfg[k] for k in (
        "model_name", "num_labels", "dropout", "split_layer", "tau", "target_keep_ratio"
    )}).to(device=device, dtype=torch.float32).eval()
    model.load_state_dict(torch.load(checkpoints["query"], map_location="cpu", weights_only=True), strict=True)
    _, observed = model.forward_dynamic(ids, mask, threshold, routing="query", collect_stats=True)
    for key in ("exit_layer", "exited_early", "stage_a_layers", "stage_b_layers", "matmul_flops"):
        if observed[key] != selected[key]:
            raise RuntimeError(f"Representative route no longer matches primary recording: {key}")

    output = args.output_dir or run / "diagnostics" / (
        f"profile_b{args.batch}_{device.type}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing diagnostic files: {output}")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for variant in ("dense", "query"):
        model.load_state_dict(torch.load(checkpoints[variant], map_location="cpu", weights_only=True), strict=True)
        model.eval()
        results.append(profile_variant(torch, model, ids, mask, variant, threshold, device,
                                       output, args.warmup, args.calls, args.top))
    report = {
        "created_at": datetime.now().isoformat(), "primary_completed_at": summary["completed_at"],
        "primary_protocol_sha256": manifest["protocol_sha256"],
        "profile_script_sha256": file_hash(__file__), "runtime_sha256": runtime_expected,
        "run_dir": str(run), "routes_file": str(routes_path), "selection": choice,
        "sample_indices": indices, "recorded_route": selected, "observed_route": observed,
        "environment": {"device": str(device), "device_name": device_name, "torch": torch.__version__,
                        "dtype": "float32", "attention": "eager", "tf32": False,
                        "cpu_threads": torch.get_num_threads()},
        "variants": results,
        "limits": [
            "One representative modal-exit batch; not an average of the full workload or evidence about every exit pattern.",
            "CPU/CUDA activity tracing and shape collection add overhead. Use benchmark/summary.json for latency conclusions.",
            "CPU/device total times are inclusive and may overlap; DO NOT sum them as total runtime.",
            "Operator counts and self-time ranks diagnose candidates. No optimization or production speedup is established here.",
            "CPU diagnostic timings, if selected, are not evidence about CUDA bottlenecks.",
        ],
    }
    (output / "profile_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output / "profile_summary.json", flush=True)


if __name__ == "__main__":
    main()
