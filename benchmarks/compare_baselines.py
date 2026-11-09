"""Matched four-model comparison plus same-weight implementation diagnosis."""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
import transformers

from benchmarks.comparator_runtime import ComparatorRuntime
from benchmarks.hdc_runtime import dense_matmul_flops
from benchmarks.measure_latency import (aggregate_results, file_hash, latency_pass,
    load_weights, quality_pass, read_json, sample_indices, shared_weight_identity,
    telemetry, write_json)
from benchmarks.run_experiment import seed_all
from models.deebert import entropy_from_logits

ROOT = Path(__file__).resolve().parents[1]


def log(event, **kwargs):
    print(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **kwargs}), flush=True)


def specs(source, out):
    return {
        "dense": (source / "stage1_best.pt", 6, "query_loop"),
        "deebert": (out / "deebert_best.pt", 12, "query_loop"),
        "router_loop": (out / "router_best.pt", 0, "query_loop"),
        "router_batched": (out / "router_best.pt", 0, "query_batched"),
        "hdc_loop": (source / "stage3_best.pt", 6, "query_loop"),
        "hdc_batched": (source / "stage3_best.pt", 6, "query_batched"),
        "hdc_exitonly": (source / "stage3_best.pt", 6, "no_router"),
        "router_legacy": (out / "router_best.pt", 0, "legacy"),
        "router_densegate": (out / "router_best.pt", 0, "dense_gate"),
    }


def make_model(name, config, resolved, source, out):
    path, split, backend = specs(source, out)[name]
    model = ComparatorRuntime(model_name=resolved["model_name"], num_labels=config["num_labels"],
        dropout=config["dropout"], split_layer=split, tau=config["tau"],
        target_keep_ratio=config["router_target_keep_ratio"], backend=backend)
    load_weights(torch, model, path, torch.device("cuda"))
    if name == "deebert":
        for key, value in model.classifier.state_dict().items():
            if not torch.equal(value, model.offramp_classifiers[-1].state_dict()[key]):
                raise RuntimeError("DeeBERT final off-ramp must equal the shared dense classifier")
    return model


def calibration_identity(config, source, out):
    identity = {"config": config, "checkpoint_sha256": file_hash(out/"deebert_best.pt"),
        "data_sha256": file_hash(source/"data.pt"),
        "source_sha256": {name: file_hash(ROOT/name) for name in (
            "benchmarks/compare_baselines.py", "benchmarks/comparator_runtime.py",
            "benchmarks/hdc_runtime.py", "benchmarks/measure_latency.py",
            "models/hdc_bert.py", "models/deebert.py")}}
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


def select_deebert(config, resolved, source, out, data):
    path = out / "deebert_selection.json"
    identity = calibration_identity(config,source,out)
    if path.exists():
        cached=read_json(path)
        if cached.get("calibration_protocol_sha256") != identity:
            raise RuntimeError("DeeBERT calibration identity changed; preserve results and use a new run")
        return cached
    seed_all(config["seed"])
    model = make_model("deebert", config, resolved, source, out)
    valid = data["validation"]
    n = len(valid["labels"])
    logits = torch.empty(n, model.n_layers, model.num_labels)
    with torch.inference_mode():
        for start in range(0, n, 32):
            ids = valid["input_ids"][start:start+32].cuda()
            mask = valid["attention_mask"][start:start+32].cuda()
            hidden, extended = model._prepare(ids, mask)
            for j, layer in enumerate(model.bert.encoder.layer):
                hidden = layer(hidden, extended)[0]
                logits[start:start+len(ids), j] = model.offramp_classifiers[j](hidden[:, 0]).cpu()
    torch.save(logits, out / "deebert_validation_logits.pt")
    entropies = entropy_from_logits(logits)
    mc = model.bert.config
    per_layer = dense_matmul_flops(1, config["max_length"], mc.hidden_size,
                                   mc.intermediate_size, model.num_labels, 1)
    dense_accuracy = int((logits[:, -1].argmax(-1) == valid["labels"]).sum()) / n
    candidates = []
    for threshold in config["threshold_candidates"]:
        condition = entropies < threshold
        condition[:, -1] = True
        exit_idx = condition.int().argmax(1)
        selected = logits[torch.arange(n), exit_idx]
        correct = int((selected.argmax(-1) == valid["labels"]).sum())
        row = {"entropy_threshold": threshold, "accuracy": correct/n, "correct": correct, "n": n,
               "matmul_flops_per_sample": float(((exit_idx+1)*per_layer).double().mean())}
        candidates.append(row)
    eligible = [x for x in candidates if x["accuracy"] >= dense_accuracy-config["maximum_validation_accuracy_drop"]]
    chosen = min(eligible, key=lambda x:(x["matmul_flops_per_sample"], -x["accuracy"])) if eligible else max(candidates, key=lambda x:x["accuracy"])
    # Validate simulated batch-1 exits with the actual runtime once, before test.
    checkdir = out / ("validation_checks_"+identity[:16])
    checkdir.mkdir(exist_ok=True)
    actual = quality_pass(torch, model, valid, "deebert", 1, chosen["entropy_threshold"], torch.device("cuda"), checkdir)
    if actual["correct"] != chosen["correct"] or abs(actual["matmul_flops_per_sample"]-chosen["matmul_flops_per_sample"]) > 1:
        raise RuntimeError("DeeBERT batch-1 calibration differs from actual runtime")
    condition = entropies < chosen["entropy_threshold"]
    condition[:, -1] = True
    expected_exits = condition.int().argmax(1)
    expected_predictions = logits[torch.arange(n),expected_exits].argmax(-1)
    with (checkdir/actual["predictions_file"]).open(encoding="utf-8",newline="") as stream:
        records=list(csv.DictReader(stream))
    if len(records)!=n or any(int(row["sample_index"])!=i or int(row["prediction"])!=int(expected_predictions[i])
        or int(row["batch_exit_layer"])!=int(expected_exits[i])+1 for i,row in enumerate(records)):
        raise RuntimeError("DeeBERT per-sample predictions/exits differ from calibration simulation")
    result = {**chosen, "constraint_met": bool(eligible), "selection_split": "validation", "candidates": candidates,
              "dense_validation_accuracy": dense_accuracy, "actual_runtime_verified": True,
              "checkpoint_sha256": file_hash(out/"deebert_best.pt"), "calibration_protocol_sha256": identity}
    write_json(path, result)
    log("deebert_selection", **chosen)
    del model
    torch.cuda.empty_cache()
    return result


def threshold_for(name, source, out):
    if name.startswith("hdc"):
        return float(read_json(source/"selection.json")["entropy_threshold"])
    if name == "deebert":
        return float(read_json(out/"deebert_selection.json")["entropy_threshold"])
    return -1.0


def verify_gpu_backends(config, resolved, source, out, data):
    """Real selected weights and GPU precision, before substantive timings."""
    identity={"config":config,"checkpoints":{name:file_hash(specs(source,out)[name][0]) for name in ("router_loop","hdc_loop")},
        "data_sha256":file_hash(source/"data.pt"),
        "source_sha256":{p:file_hash(ROOT/p) for p in ("benchmarks/compare_baselines.py",
            "benchmarks/comparator_runtime.py","benchmarks/batched_query.py","benchmarks/hdc_runtime.py")}}
    path=out/"gpu_backend_verification.json"
    if path.exists():
        cached=read_json(path)
        if cached["identity"]!=identity:
            raise RuntimeError("GPU equivalence check identity changed")
        return cached
    records=[]
    with torch.inference_mode():
        for name in ("router_loop","hdc_loop"):
            model=make_model(name,config,resolved,source,out)
            for size in config["benchmark_batches"]:
                ids=data["validation"]["input_ids"][:size].cuda()
                mask=data["validation"]["attention_mask"][:size].cuda()
                for threshold in ([-1.0] if name=="router_loop" else [-1.0,threshold_for(name,source,out)]):
                    model.backend="query_loop"
                    reference,reference_stats=model.forward_dynamic(ids,mask,threshold,collect_stats=True)
                    for backend in ("query_batched","dense_gate"):
                        model.backend=backend
                        result,stats=model.forward_dynamic(ids,mask,threshold,collect_stats=True)
                        torch.testing.assert_close(result,reference,rtol=3e-4,atol=3e-4)
                        ref_counts=[x["k_per_sample"] for x in reference_stats["stage_b_layers"]]
                        counts=[x["k_per_sample"] for x in stats["stage_b_layers"]]
                        if ref_counts!=counts or reference_stats["exit_layer"]!=stats["exit_layer"]:
                            raise RuntimeError("GPU implementation change altered routing on verification inputs")
                        if not torch.equal(result.argmax(-1),reference.argmax(-1)):
                            raise RuntimeError("GPU implementation change altered verification predictions")
                        records.append({"variant":name,"batch_size":size,"threshold":threshold,"backend":backend,
                            "maximum_absolute_logit_difference":float((result-reference).abs().max()),
                            "predictions_identical":bool(torch.equal(result.argmax(-1),reference.argmax(-1))),
                            "route_counts_identical":True})
            del model
            torch.cuda.empty_cache()
    result={"identity":identity,"records":records,"passed":True,
        "scope":"First validation inputs at each registered batch size, real selected weights, FP32 GPU. Not test quality or latency."}
    write_json(path,result)
    log("gpu_backend_verification_passed",cases=len(records))
    return result


def run_measurements(config, resolved, source, out, data, diagnostic=False):
    output = out / ("diagnostic_benchmark" if diagnostic else "benchmark")
    output.mkdir(exist_ok=True)
    names = ["dense", "router_loop", "router_legacy", "router_densegate", "router_batched"] if diagnostic else [
        "dense", "deebert", "router_loop", "router_batched", "hdc_loop", "hdc_batched", "hdc_exitonly"]
    sizes = [1, 32] if diagnostic else config["benchmark_batches"]
    indices = sample_indices(len(data["test"]["labels"]), config["benchmark_samples"], config["benchmark_sampling_seed"])
    paths = {name: specs(source,out)[name][0] for name in names}
    shared = shared_weight_identity(torch, paths)
    sources = ["benchmarks/compare_baselines.py", "benchmarks/comparator_runtime.py", "benchmarks/batched_query.py",
               "benchmarks/hdc_runtime.py", "benchmarks/measure_latency.py",
               "models/hdc_bert.py", "models/deebert.py", "models/router_tuning_bert.py"]
    manifest = {"config": config, "models": names, "batch_sizes": sizes,
        "shared_backbone_identity": shared, "checkpoints": {k:file_hash(v) for k,v in paths.items()},
        "dataset_sha256": file_hash(source/"data.pt"), "sample_indices": indices,
        "thresholds": {name: threshold_for(name,source,out) for name in names},
        "source_sha256": {p:file_hash(ROOT/p) for p in sources},
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "precision": "float32", "attention": "eager", "tf32": False, "threads": torch.get_num_threads()},
        "diagnostic": diagnostic}
    manifest["protocol_sha256"] = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    manifest_path = output/"manifest.json"
    if manifest_path.exists() and read_json(manifest_path)["protocol_sha256"] != manifest["protocol_sha256"]:
        raise RuntimeError("Changed experiment protocol; refuse mixed resume")
    if not manifest_path.exists():
        write_json(manifest_path,manifest)
    qualities,latencies = [],[]
    for name in names:
        model = make_model(name,config,resolved,source,out)
        for size in sizes:
            qualities.append(quality_pass(torch,model,data["test"],name,size,threshold_for(name,source,out),torch.device("cuda"),output))
        del model
        torch.cuda.empty_cache()
    for size in sizes:
        for repeat in range(config["benchmark_repeats"]):
            # Rotate and reverse; timings for all paths are new under this protocol.
            order = names[repeat:]+names[:repeat]
            if repeat%2:
                order.reverse()
            for name in order:
                model = make_model(name,config,resolved,source,out)
                latencies.append(latency_pass(torch,model,data["test"],name,size,threshold_for(name,source,out),torch.device("cuda"),
                    output,indices,repeat,config["warmup_batches"],config["benchmark_min_calls"]))
                del model
                torch.cuda.empty_cache()
    rows = aggregate_results(qualities,latencies)
    write_json(output/"summary.json",{"protocol_sha256":manifest["protocol_sha256"],"rows":rows,
        "completed_at":time.strftime("%Y-%m-%d %H:%M:%S"),"diagnostic":diagnostic})
    fields = ["variant","batch_size","accuracy","accuracy_delta_percentage_points",
        "latency_p50_ms_mean_of_repeats","latency_p95_ms_mean_of_repeats","latency_p50_reduction_percent",
        "timed_inputs_matmul_flops_per_sample","timed_inputs_matmul_flops_saving_percent",
        "throughput_samples_per_second_mean"]
    with (output/"summary.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    log("measurement_complete", output=str(output), rows=len(rows))


def semantic_diagnostic(config, resolved, source, out, data):
    """Full-test consequence of K/V removal, without another timing sweep."""
    output=out/"semantic_diagnostic"
    output.mkdir(exist_ok=True)
    identity={"config":config,"checkpoint_sha256":file_hash(out/"router_best.pt"),
        "data_sha256":file_hash(source/"data.pt"),"batch_size":32,
        "source_sha256":{p:file_hash(ROOT/p) for p in (
            "benchmarks/compare_baselines.py","benchmarks/comparator_runtime.py",
            "benchmarks/hdc_runtime.py","benchmarks/measure_latency.py","models/hdc_bert.py",
            "models/router_tuning_bert.py")}}
    identity["protocol_sha256"]=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    path=output/"manifest.json"
    if path.exists() and read_json(path)!=identity:
        raise RuntimeError("Semantic diagnostic identity changed")
    write_json(path,identity)
    rows=[]; predictions={}
    for name in ("router_loop","router_legacy"):
        model=make_model(name,config,resolved,source,out)
        row=quality_pass(torch,model,data["test"],name,32,-1,torch.device("cuda"),output)
        rows.append(row)
        with (output/row["predictions_file"]).open(encoding="utf-8",newline="") as stream:
            predictions[name]=list(csv.DictReader(stream))
        del model
        torch.cuda.empty_cache()
    left,right=predictions["router_loop"],predictions["router_legacy"]
    labels=data["test"]["labels"]
    if len(left)!=len(labels) or len(left)!=len(right) or any(
        int(a["sample_index"])!=i or a["sample_index"]!=b["sample_index"]
        or int(a["label"])!=int(labels[i]) or a["label"]!=b["label"]
        for i,(a,b) in enumerate(zip(left,right))):
        raise RuntimeError("Unpaired semantic diagnostic")
    for records,row in zip((left,right),rows):
        if row["samples"]!=len(records) or row["correct"]!=sum(r["prediction"]==r["label"] for r in records):
            raise RuntimeError("Semantic diagnostic quality aggregate mismatch")
    result={"protocol_sha256":identity["protocol_sha256"],"rows":rows,"samples":len(left),
        "prediction_disagreements":sum(a["prediction"]!=b["prediction"] for a,b in zip(left,right)),
        "query_correct_legacy_wrong":sum(a["prediction"]==a["label"] and b["prediction"]!=b["label"] for a,b in zip(left,right)),
        "legacy_correct_query_wrong":sum(b["prediction"]==b["label"] and a["prediction"]!=a["label"] for a,b in zip(left,right)),
        "scope":"New validation-selected Router weights, full test; K-by-K versus full-K/V query semantics. Not historical checkpoints and not a timing comparison."}
    write_json(output/"summary.json",result)
    log("semantic_diagnostic_complete", disagreements=result["prediction_disagreements"],samples=len(left))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/comparator_audit_20261109.json")
    parser.add_argument("--phase",choices=["verify","select","measure","diagnose","semantic","all"],default="all")
    args=parser.parse_args()
    config=read_json(args.config)
    source=ROOT/config["source_run"]; out=ROOT/config["output_root"]
    out.mkdir(parents=True,exist_ok=True)
    resolved=read_json(source/"config_resolved.json")
    for key in ("max_length","num_labels","tau"):
        if config[key] != resolved[key]:
            raise RuntimeError(f"Cannot reuse source HDC calibration after changing {key}")
    torch.set_num_threads(config["cpu_threads"])
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():
        raise RuntimeError("A GPU is required; no silent CPU substitution")
    if os.name=="nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    data=torch.load(source/"data.pt",map_location="cpu",weights_only=False,mmap=True)
    log("start", phase=args.phase, telemetry=telemetry())
    if args.phase in {"verify","all"}:
        verify_gpu_backends(config,resolved,source,out,data)
    if args.phase in {"select","all"}:
        select_deebert(config,resolved,source,out,data)
    if args.phase in {"measure","all"}:
        if read_json(out/"deebert_selection.json").get("calibration_protocol_sha256") != calibration_identity(config,source,out):
            raise RuntimeError("DeeBERT selection does not match current inputs/source")
        run_measurements(config,resolved,source,out,data)
    if args.phase=="diagnose":
        run_measurements(config,resolved,source,out,data,diagnostic=True)
    if args.phase in {"semantic","all"}:
        semantic_diagnostic(config,resolved,source,out,data)


if __name__=="__main__":
    main()
