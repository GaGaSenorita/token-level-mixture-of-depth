"""Reproducible new AG News training; no historical test-set checkpoint selection."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, get_linear_schedule_with_warmup


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def log(out, event, **values):
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **values}
    print(json.dumps(row, ensure_ascii=False), flush=True)
    with (out / "progress.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def environment(cfg, out):
    import transformers
    import datasets
    torch.set_num_threads(cfg["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if os.name == "nt":
        # Prevent idle sleep only while this process lives; do not alter power settings.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    record = {
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "datasets": datasets.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tf32": False, "cpu_threads": cfg["cpu_threads"],
        "config": cfg,
    }
    write_json(out / "environment.json", record)
    return record


def prepare(cfg, out):
    cfg = dict(cfg)
    cfg["model_name"] = snapshot_download(cfg["model_name"], revision=cfg["model_revision"], local_files_only=True)
    dataset_path = Path(snapshot_download(cfg["dataset"], repo_type="dataset", revision=cfg["dataset_revision"], local_files_only=True))
    write_json(out / "config_resolved.json", cfg)
    if (out / "data.pt").exists():
        log(out, "data_reuse", file=str(out / "data.pt"))
        return cfg
    import pyarrow.parquet as pq
    train_files = sorted((dataset_path / "data").glob("train-*.parquet"))
    test_files = sorted((dataset_path / "data").glob("test-*.parquet"))
    if not train_files or not test_files:
        raise RuntimeError("Pinned cached AG News parquet files not found")
    raw_train = pq.read_table(train_files).to_pydict()
    raw_test = pq.read_table(test_files).to_pydict()
    labels = np.asarray(raw_train["label"])
    rng = np.random.default_rng(cfg["seed"])
    valid_indices = []
    for label in range(cfg["num_labels"]):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        valid_indices.extend(indices[:cfg["validation_size"] // cfg["num_labels"]].tolist())
    valid_indices = np.asarray(valid_indices, dtype=np.int64)
    rng.shuffle(valid_indices)
    train_indices = np.setdiff1d(np.arange(len(labels)), valid_indices)
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], local_files_only=True)
    packed = {}
    for name, raw, indices in [
        ("train", raw_train, train_indices),
        ("validation", raw_train, valid_indices),
        ("test", raw_test, np.arange(len(raw_test["label"]))),
    ]:
        texts = [raw["text"][int(i)] for i in indices]
        enc = tok(texts, max_length=cfg["max_length"], truncation=True, padding="max_length", return_tensors="pt")
        packed[name] = {
            "input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
            "labels": torch.tensor([raw["label"][int(i)] for i in indices]), "texts": texts,
        }
        log(out, "data_prepared", split=name, samples=len(texts))
    torch.save(packed, out / "data.pt")
    write_json(out / "split_indices.json", {"train": train_indices.tolist(), "validation": valid_indices.tolist()})
    write_json(out / "data_manifest.json", {
        "dataset_revision": cfg["dataset_revision"],
        "counts": {k: len(v["labels"]) for k, v in packed.items()},
        "train_indices_sha256": hashlib.sha256(train_indices.tobytes()).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(valid_indices.tobytes()).hexdigest(),
        "tokenized_test_sha256": hashlib.sha256(packed["test"]["input_ids"].numpy().tobytes()).hexdigest(),
    })
    return cfg


def make_model(cfg):
    from benchmarks.hdc_runtime import HDCBertRuntime
    return HDCBertRuntime(**{k: cfg[k] for k in ["model_name", "num_labels", "dropout", "split_layer", "tau", "target_keep_ratio"]}).cuda()


def batches(data, size, order=None):
    if order is None:
        order = torch.arange(len(data["labels"]))
    for start in range(0, len(order), size):
        idx = order[start:start + size]
        yield {k: data[k][idx].cuda(non_blocking=False) for k in ["input_ids", "attention_mask", "labels"]}


@torch.inference_mode()
def accuracy(model, data, mode="dense", batch_size=32, threshold=-1.0):
    from benchmarks.hdc_runtime import dense_matmul_flops
    model.eval()
    correct = 0
    total = 0
    flops = 0
    for batch in batches(data, batch_size):
        if mode == "dense":
            logits = model.forward_dense(batch["input_ids"], batch["attention_mask"])
            mc = model.bert.config
            flops += dense_matmul_flops(len(batch["labels"]), batch["input_ids"].shape[1],
                                       mc.hidden_size, mc.intermediate_size, model.num_labels, model.n_layers)
        else:
            logits, stats = model.forward_dynamic(batch["input_ids"], batch["attention_mask"], threshold,
                                                  routing="query", collect_stats=True)
            flops += stats["matmul_flops"]
        correct += int((logits.argmax(-1) == batch["labels"]).sum())
        total += len(batch["labels"])
    return {"accuracy": correct / total, "correct": correct, "n": total, "matmul_flops_per_sample": flops / total}


def save_state(model, path):
    temp = path.with_suffix(".tmp")
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, temp)
    temp.replace(path)


def load_state(model, path):
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))


def configure_stage(model, stage):
    for p in model.parameters():
        p.requires_grad_(False)
    model.train()
    if stage == 1:
        for part in (model.bert, model.classifier):
            for p in part.parameters():
                p.requires_grad_(True)
    elif stage == 3:
        model.bert.eval()
        model.dropout.eval()
        for p in model.routers.parameters():
            p.requires_grad_(True)


def resource_smoke(cfg, out, data):
    seed_all(cfg["seed"])
    model = make_model(cfg)
    records = []
    batch = next(batches(data["train"], cfg["train_batch_size"]))
    for stage in (1, 3):
        configure_stage(model, stage)
        params = [p for p in model.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(params, lr=cfg[f"stage{stage}_learning_rate"])
        torch.cuda.reset_peak_memory_stats()
        times = []
        for step in range(5):
            optim.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if stage == 1:
                    logits = model(batch["input_ids"], batch["attention_mask"])
                    loss = F.cross_entropy(logits, batch["labels"])
                else:
                    logits, _, budget = model.forward_with_routing(batch["input_ids"], batch["attention_mask"])
                    loss = F.cross_entropy(logits, batch["labels"]) + cfg["lambda_mod"] * budget
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["gradient_clip"])
            optim.step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite smoke loss")
        row = {"stage": stage, "step_seconds": times, "median_last4": float(np.median(times[1:])),
               "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30, "loss": float(loss.detach())}
        records.append(row)
        log(out, "resource_smoke", **row)
        del optim
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    write_json(out / "resource_smoke.json", records)
    del model
    torch.cuda.empty_cache()


def train_stage(model, cfg, out, data, stage):
    target = out / f"stage{stage}_best.pt"
    done = out / f"stage{stage}_complete.json"
    if done.exists():
        load_state(model, target)
        log(out, "stage_reuse", stage=stage)
        return
    configure_stage(model, stage)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg[f"stage{stage}_learning_rate"], weight_decay=cfg["weight_decay"])
    epochs = cfg[f"stage{stage}_epochs"]
    per_epoch = math.ceil(len(data["train"]["labels"]) / cfg["train_batch_size"])
    scheduler = get_linear_schedule_with_warmup(optim, int(per_epoch * epochs * cfg["warmup_ratio"]), per_epoch * epochs)
    best = -1.0
    start_epoch = 0
    history = []
    resume_path = out / f"stage{stage}_resume.pt"
    if resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best, history = state["epoch"], state["best"], state["history"]
        del state
        log(out, "stage_resume", stage=stage, epoch=start_epoch)
    for epoch in range(start_epoch, epochs):
        seed_all(cfg["seed"] + stage * 100 + epoch)
        configure_stage(model, stage)
        order = torch.randperm(len(data["train"]["labels"]))
        total_loss = 0.0
        total_task = 0.0
        start = time.perf_counter()
        for step, batch in enumerate(batches(data["train"], cfg["train_batch_size"], order), 1):
            optim.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if stage == 1:
                    logits = model(batch["input_ids"], batch["attention_mask"])
                    task = F.cross_entropy(logits, batch["labels"])
                    loss = task
                else:
                    logits, _, budget = model.forward_with_routing(batch["input_ids"], batch["attention_mask"])
                    task = F.cross_entropy(logits, batch["labels"])
                    loss = task + cfg["lambda_mod"] * budget
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss stage={stage} epoch={epoch} step={step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["gradient_clip"])
            optim.step()
            scheduler.step()
            total_loss += float(loss.detach())
            total_task += float(task.detach())
            if step % 100 == 0 or step == per_epoch:
                log(out, "train_progress", stage=stage, epoch=epoch+1, step=step, steps=per_epoch,
                    mean_loss=total_loss/step, mean_task_loss=total_task/step,
                    elapsed_seconds=round(time.perf_counter()-start, 2))
        metric = accuracy(model, data["validation"], mode="dense" if stage == 1 else "query")
        row = {"epoch": epoch+1, "loss": total_loss/per_epoch, "task_loss": total_task/per_epoch,
               "validation": metric, "seconds": time.perf_counter()-start}
        history.append(row)
        if metric["accuracy"] > best:
            best = metric["accuracy"]
            save_state(model, target)
        write_json(out / f"stage{stage}_history.json", history)
        log(out, "epoch_complete", stage=stage, **row)
        temp = resume_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optim.state_dict(), "scheduler": scheduler.state_dict(),
                    "epoch": epoch+1, "best": best, "history": history}, temp)
        temp.replace(resume_path)
    write_json(done, {"best_validation_accuracy": best, "epochs": epochs})
    load_state(model, target)


@torch.inference_mode()
def cache_offramp_features(model, cfg, out, data, name):
    path = out / f"offramp_features_{name}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    model.eval()
    result = []
    start = time.perf_counter()
    for i, batch in enumerate(batches(data[name], cfg["train_batch_size"])):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=name == "train"):
            h = model.bert.embeddings(input_ids=batch["input_ids"])
            mask = model.bert.get_extended_attention_mask(batch["attention_mask"], batch["input_ids"].shape)
            parts = []
            for layer in model.bert.encoder.layer[:cfg["split_layer"]]:
                h = layer(h, mask)[0]
                parts.append(h[:, 0])
            result.append(torch.stack(parts, dim=1).cpu())
        if (i+1) % 250 == 0:
            log(out, "feature_cache_progress", split=name, batches=i+1, seconds=time.perf_counter()-start)
    features = torch.cat(result)
    torch.save(features, path)
    log(out, "feature_cache_complete", split=name, shape=list(features.shape), dtype=str(features.dtype))
    return features


def train_offramps(model, cfg, out, data):
    if (out / "stage2_complete.json").exists():
        load_state(model, out / "stage2_best.pt")
        return
    train_x = cache_offramp_features(model, cfg, out, data, "train").clone()
    valid_x = cache_offramp_features(model, cfg, out, data, "validation").clone()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.offramp_classifiers.parameters():
        p.requires_grad_(True)
    params = list(model.offramp_classifiers.parameters())
    optim = torch.optim.AdamW(params, lr=cfg["stage2_learning_rate"], weight_decay=cfg["weight_decay"])
    steps = math.ceil(len(train_x)/cfg["train_batch_size"])
    scheduler = get_linear_schedule_with_warmup(optim, int(steps*cfg["stage2_epochs"]*cfg["warmup_ratio"]), steps*cfg["stage2_epochs"])
    best = float("inf")
    history = []
    for epoch in range(cfg["stage2_epochs"]):
        seed_all(cfg["seed"]+200+epoch)
        order = torch.randperm(len(train_x))
        model.eval()
        model.dropout.train()
        total = 0.0
        for start in range(0, len(order), cfg["train_batch_size"]):
            idx = order[start:start+cfg["train_batch_size"]]
            x = train_x[idx].cuda()
            y = data["train"]["labels"][idx].cuda()
            optim.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = sum(F.cross_entropy(head(model.dropout(x[:, j])), y)
                           for j, head in enumerate(model.offramp_classifiers)) / cfg["split_layer"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["gradient_clip"])
            optim.step()
            scheduler.step()
            total += float(loss.detach())
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for start in range(0, len(valid_x), 256):
                x = valid_x[start:start+256].float().cuda()
                y = data["validation"]["labels"][start:start+256].cuda()
                val_loss += sum(float(F.cross_entropy(head(x[:,j]), y, reduction="sum"))
                                for j,head in enumerate(model.offramp_classifiers)) / cfg["split_layer"]
        val_loss /= len(valid_x)
        row = {"epoch": epoch+1, "train_loss": total/steps, "validation_loss": val_loss}
        history.append(row)
        log(out, "stage2_epoch", **row)
        if val_loss < best:
            best = val_loss
            save_state(model, out / "stage2_best.pt")
        write_json(out / "stage2_history.json", history)
    write_json(out / "stage2_complete.json", {"best_validation_loss": best, "feature_cache_precision": "train BF16; validation FP32"})
    load_state(model, out / "stage2_best.pt")


def select_threshold(model, cfg, out, data):
    if (out / "selection.json").exists():
        return
    load_state(model, out / "stage1_best.pt")
    dense = accuracy(model, data["validation"], batch_size=1)
    load_state(model, out / "stage3_best.pt")
    records = []
    for threshold in cfg["threshold_candidates"]:
        metric = accuracy(model, data["validation"], mode="query", batch_size=1, threshold=threshold)
        row = {"entropy_threshold": threshold, **metric}
        records.append(row)
        log(out, "threshold_validation", **row)
        write_json(out / "selection_progress.json", records)
    qualifying = [x for x in records if x["accuracy"] >= dense["accuracy"]-cfg["maximum_validation_accuracy_drop"]]
    if qualifying:
        best = min(qualifying, key=lambda x: (x["matmul_flops_per_sample"], -x["accuracy"]))
    else:
        best = min(records, key=lambda x: (-x["accuracy"], x["matmul_flops_per_sample"]))
    write_json(out / "selection.json", {**best, "constraint_met": bool(qualifying), "dense_validation": dense,
               "candidates": records, "rule": cfg["selection_rule"], "selection_split": "validation"})
    log(out, "selection_complete", **best, constraint_met=bool(qualifying))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/latency_20261108.json")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--phase", choices=["prepare", "smoke", "train", "select", "all"], default="all")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    out = Path(args.run_dir) if args.run_dir else ROOT / cfg["output_root"]
    out.mkdir(parents=True, exist_ok=True)
    environment(cfg, out)
    cfg = prepare(cfg, out)
    if args.phase == "prepare":
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 required for declared training")
    data = torch.load(out / "data.pt", weights_only=False, map_location="cpu")
    if args.phase in ("smoke", "all"):
        resource_smoke(cfg, out, data)
    if args.phase == "smoke":
        return
    seed_all(cfg["seed"])
    model = make_model(cfg)
    if args.phase in ("train", "all"):
        log(out, "training_started")
        train_stage(model, cfg, out, data, 1)
        train_offramps(model, cfg, out, data)
        train_stage(model, cfg, out, data, 3)
        log(out, "training_complete")
    if args.phase in ("select", "all"):
        select_threshold(model, cfg, out, data)
    log(out, "phase_complete", phase=args.phase)


if __name__ == "__main__":
    main()
