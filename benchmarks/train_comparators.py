"""Train standalone RouterBERT/DeeBERT additions on the paired frozen backbone.

All task-model/split selection comes from the completed supplementary run. This
module never reads test labels. Memory-mapped data/checkpoints and feature caches
keep host-memory use bounded on the 16 GB experimental laptop.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, get_linear_schedule_with_warmup

from benchmarks.hdc_runtime import HDCBertRuntime
from benchmarks.run_experiment import environment, log, seed_all, write_json


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frozen_hash(model):
    h = hashlib.sha256()
    count = 0
    for key, value in sorted(model.state_dict().items()):
        if key.startswith(("bert.", "classifier.")):
            h.update(key.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
            count += 1
    return {"sha256": h.hexdigest(), "tensors": count}


def save_state(model, path, prefix=None):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()
                if prefix is None or k.startswith(prefix)}, temporary)
    temporary.replace(path)


def batches(data, size, device="cuda", order=None):
    if order is None:
        order = torch.arange(len(data["labels"]))
    for start in range(0, len(order), size):
        indices = order[start:start + size]
        yield {k: data[k][indices].to(device) for k in ("input_ids", "attention_mask", "labels")}


def create_model(cfg, source_cfg, source_run, kind, device="cuda"):
    seed_all(cfg["seed"])
    bert_cfg = AutoConfig.from_pretrained(source_cfg["model_name"], local_files_only=True)
    model = HDCBertRuntime(config=bert_cfg, num_labels=cfg["num_labels"],
                           dropout=cfg["dropout"],
                           split_layer=0 if kind == "router" else bert_cfg.num_hidden_layers,
                           tau=cfg["tau"], target_keep_ratio=cfg["router_target_keep_ratio"])
    # Loading only the shared backbone/head excludes source HDC offramps/routers.
    state = torch.load(source_run / "stage1_best.pt", mmap=True, weights_only=True, map_location="cpu")
    shared = {k: v for k, v in state.items() if k.startswith(("bert.", "classifier."))}
    missing, unexpected = model.load_state_dict(shared, strict=False)
    if unexpected or any(not k.startswith(("offramp_classifiers.", "routers.")) for k in missing):
        raise RuntimeError(f"Unexpected shared checkpoint mapping: {missing}, {unexpected}")
    if kind == "deebert":
        model.offramp_classifiers[-1].load_state_dict(model.classifier.state_dict())
    del shared, state
    model.to(device)
    configure(model, kind)
    return model


def configure(model, kind):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if kind == "router":
        # The training forward uses hard STE query gates. The frozen backbone
        # stays in eval mode so attention/hidden dropout cannot change semantics.
        model.training = True
        for parameter in model.routers.parameters():
            parameter.requires_grad_(True)
    else:
        for head in model.offramp_classifiers[:-1]:
            for parameter in head.parameters():
                parameter.requires_grad_(True)
        model.dropout.train()


@torch.inference_mode()
def router_validation(model, data, batch_size=32, device="cuda"):
    model.eval()
    # Select using dense hard query gates, mathematically equal at nonpadding
    # positions to query-sparse inference, without its per-sample launch costs.
    model.training = True
    correct, count, loss = 0, 0, 0.0
    for batch in batches(data, batch_size, device):
        logits, _, _ = model.forward_with_routing(batch["input_ids"], batch["attention_mask"])
        correct += int((logits.argmax(-1) == batch["labels"]).sum())
        loss += float(F.cross_entropy(logits, batch["labels"], reduction="sum"))
        count += len(batch["labels"])
    model.eval()
    return {"accuracy": correct / count, "correct": correct, "n": count,
            "cross_entropy": loss / count, "path": "FP32 dense hard query gate"}


def optimizer_for(model, cfg, steps):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(steps * cfg["epochs"] * cfg["warmup_ratio"]), steps * cfg["epochs"])
    return params, optimizer, scheduler


def train_router(model, cfg, out, data):
    if (out / "router_complete.json").exists():
        log(out, "router_already_complete")
        return
    initial_hash = frozen_hash(model)
    steps = math.ceil(len(data["train"]["labels"]) / cfg["train_batch_size"])
    configure(model, "router")
    params, optim, scheduler = optimizer_for(model, cfg, steps)
    history, best, start_epoch = [], -1.0, 0
    resume = out / "router_resume.pt"
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.routers.load_state_dict(state["routers"])
        optim.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best, start_epoch = state["history"], state["best"], state["epoch"]
        del state
    for epoch in range(start_epoch, cfg["epochs"]):
        seed_all(cfg["seed"] + 300 + epoch)
        configure(model, "router")
        order = torch.randperm(len(data["train"]["labels"]))
        total_loss, total_task, total_budget = 0.0, 0.0, 0.0
        start = time.perf_counter()
        for step, batch in enumerate(batches(data["train"], cfg["train_batch_size"], order=order), 1):
            optim.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _, budget = model.forward_with_routing(batch["input_ids"], batch["attention_mask"])
                task = F.cross_entropy(logits, batch["labels"])
                loss = task + cfg["lambda_mod"] * budget
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite router training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["gradient_clip"])
            optim.step()
            scheduler.step()
            total_loss += float(loss.detach())
            total_task += float(task.detach())
            total_budget += float(budget.detach())
            if step % 100 == 0 or step == steps:
                log(out, "router_progress", epoch=epoch + 1, step=step, steps=steps,
                    loss=total_loss / step, task_loss=total_task / step,
                    elapsed_seconds=round(time.perf_counter() - start, 2))
        validation = router_validation(model, data["validation"], cfg["train_batch_size"])
        row = {"epoch": epoch + 1, "loss": total_loss / steps,
               "task_loss": total_task / steps, "budget_loss": total_budget / steps,
               "validation": validation, "seconds": time.perf_counter() - start}
        history.append(row)
        save_state(model, out / f"router_epoch{epoch + 1}_routers.pt", "routers.")
        if validation["accuracy"] > best:
            best = validation["accuracy"]
            save_state(model, out / "router_best.pt")
        write_json(out / "router_history.json", history)
        log(out, "router_epoch_complete", **row)
        tmp = resume.with_suffix(".tmp")
        torch.save({"routers": model.routers.state_dict(), "optimizer": optim.state_dict(),
                    "scheduler": scheduler.state_dict(), "history": history,
                    "best": best, "epoch": epoch + 1}, tmp)
        tmp.replace(resume)
    final_hash = frozen_hash(model)
    if initial_hash != final_hash:
        raise RuntimeError("Frozen backbone changed during router training")
    selected_epoch = max(history, key=lambda x: x["validation"]["accuracy"])["epoch"]
    write_json(out / "router_complete.json", {"best_validation_accuracy": best,
               "selected_epoch": selected_epoch, "epochs": cfg["epochs"],
               "frozen_backbone": final_hash, "split_layer": 0,
               "target_keep_ratio": cfg["router_target_keep_ratio"],
               "selection": "validation accuracy; earliest epoch on ties"})


@torch.inference_mode()
def feature_cache(model, cfg, out, data, split):
    path = out / f"deebert_features_{split}.npy"
    done = out / f"deebert_features_{split}_complete.json"
    shape = (len(data[split]["labels"]), model.n_layers - 1, model.bert.config.hidden_size)
    if done.exists():
        info = json.loads(done.read_text())
        if info["shape"] != list(shape):
            raise RuntimeError("Feature cache shape mismatch")
        return np.load(path, mmap_mode="r")
    # Never concatenate all features in RAM (eleven offramps are ~3.6 GiB).
    features = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    model.eval()
    offset = 0
    start = time.perf_counter()
    for index, batch in enumerate(batches(data[split], cfg["train_batch_size"])):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=split == "train"):
            hidden = model.bert.embeddings(input_ids=batch["input_ids"])
            mask = model.bert.get_extended_attention_mask(batch["attention_mask"], batch["input_ids"].shape)
            for layer_index, layer in enumerate(model.bert.encoder.layer[:-1]):
                hidden = layer(hidden, mask)[0]
                features[offset:offset + len(hidden), layer_index] = hidden[:, 0].float().cpu().numpy()
        offset += len(hidden)
        if (index + 1) % 250 == 0:
            features.flush()
            log(out, "deebert_cache_progress", split=split, samples=offset,
                elapsed_seconds=round(time.perf_counter() - start, 2))
    features.flush()
    del features
    write_json(done, {"shape": list(shape), "storage_dtype": "float32",
                     "compute_dtype": "bfloat16" if split == "train" else "float32",
                     "seconds": time.perf_counter() - start})
    return np.load(path, mmap_mode="r")


def train_deebert(model, cfg, out, data):
    if (out / "deebert_complete.json").exists():
        log(out, "deebert_already_complete")
        return
    initial_hash = frozen_hash(model)
    train_x = feature_cache(model, cfg, out, data, "train")
    valid_x = feature_cache(model, cfg, out, data, "validation")
    configure(model, "deebert")
    steps = math.ceil(len(train_x) / cfg["train_batch_size"])
    params, optim, scheduler = optimizer_for(model, cfg, steps)
    history, best, start_epoch = [], float("inf"), 0
    resume = out / "deebert_resume.pt"
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.offramp_classifiers.load_state_dict(state["heads"])
        optim.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best, start_epoch = state["history"], state["best"], state["epoch"]
        del state
    n_heads = model.n_layers - 1
    for epoch in range(start_epoch, cfg["epochs"]):
        seed_all(cfg["seed"] + 200 + epoch)
        configure(model, "deebert")
        order = torch.randperm(len(train_x))
        total, start = 0.0, time.perf_counter()
        for start_index in range(0, len(order), cfg["train_batch_size"]):
            idx = order[start_index:start_index + cfg["train_batch_size"]]
            # Advanced indexing copies only this small batch, never the full file.
            x = torch.from_numpy(np.array(train_x[idx.numpy()], copy=True)).cuda()
            y = data["train"]["labels"][idx].cuda()
            optim.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = sum(F.cross_entropy(head(model.dropout(x[:, j])), y)
                           for j, head in enumerate(model.offramp_classifiers[:-1])) / n_heads
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite DeeBERT training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["gradient_clip"])
            optim.step()
            scheduler.step()
            total += float(loss.detach())
        model.eval()
        val_loss, val_correct = 0.0, [0] * n_heads
        with torch.inference_mode():
            for start_index in range(0, len(valid_x), 256):
                x = torch.from_numpy(np.array(valid_x[start_index:start_index + 256], copy=True)).cuda()
                y = data["validation"]["labels"][start_index:start_index + 256].cuda()
                for j, head in enumerate(model.offramp_classifiers[:-1]):
                    logits = head(x[:, j])
                    val_loss += float(F.cross_entropy(logits, y, reduction="sum")) / n_heads
                    val_correct[j] += int((logits.argmax(-1) == y).sum())
        val_loss /= len(valid_x)
        row = {"epoch": epoch + 1, "train_loss": total / steps, "validation_loss": val_loss,
               "per_exit_validation_accuracy": [x / len(valid_x) for x in val_correct],
               "seconds": time.perf_counter() - start}
        history.append(row)
        save_state(model, out / f"deebert_epoch{epoch + 1}_heads.pt", "offramp_classifiers.")
        if val_loss < best:
            best = val_loss
            save_state(model, out / "deebert_best.pt")
        write_json(out / "deebert_history.json", history)
        log(out, "deebert_epoch_complete", **row)
        tmp = resume.with_suffix(".tmp")
        torch.save({"heads": model.offramp_classifiers.state_dict(), "optimizer": optim.state_dict(),
                    "scheduler": scheduler.state_dict(), "history": history, "best": best,
                    "epoch": epoch + 1}, tmp)
        tmp.replace(resume)
    final_hash = frozen_hash(model)
    if initial_hash != final_hash:
        raise RuntimeError("Frozen backbone changed during DeeBERT training")
    for name, value in model.classifier.state_dict().items():
        if not torch.equal(value, model.offramp_classifiers[-1].state_dict()[name]):
            raise RuntimeError("DeeBERT final exit changed")
    write_json(out / "deebert_complete.json", {"best_validation_loss": best,
               "selected_epoch": min(history, key=lambda x: x["validation_loss"])["epoch"],
               "epochs": cfg["epochs"], "split_layer": model.n_layers,
               "trainable_exits": n_heads, "final_exit": "frozen dense classifier copy",
               "frozen_backbone": final_hash, "selection": "mean validation CE over first 11 exits"})


def smoke(model, cfg, out, data):
    before = frozen_hash(model)
    configure(model, "router")
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg["learning_rate"])
    batch = next(batches(data["train"], cfg["train_batch_size"]))
    durations = []
    for _ in range(3):
        optim.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _, budget = model.forward_with_routing(batch["input_ids"], batch["attention_mask"])
            loss = F.cross_entropy(logits, batch["labels"]) + cfg["lambda_mod"] * budget
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg["gradient_clip"])
        optim.step()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
        if not torch.isfinite(loss):
            raise RuntimeError("Smoke loss nonfinite")
    if before != frozen_hash(model):
        raise RuntimeError("Smoke modified frozen backbone")
    model.eval()
    model.training = True
    with torch.inference_mode():
        dense_gate = model.forward_with_routing(batch["input_ids"], batch["attention_mask"])[0]
        model.eval()
        sparse = model.forward_dynamic(batch["input_ids"], batch["attention_mask"], -1, "query")[0]
        torch.testing.assert_close(dense_gate, sparse, atol=3e-4, rtol=3e-4)
    record = {"step_seconds": durations, "loss": float(loss.detach()),
              "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "dense_query_max_abs_difference": float((dense_gate - sparse).abs().max()),
              "frozen_backbone": before, "result": "passed"}
    write_json(out / "training_smoke.json", record)
    log(out, "training_smoke_complete", **record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/comparator_audit_20261109.json")
    parser.add_argument("--source-run")
    parser.add_argument("--run-dir")
    parser.add_argument("--phase", choices=("smoke", "router", "deebert", "all"), default="all")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    source = (ROOT / (args.source_run or cfg["source_run"])).resolve()
    out = (ROOT / (args.run_dir or cfg["output_root"])).resolve()
    if source == out:
        raise RuntimeError("Comparator outputs must not overwrite the source run")
    out.mkdir(parents=True, exist_ok=True)
    source_cfg = json.loads((source / "config_resolved.json").read_text(encoding="utf-8"))
    manifest = {"config": cfg, "source_checkpoint_sha256": file_hash(source / "stage1_best.pt"),
                "source_data_sha256": file_hash(source / "data.pt"),
                "source_data_manifest": json.loads((source / "data_manifest.json").read_text()),
                "training_source_sha256": file_hash(__file__),
                "dependency_sources_sha256": {name: file_hash(ROOT / name) for name in (
                    "benchmarks/hdc_runtime.py", "models/hdc_bert.py",
                    "models/router_tuning_bert.py", "benchmarks/run_experiment.py")},
                "source_checkpoint_mapping": "Only bert.* and classifier.* tensors copied strictly by key/shape; router zero-init; DeeBERT head12 copies frozen classifier",
                "split_policy": "exact source train/validation tensors; no test selection"}
    manifest_path = out / "training_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("Training identity changed; use a fresh output directory")
    write_json(manifest_path, manifest)
    write_json(out / "training_config_resolved.json", {**cfg, "model_name": source_cfg["model_name"]})
    environment(cfg, out)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required for this declared experiment")
    # mmap avoids loading 250 MB of padded tensors into an additional allocation.
    packed = torch.load(source / "data.pt", map_location="cpu", mmap=True, weights_only=False)
    data = {name: {k: packed[name][k] for k in ("input_ids", "attention_mask", "labels")}
            for name in ("train", "validation")}
    del packed
    phases = ("router", "deebert") if args.phase == "all" else (args.phase,)
    for phase in phases:
        model = create_model(cfg, source_cfg, source, "router" if phase == "smoke" else phase)
        log(out, "comparator_started", phase=phase, frozen_backbone=frozen_hash(model))
        if phase == "smoke":
            smoke(model, cfg, out, data)
        elif phase == "router":
            train_router(model, cfg, out, data)
        else:
            train_deebert(model, cfg, out, data)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    log(out, "comparator_training_complete", phase=args.phase)


if __name__ == "__main__":
    main()
