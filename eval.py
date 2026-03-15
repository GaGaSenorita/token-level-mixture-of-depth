import torch
from collections import Counter

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = logits.argmax(dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / max(1, total)


def evaluate_with_time(model, dataloader, device):
    """Returns (accuracy, inference_ms_per_sample). Called once on best checkpoint after training."""
    import time
    model.eval()
    correct = 0
    total = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            logits         = model(input_ids=input_ids, attention_mask=attention_mask)
            preds          = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    elapsed_ms = (time.time() - t0) * 1000
    acc = correct / max(1, total)
    return round(acc, 4), round(elapsed_ms / total, 4)


def evaluate_router(model, dataloader, device):
    """
    Router-Tuning 模型评估：
    forward() 现在只返回 logits（drop-in replacement）
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = logits.argmax(dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / max(1, total)


def evaluate_router_full(model, dataloader, device):
    """
    Router-Tuning 完整评估：返回 accuracy + 每层 keep_rate。

    keep_rate 来自推理路径的 mask_hard（真实跳过比例），可直接用于：
      - report 中汇报各层实际保留比例
      - 后续 FLOPs 估算（FLOPs_saved ∝ 1 - keep_rate_i）

    Returns:
        acc            : float，测试准确率
        eval_keep_rates: List[float|None]，长度=n_layers，每层平均 keep_rate
                         None 表示该层为 NoRouter（不做路由）
        avg_keep_rate  : float，所有路由层 keep_rate 的平均值
    """
    model.eval()
    correct = 0
    total = 0
    keep_rate_sums = None
    keep_rate_counts = None

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # 用 forward_with_routing 拿到 router_stats（含每层 keep_rate）
            logits, router_stats, _ = model.forward_with_routing(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            keep_rates = router_stats.get("keep_rates", [])
            if keep_rate_sums is None:
                keep_rate_sums = [0.0] * len(keep_rates)
                keep_rate_counts = [0] * len(keep_rates)
            for i, kr in enumerate(keep_rates):
                if kr is not None:
                    keep_rate_sums[i] += kr
                    keep_rate_counts[i] += 1

    acc = correct / max(1, total)

    eval_keep_rates = None
    avg_keep_rate = None
    if keep_rate_sums is not None:
        eval_keep_rates = [
            s / c if c > 0 else None
            for s, c in zip(keep_rate_sums, keep_rate_counts)
        ]
        valid = [kr for kr in eval_keep_rates if kr is not None]
        avg_keep_rate = sum(valid) / len(valid) if valid else None

    return acc, eval_keep_rates, avg_keep_rate


@torch.no_grad()
def evaluate_early_exit(model, dataloader, device, entropy_threshold, fn_name="forward_early_exit_batchwise"):
    '''
    建议：dataloader 用 batch_size=1，这样 batchwise 就等价 samplewise（与论文语义一致）。
    fn_name:
        - "forward_early_exit_batchwise"  (你现在的实现)
        - 如果你以后写 samplewise，可传 "forward_early_exit_samplewise"
    '''
    model.eval()
    correct = 0
    total = 0
    exit_layers = []

    if not hasattr(model, fn_name):
        raise AttributeError(f"Model has no method `{fn_name}`. "
                             f"Available: {[m for m in dir(model) if 'early_exit' in m]}")

    ee_forward = getattr(model, fn_name)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits, exited_layer = ee_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entropy_threshold=entropy_threshold
        )

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        exit_layers.append(int(exited_layer)) # 如果batch_size=1, len(exit_layers) == total

    acc = correct / max(1, total)
    avg_exit_layer = sum(exit_layers) / max(1, len(exit_layers))
    exit_hist = dict(sorted(Counter(exit_layers).items(), key=lambda x: x[0]))

    return acc, avg_exit_layer, exit_hist


# ====================== HDC Evaluation ======================

def evaluate_hdc_routing_full(model, dataloader, device):
    """
    HDC Stage 3 完整评估：routing accuracy + Stage B 每层 eval-time keep_rate。
    使用 forward_with_routing()，model.eval() 模式（推理路径，token skipping）。

    Returns:
        acc             : float，routing 推理准确率
        eval_keep_rates : List[float]，长度 = n_stage_b_layers（Stage B 每层 keep_rate）
        avg_keep_rate   : float，Stage B 平均 keep_rate
    """
    model.eval()
    correct = 0
    total = 0
    keep_rate_sums = None
    keep_rate_counts = None

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits, router_stats, _ = model.forward_with_routing(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            keep_rates = router_stats.get("keep_rates", [])
            if keep_rate_sums is None:
                keep_rate_sums = [0.0] * len(keep_rates)
                keep_rate_counts = [0] * len(keep_rates)
            for i, kr in enumerate(keep_rates):
                if kr is not None:
                    keep_rate_sums[i] += kr
                    keep_rate_counts[i] += 1

    acc = correct / max(1, total)
    eval_keep_rates = None
    avg_keep_rate = None
    if keep_rate_sums is not None:
        eval_keep_rates = [
            s / c if c > 0 else None
            for s, c in zip(keep_rate_sums, keep_rate_counts)
        ]
        valid = [kr for kr in eval_keep_rates if kr is not None]
        avg_keep_rate = sum(valid) / len(valid) if valid else None

    return acc, eval_keep_rates, avg_keep_rate


def evaluate_hdc(model, dataloader, device):
    """
    HDC 模型评估（带 routing 的 accuracy）。
    调用 forward_with_routing()，用于 Stage 3 训练期间。
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits, _, _ = model.forward_with_routing(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / max(1, total)


@torch.no_grad()
def evaluate_hdc_inference(model, dataloader, device, entropy_threshold=0.2):
    """
    完整 HDC 推理评估: Stage A early-exit + Stage B token routing。
    建议: 使用 batch_size=1 的 dataloader (与 evaluate_early_exit 一致)。

    Returns:
        dict with:
            'accuracy':           HDC 推理准确率
            'stage_a_exit_rate':  Stage A 提前退出比例
            'avg_exit_layer_a':   Stage A 平均退出层 (1-indexed)
            'avg_keep_rate_b':    Stage B 平均 token 保留率
            'exit_histogram':     各层退出分布
    """
    model.eval()
    correct = 0
    total = 0
    exit_layers = []
    stage_a_count = 0
    stage_b_count = 0
    keep_rate_accum = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits, exit_info = model.forward_hdc_inference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entropy_threshold=entropy_threshold,
        )

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        el = exit_info["exited_layer"]
        exit_layers.append(el)

        if exit_info["stage"] == "A":
            stage_a_count += labels.size(0)
        else:
            stage_b_count += labels.size(0)
            if exit_info["keep_rates"]:
                keep_rate_accum.append(
                    sum(exit_info["keep_rates"]) / len(exit_info["keep_rates"])
                )

    accuracy = correct / max(1, total)
    stage_a_exit_rate = stage_a_count / max(1, total)

    # Stage A 平均退出层
    stage_a_layers = [el for el in exit_layers if isinstance(el, int)]
    avg_exit_layer_a = (
        sum(stage_a_layers) / max(1, len(stage_a_layers))
        if stage_a_layers else 0.0
    )

    # Stage B 平均 keep_rate
    avg_keep_rate_b = (
        sum(keep_rate_accum) / max(1, len(keep_rate_accum))
        if keep_rate_accum else 0.0
    )

    # 退出分布
    exit_histogram = {}
    for el in exit_layers:
        key = str(el)
        exit_histogram[key] = exit_histogram.get(key, 0) + 1
    exit_histogram = dict(sorted(exit_histogram.items()))

    return {
        "accuracy": accuracy,
        "stage_a_exit_rate": stage_a_exit_rate,
        "avg_exit_layer_a": avg_exit_layer_a,
        "avg_keep_rate_b": avg_keep_rate_b,
        "exit_histogram": exit_histogram,
    }
