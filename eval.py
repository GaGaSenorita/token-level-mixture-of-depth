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
