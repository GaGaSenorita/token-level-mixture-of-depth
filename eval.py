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
    Router-Tuning model evaluation.
    forward() now returns logits only as a drop-in replacement.
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
    Full Router-Tuning evaluation returning accuracy plus per-layer keep rates.

    keep_rate comes from mask_hard on the inference path (the true skip ratio)
    and can be used directly for:
      - reporting the actual keep ratio of each layer in the report
      - later FLOPs estimation (FLOPs_saved ∝ 1 - keep_rate_i)

    Returns:
        acc            : float, test accuracy
        eval_keep_rates: List[float|None], length=n_layers, average keep_rate
                         for each layer. None means the layer is NoRouter
                         (routing is disabled).
        avg_keep_rate  : float, the mean keep_rate across all routed layers
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

            # Use forward_with_routing to obtain router_stats, including each layer's keep_rate
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
    Recommendation: use batch_size=1 so that batchwise matches samplewise
    behaviour and stays aligned with the paper's semantics.
    fn_name:
        - "forward_early_exit_batchwise"  (the current implementation)
        - if you later add a samplewise version, pass "forward_early_exit_samplewise"
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
        exit_layers.append(int(exited_layer)) # With batch_size=1, len(exit_layers) == total

    acc = correct / max(1, total)
    avg_exit_layer = sum(exit_layers) / max(1, len(exit_layers))
    exit_hist = dict(sorted(Counter(exit_layers).items(), key=lambda x: x[0]))

    return acc, avg_exit_layer, exit_hist


# ====================== HDC Evaluation ======================

def evaluate_hdc_routing_full(model, dataloader, device):
    """
    Full HDC Stage 3 evaluation: routing accuracy plus eval-time keep_rate
    for each Stage B layer. Uses forward_with_routing() in model.eval() mode
    on the inference path with token skipping.

    Returns:
        acc             : float, routing inference accuracy
        eval_keep_rates : List[float], length = n_stage_b_layers
                           (Stage B keep_rate for each layer)
        avg_keep_rate   : float, average Stage B keep_rate
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
    HDC model evaluation with routing-enabled accuracy.
    Calls forward_with_routing() and is used during Stage 3 training.
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
    Full HDC inference evaluation: Stage A early exit plus Stage B token routing.
    Recommendation: use a dataloader with batch_size=1 to match
    evaluate_early_exit.

    Returns:
        dict with:
            'accuracy':           HDC inference accuracy
            'stage_a_exit_rate':  proportion of samples that exit in Stage A
            'avg_exit_layer_a':   average Stage A exit layer (1-indexed)
            'avg_keep_rate_b':    average token keep rate in Stage B
            'exit_histogram':     exit distribution across layers
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

    # Average exit layer for Stage A
    stage_a_layers = [el for el in exit_layers if isinstance(el, int)]
    avg_exit_layer_a = (
        sum(stage_a_layers) / max(1, len(stage_a_layers))
        if stage_a_layers else 0.0
    )

    # Average keep_rate for Stage B
    avg_keep_rate_b = (
        sum(keep_rate_accum) / max(1, len(keep_rate_accum))
        if keep_rate_accum else 0.0
    )

    # Exit distribution
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
