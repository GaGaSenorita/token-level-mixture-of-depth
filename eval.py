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
