import torch

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

    return correct / total


@torch.no_grad()
def evaluate_early_exit(model, dataloader, device, entropy_threshold):
    model.eval()
    correct = 0
    total = 0
    exit_layers = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits, exited_layer = model.forward_early_exit(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entropy_threshold=entropy_threshold
        )

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        exit_layers.append(exited_layer)

    acc = correct / total
    avg_exit_layer = sum(exit_layers) / len(exit_layers)

    return acc, avg_exit_layer
