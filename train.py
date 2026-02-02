"""
训练模块：标准 supervised fine-tuning
与模型/数据解耦，可直接复用
"""
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """训练一个 epoch，返回平均 loss"""
    model.train()
    total_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss()

    for batch in tqdm(dataloader, desc="Train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


def train(model, train_loader, test_loader, args, device):

    from eval import evaluate

    os.makedirs(args.output_dir, exist_ok=True)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = {"train_loss": [], "test_acc": []}
    best_acc = 0.0
    best_path = os.path.join(args.output_dir, "best_model.pt")

    for epoch in range(args.epochs):
        print(f"\n===== Epoch {epoch + 1}/{args.epochs} =====")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        test_acc = evaluate(model, test_loader, device)

        history["train_loss"].append(train_loss)
        history["test_acc"].append(test_acc)

        print(f"Train loss: {train_loss:.4f}")
        print(f"Test acc:    {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)
            print(f"Best model saved: {best_path} (acc={best_acc:.4f})")

    print(f"\nTraining finished. Best acc = {best_acc:.4f}")
    return history


def train_deebert_stage2(model, train_loader, test_loader, args, device):
    """
    DeeBERT Stage 2:
    - backbone + last head frozen
    - train intermediate off-ramps
    """

    from eval import evaluate

    model.freeze_backbone_and_last_head()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate_stage2
    )

    total_steps = len(train_loader) * args.stage2_epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(args.stage2_epochs):
        print(f"\n===== DeeBERT Stage2 Epoch {epoch + 1}/{args.stage2_epochs} =====")
        model.train()

        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            logits_list = model.forward_all_exits(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # 只用前 n-1 个 off-ramps
            loss = 0.0
            for logits in logits_list[:-1]:
                loss = loss + loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        test_acc = evaluate(model, test_loader, device)

        print(f"Stage2 train loss: {avg_loss:.4f}")
        print(f"Stage2 test acc (last head): {test_acc:.4f}")
