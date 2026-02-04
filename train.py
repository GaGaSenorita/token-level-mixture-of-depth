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


def _build_scheduler(optimizer, steps_per_epoch, epochs, warmup_ratio=0.1):
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def train_step1_deebert(model, train_loader, test_loader, args, device):
    """
    Step1: Dense BERT fine-tuning
    - 训练 backbone + last head
    - 用你原来的 train_epoch / evaluate（评估 last head）
    """
    from eval import evaluate
    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "best_model_step1.pt")

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = _build_scheduler(optimizer, len(train_loader), args.epochs, warmup_ratio=0.1)

    history = {"Deebert_step1_train_loss": [], "Deebert_step1_test_acc": []}
    best_acc = 0.0

    for epoch in range(args.epochs):
        print(f"\n===== DeeBert Step1: Epoch {epoch+1}/{args.epochs} =====")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        test_acc = evaluate(model, test_loader, device)

        history["Deebert_step1_train_loss"].append(train_loss)
        history["Deebert_step1_test_acc"].append(test_acc)
        print(f"Train loss: {train_loss:.4f}")
        print(f"Test  acc : {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)
            print(f"Best DeeBert step1 model saved: {best_path} (acc={best_acc:.4f})")

    return history


def train_step2_deebert(model, train_loader, test_loader, args, device, entropy_threshold=0.2, eval_early_exit=True):
    """
    Step2: DeeBERT off-ramps training
    - freeze backbone + last head
    - 只训练中间 heads (1..n-1)
    - 注意：evaluate(model, ...) 评的是 last head（冻结了），所以它涨不涨不代表 step2 没效果
      真正要看的是 early-exit 的指标（你后面可以再加一个 evaluate_early_exit）
    """
    # 冻结 backbone 和最后一个 head（你在 deebert.py 里已经写了这个函数）
    from eval import evaluate, evaluate_early_exit
    model.freeze_backbone_and_last_head()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate_stage2
    )
    scheduler = _build_scheduler(optimizer, len(train_loader), args.stage2_epochs, warmup_ratio=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    history = {"step2_train_loss": [], "step2_test_acc_last": []}
    if eval_early_exit:
        history["step2_test_acc_ee"] = []
        history["step2_avg_exit_layer"] = []

    for epoch in range(args.stage2_epochs):
        print(f"\n===== DeeBert Step2: Epoch {epoch+1}/{args.stage2_epochs} =====")
        model.train()

        total_loss = 0.0
        for batch in tqdm(train_loader, desc="Step2 Train", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            logits_list = model.forward_all_exits(
                input_ids=input_ids,
                attention_mask=attention_mask
            )  # list length = n_layers, each [B,C]

            # 只训前 n-1 个 head
            loss = 0.0
            for logits in logits_list[:-1]:
                loss = loss + loss_fn(logits, labels)

            # 稳一点：做平均（不然 head 数多了 loss 尺度会变）
            loss = loss / max(1, (len(logits_list) - 1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))
        history["step2_train_loss"].append(avg_train_loss)

        # ---- Eval (last head acc, note: last head is frozen) ----
        test_acc_last = evaluate(model, test_loader, device)
        history["step2_test_acc_last"].append(test_acc_last)

        print(f"Step2 train loss: {avg_train_loss:.4f}")
        print(f"Step2 test acc (last head, frozen): {test_acc_last:.4f}")

        # ---- Optional: early-exit eval (DeeBERT key metric) ----
        if eval_early_exit:
            from eval import evaluate_early_exit
            ee_acc, avg_exit_layer, exit_hist = evaluate_early_exit(
                model, test_loader, device,
                entropy_threshold=entropy_threshold,
                fn_name="forward_early_exit_batchwise"  # 你现在的实现名
            )
            history["step2_test_acc_ee"].append(ee_acc)
            history["step2_avg_exit_layer"].append(avg_exit_layer)

            print(f"Step2 early-exit acc (thr={entropy_threshold}): {ee_acc:.4f}")
            print(f"Step2 avg exit layer (thr={entropy_threshold}): {avg_exit_layer:.2f}")
            print(f"Exit histogram: {exit_hist}")

    return history