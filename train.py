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

    optimizer = AdamW(model.parameters(), lr=args.stage1_learning_rate)
    scheduler = _build_scheduler(optimizer, len(train_loader), args.stage1_epochs, warmup_ratio=0.1)

    history = {"Deebert_step1_train_loss": [], "Deebert_step1_test_acc": []}
    best_acc = 0.0

    for epoch in range(args.stage1_epochs):
        print(f"\n===== DeeBert Step1: Epoch {epoch+1}/{args.stage1_epochs} =====")

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

    return history, best_path


def train_step2_deebert(model, train_loader, test_loader, test_loader_ee, args, device, entropy_threshold=0.2, eval_early_exit=True):
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
        lr=args.stage2_learning_rate
    )
    print("===== Trainable parameters =====")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
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
                model, test_loader_ee, device,
                entropy_threshold=entropy_threshold,
                fn_name="forward_early_exit_batchwise"
            )
            history["step2_test_acc_ee"].append(ee_acc)
            history["step2_avg_exit_layer"].append(avg_exit_layer)

            print(f"Step2 early-exit acc (thr={entropy_threshold}): {ee_acc:.4f}")
            print(f"Step2 avg exit layer (thr={entropy_threshold}): {avg_exit_layer:.2f}")
            print(f"Exit histogram: {exit_hist}")

    return history


def train_epoch_router_tuning(model, dataloader, optimizer, scheduler, device, lambda_mod):
    """
    Router-Tuning: 训练一个 epoch，返回 loss 和 keep_rates
    """
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_task = 0.0
    total_mod = 0.0
    keep_rate_sums = None
    keep_rate_counts = None

    for batch in tqdm(dataloader, desc="Train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, router_stats, l_mod = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        loss_task = loss_fn(logits, labels)
        loss = loss_task + lambda_mod * l_mod

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_task += loss_task.item()
        total_mod += l_mod.item()

        keep_rates = router_stats.get("keep_rates", [])
        if keep_rate_sums is None:
            keep_rate_sums = [0.0 for _ in keep_rates]
            keep_rate_counts = [0 for _ in keep_rates]
        for i, kr in enumerate(keep_rates):
            if kr is not None:
                keep_rate_sums[i] += float(kr)
                keep_rate_counts[i] += 1

    avg_loss = total_loss / max(1, len(dataloader))
    avg_task = total_task / max(1, len(dataloader))
    avg_mod = total_mod / max(1, len(dataloader))

    keep_rate_avgs = None
    if keep_rate_sums is not None:
        keep_rate_avgs = []
        for s, c in zip(keep_rate_sums, keep_rate_counts):
            if c == 0:
                keep_rate_avgs.append(None)
            else:
                keep_rate_avgs.append(s / c)

    return avg_loss, avg_task, avg_mod, keep_rate_avgs


def train_router_tuning(model, train_loader, test_loader, args, device):
    """
    Router-Tuning 训练主循环
    """
    from eval import evaluate_router

    os.makedirs(args.output_dir, exist_ok=True)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = {
        "train_loss": [],
        "train_loss_task": [],
        "train_loss_mod": [],
        "test_acc": [],
        "keep_rates": [],
    }
    best_acc = 0.0
    best_path = os.path.join(args.output_dir, "best_model.pt")

    for epoch in range(args.epochs):
        print(f"\n===== Router-Tuning: Epoch {epoch + 1}/{args.epochs} =====")

        train_loss, train_task, train_mod, keep_rate_avgs = train_epoch_router_tuning(
            model, train_loader, optimizer, scheduler, device, args.lambda_mod
        )
        test_acc = evaluate_router(model, test_loader, device)

        history["train_loss"].append(train_loss)
        history["train_loss_task"].append(train_task)
        history["train_loss_mod"].append(train_mod)
        history["test_acc"].append(test_acc)
        history["keep_rates"].append(keep_rate_avgs)

        print(f"Train loss (total): {train_loss:.4f}")
        print(f"Train loss (task):  {train_task:.4f}")
        print(f"Train loss (MoD):   {train_mod:.4f}")
        print(f"Test acc:           {test_acc:.4f}")

        if keep_rate_avgs is not None:
            routed_keep = {i: kr for i, kr in enumerate(keep_rate_avgs) if kr is not None}
            print(f"Keep rates: {routed_keep}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)
            print(f"Best model saved: {best_path} (acc={best_acc:.4f})")

    print(f"\nRouter-Tuning finished. Best acc = {best_acc:.4f}")
    return history
