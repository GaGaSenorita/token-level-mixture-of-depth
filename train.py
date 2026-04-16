"""
Training module for standard supervised fine-tuning.
It is decoupled from the model and data code so it can be reused directly.
"""
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train one epoch and return the average loss."""
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
    - train the backbone + last head
    - reuse the original train_epoch / evaluate to assess the last head
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
    - train only the intermediate heads (1..n-1)
    - note: evaluate(model, ...) measures the frozen last head, so whether
      that score changes does not indicate whether Step 2 is effective.
      The real metrics to watch are the early-exit metrics.
    """
    # Freeze the backbone and the last head (this helper already exists in deebert.py)
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

    best_ee_acc = 0.0
    best_path = os.path.join(args.output_dir, "best_model_step2.pt")

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

            # Train only the first n-1 heads
            loss = 0.0
            for logits in logits_list[:-1]:
                loss = loss + loss_fn(logits, labels)

            # Average the losses for stability; otherwise the scale grows with the number of heads
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

            if ee_acc > best_ee_acc:
                best_ee_acc = ee_acc
                torch.save(model.state_dict(), best_path)
                print(f"Best Step2 model saved: {best_path} (ee_acc={best_ee_acc:.4f})")

    return history


def train_step1_router_tuning(model, train_loader, test_loader, args, device):
    """
    Router-Tuning Step 1: standard BERT fine-tuning
    - train the backbone + classifier (same as the baseline)
    - the router is not involved yet; this step simply tunes the BERT model
    - reuse train_epoch / evaluate exactly as in the baseline
    """
    from eval import evaluate
    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "best_model_step1.pt")

    # Step 1 does not train the router; freeze router parameters so they stay at zero
    # (sigmoid(0)=0.5=tau, meaning everything is kept)
    for p in model.routers.parameters():
        p.requires_grad = False

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.stage1_learning_rate
    )
    print(f"num parameters: {sum(p.numel() for p in model.parameters()):,}")
    scheduler = _build_scheduler(optimizer, len(train_loader), args.stage1_epochs, warmup_ratio=0.1)

    history = {"router_step1_train_loss": [], "router_step1_test_acc": []}
    best_acc = 0.0

    for epoch in range(args.stage1_epochs):
        print(f"\n===== Router-Tuning Step1: Epoch {epoch+1}/{args.stage1_epochs} =====")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        test_acc = evaluate(model, test_loader, device)

        history["router_step1_train_loss"].append(train_loss)
        history["router_step1_test_acc"].append(test_acc)
        print(f"Train loss: {train_loss:.4f}")
        print(f"Test  acc : {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)
            print(f"Best Router-Tuning step1 model saved: {best_path} (acc={best_acc:.4f})")

    return history, best_path


def train_step2_router_tuning(model, train_loader, test_loader, args, device):
    """
    Router-Tuning Step 2: freeze the backbone and train only the router
    - call model.freeze_backbone() to freeze BERT + classifier
    - only router parameters (one nn.Linear(768, 1) per layer) receive gradients
    - Loss = L_task + lambda_mod * L_MoD
      L_task: cross-entropy to preserve classification performance
      L_MoD:  ReLU(actual_kept - target_kept), encouraging more skipping
    """
    from eval import evaluate_router_full

    # ---- Freeze backbone + classifier, unfreeze router ----
    model.freeze_backbone()
    # Step 1 may have frozen the router; ensure it is trainable here
    for p in model.routers.parameters():
        p.requires_grad = True

    # Pass only trainable parameters (the routers) to the optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), # Select only parameters with requires_grad=True
        lr=args.stage2_learning_rate
    )
    print(f'number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    # Print trainable parameters to confirm that only the router is updated
    print("===== Trainable parameters =====")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    scheduler = _build_scheduler(optimizer, len(train_loader), args.stage2_epochs, warmup_ratio=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    history = {
        "step2_train_loss": [],
        "step2_train_loss_task": [],
        "step2_train_loss_mod": [],
        "step2_test_acc": [],
        "step2_keep_rates": [],         # Per-layer keep_rate during training (mask_hard, averaged at the end of each epoch)
        "step2_eval_keep_rates": [],    # Per-layer keep_rate on the inference path (measured during eval for reporting and FLOPs)
        "step2_avg_keep_rate": [],      # Average keep_rate across all routed layers on the inference path
    }
    best_acc = 0.0
    best_path = os.path.join(args.output_dir, "best_model_step2.pt")

    for epoch in range(args.stage2_epochs):
        print(f"\n===== Router-Tuning Step2: Epoch {epoch+1}/{args.stage2_epochs} =====")
        model.train()

        total_loss = 0.0 # Track total loss (task + MoD) for gradient updates
        total_task = 0.0 # Track the pure classification loss
        total_mod = 0.0 # Track the pure MoD penalty
        keep_rate_sums = None
        keep_rate_counts = None

        for batch in tqdm(train_loader, desc="Step2 Train", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            # forward_with_routing returns three items: logits for cross-entropy,
            # l_mod for the sparsity penalty, and router_stats for logging
            logits, router_stats, l_mod = model.forward_with_routing(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Loss = classification loss + lambda * MoD penalty
            loss_task = loss_fn(logits, labels)
            loss = loss_task + args.lambda_mod * l_mod

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            total_task += loss_task.item()
            total_mod += l_mod.item()

            # Accumulate keep_rate for each layer and average it at the end of the epoch.
            # keep_rate is the fraction kept (not skipped) in each layer; a
            # smaller value means the model skips more aggressively.
            keep_rates = router_stats.get("keep_rates", [])
            if keep_rate_sums is None:
                keep_rate_sums = [0.0 for _ in keep_rates]
                keep_rate_counts = [0 for _ in keep_rates]
            for i, kr in enumerate(keep_rates):
                if kr is not None:
                    keep_rate_sums[i] += float(kr)
                    keep_rate_counts[i] += 1

        # ---- End of the epoch: compute averages ----
        n_batches = max(1, len(train_loader))
        avg_loss = total_loss / n_batches
        avg_task = total_task / n_batches
        avg_mod = total_mod / n_batches

        keep_rate_avgs = None
        if keep_rate_sums is not None:
            keep_rate_avgs = []
            for s, c in zip(keep_rate_sums, keep_rate_counts):
                keep_rate_avgs.append(s / c if c > 0 else None)

        # ---- Evaluation (inference path, measuring the true keep_rate) ----
        test_acc, eval_keep_rates, avg_keep_rate = evaluate_router_full(model, test_loader, device)

        history["step2_train_loss"].append(avg_loss)
        history["step2_train_loss_task"].append(avg_task)
        history["step2_train_loss_mod"].append(avg_mod)
        history["step2_test_acc"].append(test_acc)
        history["step2_keep_rates"].append(keep_rate_avgs)
        history["step2_eval_keep_rates"].append(eval_keep_rates)
        history["step2_avg_keep_rate"].append(avg_keep_rate)

        print(f"Step2 train loss (total): {avg_loss:.4f}")
        print(f"Step2 train loss (task):  {avg_task:.4f}")
        print(f"Step2 train loss (MoD):   {avg_mod:.4f}")
        print(f"Step2 test acc:           {test_acc:.4f}")
        if avg_keep_rate is not None:
            print(f"Step2 avg keep rate (eval): {avg_keep_rate:.4f}")

        if keep_rate_avgs is not None:
            routed_keep = {i: f"{kr:.3f}" for i, kr in enumerate(keep_rate_avgs) if kr is not None}
            print(f"Keep rates (train): {routed_keep}")
        if eval_keep_rates is not None:
            routed_eval = {i: f"{kr:.3f}" for i, kr in enumerate(eval_keep_rates) if kr is not None}
            print(f"Keep rates (eval):  {routed_eval}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)
            print(f"Best model saved: {best_path} (acc={best_acc:.4f})")

    print(f"\nRouter-Tuning Step2 finished. Best acc = {best_acc:.4f}")
    return history


# ====================== HDC-BERT Training ======================

def train_step1_hdc(model, train_loader, test_loader, args, device):
    """
    HDC Stage 1: standard BERT fine-tuning.
    - train the backbone + final classifier
    - freeze off-ramps + routers
    - reuse train_epoch() + evaluate()
    """
    from eval import evaluate
    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "best_model_step1.pt")

    model.freeze_for_stage1()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.stage1_learning_rate,
    )
    scheduler = _build_scheduler(optimizer, len(train_loader), args.stage1_epochs)

    history = {"hdc_step1_train_loss": [], "hdc_step1_test_acc": []}
    best_acc = 0.0

    for epoch in range(args.stage1_epochs):
        print(f"\n===== HDC Stage 1: Epoch {epoch+1}/{args.stage1_epochs} =====")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        test_acc = evaluate(model, test_loader, device)

        history["hdc_step1_train_loss"].append(train_loss)
        history["hdc_step1_test_acc"].append(test_acc)
        print(f"Train loss: {train_loss:.4f}")
        print(f"Test  acc : {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)
            print(f"Best HDC step1 model saved: {best_path} (acc={best_acc:.4f})")

    return history, best_path


def train_step2_hdc(model, train_loader, test_loader, test_loader_ee,
                    args, device, entropy_threshold=0.2, eval_early_exit=True):
    """
    HDC Stage 2: train the off-ramp classifiers.
    - freeze the backbone + routers + final classifier
    - only the off-ramp classifiers receive gradients
    - Loss = (1/K) * sum CE(offramp_i, labels), K = split_layer
    """
    from eval import evaluate, evaluate_hdc_inference
    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "best_model_step2.pt")

    model.freeze_for_stage2()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.stage2_learning_rate,
    )
    print("===== HDC Stage 2 Trainable parameters =====")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name} {param.shape}")

    scheduler = _build_scheduler(optimizer, len(train_loader), args.stage2_epochs)
    loss_fn = torch.nn.CrossEntropyLoss()

    history = {"hdc_step2_train_loss": [], "hdc_step2_test_acc_last": []}
    if eval_early_exit:
        history["hdc_step2_test_acc_hdc"] = []
        history["hdc_step2_stage_a_exit_rate"] = []

    best_acc = 0.0

    for epoch in range(args.stage2_epochs):
        print(f"\n===== HDC Stage 2: Epoch {epoch+1}/{args.stage2_epochs} =====")
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc="Stage2 Train", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            offramp_logits_list = model.forward_all_offramps(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = 0.0
            for logits in offramp_logits_list:
                loss = loss + loss_fn(logits, labels)
            loss = loss / max(1, len(offramp_logits_list))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        history["hdc_step2_train_loss"].append(avg_loss)

        # Eval: last-head accuracy (frozen, for reference)
        test_acc_last = evaluate(model, test_loader, device)
        history["hdc_step2_test_acc_last"].append(test_acc_last)

        print(f"Stage2 train loss: {avg_loss:.4f}")
        print(f"Stage2 test acc (last head, frozen): {test_acc_last:.4f}")

        # Optional: HDC inference evaluation
        if eval_early_exit:
            hdc_results = evaluate_hdc_inference(
                model, test_loader_ee, device,
                entropy_threshold=entropy_threshold,
            )
            history["hdc_step2_test_acc_hdc"].append(hdc_results["accuracy"])
            history["hdc_step2_stage_a_exit_rate"].append(hdc_results["stage_a_exit_rate"])
            print(f"Stage2 HDC acc: {hdc_results['accuracy']:.4f}, "
                  f"Stage A exit rate: {hdc_results['stage_a_exit_rate']:.4f}")

            # Use HDC inference accuracy as the save criterion
            if hdc_results["accuracy"] > best_acc:
                best_acc = hdc_results["accuracy"]
                torch.save(model.state_dict(), best_path)
                print(f"Best HDC step2 model saved: {best_path} (acc={best_acc:.4f})")
        else:
            if test_acc_last > best_acc:
                best_acc = test_acc_last
                torch.save(model.state_dict(), best_path)
                print(f"Best HDC step2 model saved: {best_path} (acc={best_acc:.4f})")

    print(f"\nHDC Stage 2 finished. Best acc = {best_acc:.4f}")
    return history, best_path


def train_step3_hdc(model, train_loader, test_loader, args, device):
    """
    HDC Stage 3: train the token routers.
    - freeze the backbone + off-ramps + final classifier
    - only the routers receive gradients
    - Loss = CE(final_logits, labels) + lambda_mod * sum(l_mod_i)
    """
    from eval import evaluate_hdc_routing_full
    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "best_model_step3.pt")

    model.freeze_for_stage3()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.stage3_learning_rate,
    )
    print("===== HDC Stage 3 Trainable parameters =====")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name} {param.shape}")

    scheduler = _build_scheduler(optimizer, len(train_loader), args.stage3_epochs)
    loss_fn = torch.nn.CrossEntropyLoss()

    history = {
        "hdc_step3_train_loss": [],
        "hdc_step3_train_loss_task": [],
        "hdc_step3_train_loss_mod": [],
        "hdc_step3_test_acc": [],
        "hdc_step3_keep_rates": [],       # train-time keep_rate per Stage-B layer
        "hdc_step3_eval_keep_rates": [],  # eval-time keep_rate per Stage-B layer
        "hdc_step3_avg_keep_rate": [],    # eval-time scalar avg keep_rate
    }
    best_acc = 0.0

    for epoch in range(args.stage3_epochs):
        print(f"\n===== HDC Stage 3: Epoch {epoch+1}/{args.stage3_epochs} =====")
        model.train()

        total_loss = 0.0
        total_task = 0.0
        total_mod = 0.0
        keep_rate_sums = None
        keep_rate_counts = None

        for batch in tqdm(train_loader, desc="Stage3 Train", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            logits, router_stats, l_mod = model.forward_with_routing(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss_task = loss_fn(logits, labels)
            loss = loss_task + args.lambda_mod * l_mod

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            total_task += loss_task.item()
            total_mod += l_mod.item()

            # Accumulate keep_rates
            keep_rates = router_stats.get("keep_rates", [])
            if keep_rate_sums is None:
                keep_rate_sums = [0.0] * len(keep_rates)
                keep_rate_counts = [0] * len(keep_rates)
            for idx, kr in enumerate(keep_rates):
                if kr is not None:
                    keep_rate_sums[idx] += float(kr)
                    keep_rate_counts[idx] += 1

        n_batches = max(1, len(train_loader))
        avg_loss = total_loss / n_batches
        avg_task = total_task / n_batches
        avg_mod = total_mod / n_batches

        keep_rate_avgs = []
        if keep_rate_sums is not None:
            for s, c in zip(keep_rate_sums, keep_rate_counts):
                keep_rate_avgs.append(s / c if c > 0 else None)

        # Evaluation (inference path, where eval-time keep_rate reflects the true token skip ratio)
        test_acc, eval_keep_rates, avg_keep_rate = evaluate_hdc_routing_full(model, test_loader, device)

        history["hdc_step3_train_loss"].append(avg_loss)
        history["hdc_step3_train_loss_task"].append(avg_task)
        history["hdc_step3_train_loss_mod"].append(avg_mod)
        history["hdc_step3_test_acc"].append(test_acc)
        history["hdc_step3_keep_rates"].append(keep_rate_avgs)       # train-time (STE)
        history["hdc_step3_eval_keep_rates"].append(eval_keep_rates) # eval-time (hard mask)
        history["hdc_step3_avg_keep_rate"].append(avg_keep_rate)      # scalar

        print(f"Stage3 loss (total): {avg_loss:.4f}")
        print(f"Stage3 loss (task):  {avg_task:.4f}")
        print(f"Stage3 loss (MoD):   {avg_mod:.4f}")
        print(f"Stage3 test acc:     {test_acc:.4f}")
        if keep_rate_avgs:
            kr_str = {f"B_layer_{i}": f"{kr:.3f}" for i, kr in enumerate(keep_rate_avgs) if kr is not None}
            print(f"Train keep rates: {kr_str}")
        if eval_keep_rates:
            ekr_str = {f"B_layer_{i}": f"{kr:.3f}" for i, kr in enumerate(eval_keep_rates) if kr is not None}
            print(f"Eval  keep rates: {ekr_str} | avg={avg_keep_rate:.3f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)
            print(f"Best HDC step3 model saved: {best_path} (acc={best_acc:.4f})")

    print(f"\nHDC Stage 3 finished. Best acc = {best_acc:.4f}")
    return history, best_path
