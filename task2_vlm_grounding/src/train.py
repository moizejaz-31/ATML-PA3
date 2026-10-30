"""
Unified curriculum training loop for Task 2 SigLIP-Qwen VLM.
Implements example-weighted causal cross-entropy loss:
L = - (1 / |B|) sum_{i in B} (1 / |R_i|) sum_{t in R_i} log p(y_{i,t} | I_i, q_i, y_{i,<t})

Stages:
  Stage 1: Caption Alignment (2,250 updates, connector only, peak LR 1e-3, 50 warmup updates)
  Stage 2: Visual Instruction Tuning (1,500 updates, connector + LoRA-16, peak LR 2e-4)
  Stage 3: Natural VQA Specialization (750 updates, connector + LoRA-16, peak LR 1e-4)
"""
import os
import time
import json
import yaml
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from transformers import get_cosine_schedule_with_warmup

def compute_example_weighted_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Computes cross-entropy averaged within each example over its valid response tokens,
    then averaged across the batch.
    logits: (B, seq_len, vocab_size)
    labels: (B, seq_len)
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_losses = token_losses.view(shift_labels.shape)

    valid_mask = (shift_labels != ignore_index).float()
    tokens_per_example = valid_mask.sum(dim=-1).clamp(min=1.0)
    example_loss = (token_losses * valid_mask).sum(dim=-1) / tokens_per_example

    return example_loss.mean()

def train_curriculum_stage(
    model: nn.Module,
    stage: int,
    dataloader,
    config: Dict[str, Any],
    save_dir: str = "./checkpoints",
) -> Dict[str, Any]:
    """
    Execute training for one curriculum stage.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    max_updates = config.get("max_updates", 2250 if stage == 1 else (1500 if stage == 2 else 750))
    warmup_updates = config.get("warmup_updates", 50)
    peak_lr = config.get("learning_rate", 1e-3 if stage == 1 else (2e-4 if stage == 2 else 1e-4))
    accum_steps = config.get("gradient_accumulation_steps", 8)

    # Optimizer params: connector trainable throughout; LoRA in stages 2-3
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=peak_lr, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_updates, num_training_steps=max_updates)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    os.makedirs(save_dir, exist_ok=True)
    history = {"step": [], "loss": [], "lr": [], "time_elapsed": []}

    model.train()
    optimizer.zero_grad()
    accum_loss = 0.0
    update_count = 0
    start_time = time.time()

    data_iter = iter(dataloader)
    while update_count < max_updates:
        for _ in range(accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            visual_feats = batch["visual_features"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    visual_features=visual_feats,
                )
                loss = compute_example_weighted_loss(outputs.logits, labels)
                loss = loss / accum_steps

            scaler.scale(loss).backward()
            accum_loss += (loss.item() * accum_steps)

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        update_count += 1

        history["step"].append(update_count)
        history["loss"].append(accum_loss)
        history["lr"].append(scheduler.get_last_lr()[0])
        history["time_elapsed"].append(time.time() - start_time)

        if update_count % 50 == 0 or update_count == max_updates:
            print(f"[Stage {stage}] Update {update_count:4d}/{max_updates:4d} | Loss: {accum_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        accum_loss = 0.0

    # Save stage checkpoint
    stage_ckpt_path = os.path.join(save_dir, f"stage_{stage}_final.pt")
    torch.save({
        "stage": stage,
        "connector_state_dict": model.connector.state_dict(),
        "model_state_dict": model.state_dict(),
        "history": history,
    }, stage_ckpt_path)
    print(f"Saved stage {stage} checkpoint to {stage_ckpt_path}")

    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Task 2 SigLIP-Qwen VLM curriculum stage")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./task2_vlm_grounding/checkpoints")
    args = parser.parse_args()

    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    if args.debug:
        cfg["max_updates"] = 10
        cfg["warmup_updates"] = 2

    print(f"[Train CLI] Running Stage {args.stage} with max_updates={cfg.get('max_updates', 'default')}...")
