"""
Unified training loop for Task 1: Qwen3-0.6B Fine-Tuning.
Supports:
  1. Full Fine-Tuning (Full FT, lambda=0)
  2. Full FT with L2-SP (lambda=1e-3)
  3. LoRA-4 (r=4, alpha=8)
  4. LoRA-16 (r=16, alpha=32)
  5. OPLoRA-16 (r=16, alpha=32, k=128)

Settings:
  - Effective batch: 32 (microbatch 2, accum 16)
  - 1 epoch = 40,000 examples, 1,250 optimizer updates
  - Optimizer: AdamW (betas 0.9, 0.999, eps 1e-8, weight decay 0.01)
  - Schedule: Cosine decay with 38 warmup updates
  - Gradient clipping: 1.0 (unscale before clipping)
  - Gradient checkpointing: enabled
  - Precision: FP16 AMP with dynamic GradScaler
  - Checkpoints: updates 312, 625, 937, 1250
"""
import os
import time
import math
import yaml
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from .lora import apply_lora_to_model
from .oplora import apply_oplora_to_model
from .l2sp import L2SPRegularizer
from .data import MetaMathDataset, collate_fn_pad
from shared.src.seed import set_seed
from shared.src.utils import get_gpu_memory, count_trainable_parameters

def configure_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.01) -> torch.optim.AdamW:
    """
    Configure AdamW optimizer:
    Exclude biases and 1D parameters from weight decay.
    FP32 optimizer states.
    """
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "bias" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)

def train_task1_model(
    model: nn.Module,
    train_dataset: MetaMathDataset,
    config: Dict[str, Any],
    tokenizer = None,
    save_dir: str = "./checkpoints",
    reference_model: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """
    Execute 1 epoch (1,250 optimizer updates) training loop.
    Logs loss, lr, l2sp penalty, displacement, and resource usage.
    """
    method = config.get("method", "full_ft")
    seed = config.get("seed", 6304)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Enable gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    micro_batch = config.get("micro_batch_size", 2)
    accum_steps = config.get("gradient_accumulation_steps", 16)
    max_updates = config.get("max_updates", 1250)
    warmup_updates = config.get("warmup_updates", 38)
    peak_lr = config.get("learning_rate", 2e-5)
    weight_decay = config.get("weight_decay", 0.01)

    # Initialize L2-SP regularizer if applicable
    l2sp_reg = None
    if method == "full_ft_l2sp" and reference_model is not None:
        lambda_val = config.get("lambda_l2sp", 1e-3)
        l2sp_reg = L2SPRegularizer(reference_model, lambda_reg=lambda_val, device="cpu")

    # DataLoader
    dataloader = DataLoader(
        train_dataset,
        batch_size=micro_batch,
        shuffle=False,  # Strict source order per manual instructions!
        collate_fn=lambda b: collate_fn_pad(b, pad_token_id=tokenizer.pad_token_id if tokenizer else 0),
    )

    optimizer = configure_optimizer(model, lr=peak_lr, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_updates,
        num_training_steps=max_updates,
    )
    use_amp = (device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    os.makedirs(save_dir, exist_ok=True)
    history = {
        "step": [],
        "sft_loss": [],
        "total_loss": [],
        "learning_rate": [],
        "l2sp_penalty": [],
        "displacement": [],
        "gpu_peak_mb": [],
        "time_elapsed": [],
    }

    param_stats = count_trainable_parameters(model)
    print(f"[{method.upper()}] Trainable parameters: {param_stats['trainable']:,} / {param_stats['total']:,} ({param_stats['fraction']*100:.2f}%)")

    model.train()
    optimizer.zero_grad()
    accum_loss = 0.0
    accum_sft_loss = 0.0
    update_count = 0
    start_time = time.time()

    data_iter = iter(dataloader)
    while update_count < max_updates:
        # Microbatch accumulation
        for _ in range(accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            try:
                autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16)
            except (AttributeError, TypeError):
                autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16)

            with autocast_ctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                sft_loss = outputs.loss
                loss = sft_loss

                if l2sp_reg is not None:
                    penalty_tensor, raw_omega, scaled_omega = l2sp_reg.compute_penalty(model)
                    loss = loss + penalty_tensor

                loss = loss / accum_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_sft_loss += (sft_loss.item() / accum_steps)
            accum_loss += (loss.item() * accum_steps)

        # Unscale before clipping
        if use_amp:
            try:
                scaler.unscale_(optimizer)
            except (ValueError, RuntimeError):
                pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        optimizer.zero_grad()
        scheduler.step()
        update_count += 1

        current_lr = scheduler.get_last_lr()[0]
        mem = get_gpu_memory()

        # Displacement tracking at midpoint and final
        displacement = 0.0
        if l2sp_reg is not None and update_count in [625, 1250]:
            displacement = l2sp_reg.compute_displacement(model)

        history["step"].append(update_count)
        history["sft_loss"].append(accum_sft_loss)
        history["total_loss"].append(accum_loss)
        history["learning_rate"].append(current_lr)
        history["l2sp_penalty"].append(scaled_omega if l2sp_reg else 0.0)
        history["displacement"].append(displacement)
        history["gpu_peak_mb"].append(mem["peak_mb"])
        history["time_elapsed"].append(time.time() - start_time)

        if update_count % 50 == 0 or update_count == max_updates:
            print(f"Update {update_count:4d}/{max_updates:4d} | SFT Loss: {accum_sft_loss:.4f} | LR: {current_lr:.2e} | Peak GPU: {mem['peak_mb']:.1f}MB")

        # Checkpoint saving
        if update_count in config.get("save_checkpoints", [312, 625, 937, 1250]):
            ckpt_path = os.path.join(save_dir, f"{method}_update_{update_count}.pt")
            torch.save({
                "update": update_count,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "history": history,
            }, ckpt_path)
            print(f"Saved recovery checkpoint to {ckpt_path}")

        accum_loss = 0.0
        accum_sft_loss = 0.0

    return history


if __name__ == "__main__":
    import argparse
    import json
    from .data import prepare_metamath_dataset

    parser = argparse.ArgumentParser(description="Train Qwen3-0.6B on MetaMathQA with full FT / LoRA / OPLoRA / L2-SP")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--debug", action="store_true", help="Run 10 updates for quick pipeline verification")
    parser.add_argument("--save_dir", type=str, default="./task1_llm_finetuning/checkpoints")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.debug:
        cfg["max_updates"] = 10
        cfg["save_checkpoints"] = [5, 10]
        cfg["num_examples"] = 200

    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"[Train CLI] Loading base model {cfg['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    method = cfg.get("method", "full_ft")
    reference_model = None

    # Assignment Manual Precision Specification:
    # "FP16 computation with dynamic gradient scaling; FP32 master parameters for Full FT;
    #  frozen FP16 backbone with FP32 adapters for LoRA/OPLoRA."
    if method in ["full_ft", "full_ft_l2sp"]:
        model_dtype = torch.float32
    else:
        model_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=model_dtype,
    )

    if method == "full_ft_l2sp":
        import copy
        print("[Train CLI] Creating frozen reference copy for L2-SP regularization...")
        reference_model = copy.deepcopy(model)
        for p in reference_model.parameters():
            p.requires_grad = False
    elif method == "lora":
        print(f"[Train CLI] Injecting LoRA adapters (r={cfg.get('lora_r', 16)}, alpha={cfg.get('lora_alpha', 32)})...")
        model = apply_lora_to_model(
            model,
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            target_modules=cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        )
    elif method == "oplora":
        print(f"[Train CLI] Injecting OPLoRA adapters (r={cfg.get('lora_r', 16)}, alpha={cfg.get('lora_alpha', 32)}, k={cfg.get('oplora_k', 128)})...")
        model = apply_oplora_to_model(
            model,
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            k=cfg.get("oplora_k", 128),
            target_modules=cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        )

    print("[Train CLI] Loading & filtering MetaMathQA dataset...")
    train_ds, stats = prepare_metamath_dataset(
        tokenizer=tokenizer,
        target_count=cfg.get("num_examples", 40000),
        max_length=cfg.get("max_seq_length", 2048),
        seed=cfg.get("seed", 6304),
    )

    os.makedirs(args.save_dir, exist_ok=True)
    history = train_task1_model(
        model=model,
        train_dataset=train_ds,
        config=cfg,
        tokenizer=tokenizer,
        save_dir=args.save_dir,
        reference_model=reference_model,
    )

    history_path = os.path.join(args.save_dir, f"{method}_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[Train CLI] Training completed successfully. History saved to {history_path}")
