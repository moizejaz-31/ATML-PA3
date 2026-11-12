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
  - Loss: summed over response tokens, normalized by all valid tokens in the effective batch
  - Optimizer: AdamW (betas 0.9, 0.999, eps 1e-8, weight decay 0.01)
  - Schedule: Cosine decay with 38 warmup updates
  - Gradient clipping: 1.0 (unscale before clipping)
  - Gradient checkpointing: enabled
  - Precision: FP16 AMP with dynamic GradScaler
  - Recovery checkpoints: updates 312, 625, 937, 1250, plus any time-limit stop

Each run keeps its files in its own folder:
  <save_dir>/<run_name>/recovery_update_<n>.pt   optimizer, scheduler, scaler, RNG, data position
  <save_dir>/<run_name>/model_update_<n>.pt      Full FT weights at evaluation stages
  <save_dir>/<run_name>/adapters_update_<n>.pt   adapter factors at evaluation stages
  <results_dir>/<run_name>/history.json, summary.json
"""
import glob
import itertools
import json
import os
import random
import re
import time
import yaml
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import get_cosine_schedule_with_warmup

from .lora import apply_lora_to_model, export_adapter_factors
from .oplora import apply_oplora_to_model
from .l2sp import L2SPRegularizer
from .data import MetaMathDataset, collate_fn_pad
from shared.src.seed import set_seed
from shared.src.utils import get_gpu_memory, count_trainable_parameters

FULL_FT_METHODS = ["full_ft", "full_ft_l2sp"]
MIB = 1024 ** 2

def configure_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.01) -> torch.optim.AdamW:
    """
    Configure AdamW optimizer:
    Exclude biases and parameters with fewer than two dimensions from weight decay.
    FP32 optimizer states.
    """
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or name.endswith("bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    # The fused kernel avoids the temporary copies of every gradient that the default implementation makes
    fused = all(p.is_cuda for group in optim_groups for p in group["params"]) and torch.cuda.is_available()
    return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8, fused=fused)

def count_valid_tokens(labels: torch.Tensor) -> int:
    """Number of supervised next-token targets in a microbatch."""
    return int((labels[:, 1:] != -100).sum().item())

LOSS_CHUNK_TOKENS = 1024

def _chunk_cross_entropy(hidden: torch.Tensor, targets: torch.Tensor, lm_head: nn.Module) -> torch.Tensor:
    return F.cross_entropy(lm_head(hidden).float(), targets, reduction="sum")

def chunked_next_token_loss(hidden: torch.Tensor, labels: torch.Tensor, lm_head: nn.Module, chunk: int = LOSS_CHUNK_TOKENS) -> torch.Tensor:
    """
    The same sum as summed_next_token_loss, without full-vocabulary logits for every position.
    Only supervised positions are projected, a chunk at a time, and each chunk's logits are
    recomputed during backward instead of stored.
    """
    mask = labels[:, 1:] != -100
    selected = hidden[:, :-1][mask]
    targets = labels[:, 1:][mask]
    total = selected.new_zeros((), dtype=torch.float32)
    for start in range(0, selected.shape[0], chunk):
        total = total + torch.utils.checkpoint.checkpoint(_chunk_cross_entropy, selected[start:start + chunk], targets[start:start + chunk], lm_head, use_reentrant=False)
    return total

def summed_next_token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy summed over supervised tokens, computed in FP32."""
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)).float(),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )

def capture_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state

def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])

def trainable_state(model: nn.Module, method: str) -> Dict[str, torch.Tensor]:
    """Adapter runs only store their adapters; the frozen backbone comes from the Hub."""
    if method in FULL_FT_METHODS:
        return model.state_dict()
    return {name: p.detach() for name, p in model.named_parameters() if p.requires_grad}

def latest_recovery_path(run_dir: str) -> Optional[str]:
    paths = glob.glob(os.path.join(run_dir, "recovery_update_*.pt"))
    if not paths:
        return None
    return max(paths, key=lambda p: int(re.search(r"recovery_update_(\d+)\.pt", p).group(1)))

def file_mb(path: str) -> float:
    return os.path.getsize(path) / MIB

def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)

def train_task1_model(
    model: nn.Module,
    train_dataset: MetaMathDataset,
    config: Dict[str, Any],
    tokenizer = None,
    save_dir: str = "./checkpoints",
    results_dir: str = "./results/training",
    time_limit_min: Optional[float] = None,
    keep_all_recovery: bool = False,
    preprocessing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute or resume 1 epoch (1,250 optimizer updates) of training.
    Logs loss, lr, l2sp penalty, displacement, skipped updates, and resource usage.
    A run stopped by time_limit_min saves a recovery checkpoint and continues on the next call.
    Returns the summary dict; summary["completed"] says whether all updates finished.
    """
    method = config.get("method", "full_ft")
    run_name = config.get("run_name", method)
    seed = config.get("seed", 6304)
    set_seed(seed)

    run_dir = os.path.join(save_dir, run_name)
    run_results_dir = os.path.join(results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    summary_path = os.path.join(run_results_dir, "summary.json")
    history_path = os.path.join(run_results_dir, "history.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Enable gradient checkpointing
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    micro_batch = config.get("micro_batch_size", 2)
    accum_steps = config.get("gradient_accumulation_steps", 16)
    max_updates = config.get("max_updates", 1250)
    warmup_updates = config.get("warmup_updates", 38)
    peak_lr = config.get("learning_rate", 2e-5)
    weight_decay = config.get("weight_decay", 0.01)
    save_updates = set(config.get("save_checkpoints", [312, 625, 937, 1250]))
    eval_updates = set(config.get("eval_checkpoints", [1250]))

    required_examples = micro_batch * accum_steps * max_updates
    if len(train_dataset) < required_examples:
        raise ValueError(f"{max_updates} updates need {required_examples} examples, dataset has {len(train_dataset)}")

    # Snapshot theta_0 before any recovery state is loaded. It stays on the CPU because Full FT
    # already holds parameters, gradients, and Adam states on the GPU; each parameter's reference
    # moves over only while the penalty is computed, and l2sp_seconds includes that transfer.
    reference = None
    if method in FULL_FT_METHODS:
        reference = L2SPRegularizer(model, lambda_reg=config.get("lambda_l2sp", 0.0), device="cpu")
    use_l2sp = method == "full_ft_l2sp"

    optimizer = configure_optimizer(model, lr=peak_lr, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_updates,
        num_training_steps=max_updates,
    )
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {
        "update": [],
        "sft_loss": [],
        "total_loss": [],
        "learning_rate": [],
        "l2sp_omega": [],
        "l2sp_penalty": [],
        "grad_norm": [],
        "skipped": [],
        "gpu_peak_mb": [],
        "time_elapsed": [],
    }
    param_stats = count_trainable_parameters(model)
    summary = {
        "run_name": run_name,
        "method": method,
        "max_updates": max_updates,
        "effective_batch_size": micro_batch * accum_steps,
        "trainable_params": param_stats["trainable"],
        "total_params": param_stats["total"],
        "displacement": {},
        "skipped_updates": [],
        "checkpoint_sizes_mb": {},
        "peak_gpu_mb": 0.0,
        "training_seconds": 0.0,
        "l2sp_seconds": 0.0,
        "preprocessing": preprocessing or {},
        "completed": False,
    }
    if reference is not None:
        summary["reference_storage_mb"] = reference.storage_mb
        summary["reference_snapshot_seconds"] = reference.snapshot_seconds
    print(f"[{run_name.upper()}] Trainable parameters: {param_stats['trainable']:,} / {param_stats['total']:,} ({param_stats['fraction']*100:.2f}%)")

    update_count = 0
    microbatches_done = 0
    rng_state = None
    recovery_path = latest_recovery_path(run_dir)
    if recovery_path:
        print(f"[{run_name.upper()}] Resuming from {recovery_path}")
        # Load on the CPU: RNG states must stay CPU tensors, and the optimizer moves its own states
        ckpt = torch.load(recovery_path, map_location="cpu", weights_only=False)
        load_result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if load_result.unexpected_keys or set(ckpt["model_state_dict"]) & set(load_result.missing_keys):
            raise KeyError(f"Recovery checkpoint does not match this model: {load_result.unexpected_keys[:3]}")
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        update_count = ckpt["update"]
        microbatches_done = ckpt["microbatches_done"]
        history = ckpt["history"]
        summary = ckpt["summary"]
        rng_state = ckpt["rng_state"]
        del ckpt

    # Source order is fixed, so the data position is just the number of microbatches consumed
    remaining = Subset(train_dataset, range(microbatches_done * micro_batch, len(train_dataset)))
    dataloader = DataLoader(
        remaining,
        batch_size=micro_batch,
        shuffle=False,  # Strict source order per manual instructions!
        collate_fn=lambda b: collate_fn_pad(b, pad_token_id=tokenizer.pad_token_id if tokenizer else 0),
    )

    def save_recovery(update: int) -> None:
        if not keep_all_recovery:
            # Full FT recovery files are several GB, so only the newest one is kept on disk
            for old_path in glob.glob(os.path.join(run_dir, "recovery_update_*.pt")):
                os.remove(old_path)
        path = os.path.join(run_dir, f"recovery_update_{update}.pt")
        torch.save({
            "update": update,
            "microbatches_done": microbatches_done,
            "model_state_dict": trainable_state(model, method),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "rng_state": capture_rng_state(),
            "history": history,
            "summary": summary,
        }, path + ".tmp")
        os.replace(path + ".tmp", path)
        summary["checkpoint_sizes_mb"][f"recovery_{update}"] = file_mb(path)
        write_json(history_path, history)
        write_json(summary_path, summary)
        print(f"Saved recovery checkpoint to {path} ({file_mb(path):.1f} MB)")

    def save_evaluation_state(update: int) -> None:
        if method in FULL_FT_METHODS:
            path = os.path.join(run_dir, f"model_update_{update}.pt")
            torch.save(model.state_dict(), path + ".tmp")
        else:
            path = os.path.join(run_dir, f"adapters_update_{update}.pt")
            start = time.perf_counter()
            factors = export_adapter_factors(model)
            summary["adapter_export_seconds"] = time.perf_counter() - start
            torch.save({
                "method": method,
                "r": config.get("lora_r"),
                "alpha": config.get("lora_alpha"),
                "k": config.get("oplora_k"),
                "adapters": factors,
            }, path + ".tmp")
        os.replace(path + ".tmp", path)
        summary["checkpoint_sizes_mb"][f"evaluation_{update}"] = file_mb(path)
        print(f"Saved evaluation state to {path} ({file_mb(path):.1f} MB)")

    model.train()
    optimizer.zero_grad()
    session_start = time.time()
    session_base_seconds = summary["training_seconds"]

    # Creating the iterator draws from the torch RNG, so restore the saved state after it
    data_iter = iter(dataloader)
    if rng_state is not None:
        restore_rng_state(rng_state)

    while update_count < max_updates:
        # Collect the effective batch first: the loss is normalized over all of its valid tokens
        batches: List[Dict[str, torch.Tensor]] = list(itertools.islice(data_iter, accum_steps))
        if len(batches) < accum_steps:
            raise RuntimeError(f"Training data ran out at update {update_count + 1}")
        valid_tokens = sum(count_valid_tokens(b["labels"]) for b in batches)

        accum_sft_loss = 0.0
        for batch in batches:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                hidden = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                loss = chunked_next_token_loss(hidden, labels, model.lm_head) / valid_tokens

            scaler.scale(loss).backward()
            accum_sft_loss += loss.item()
        microbatches_done += accum_steps

        # lambda * Omega depends only on the parameters, so it enters once per update
        raw_omega, scaled_omega = 0.0, 0.0
        if use_l2sp:
            l2sp_start = time.perf_counter()
            raw_omega, scaled_omega = reference.add_penalty_gradient(model, grad_scale=scaler.get_scale())
            summary["l2sp_seconds"] += time.perf_counter() - l2sp_start

        current_lr = scheduler.get_last_lr()[0]

        # Unscale before clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < scale_before

        optimizer.zero_grad()
        scheduler.step()
        update_count += 1

        mem = get_gpu_memory()
        summary["peak_gpu_mb"] = max(summary["peak_gpu_mb"], mem["peak_mb"])
        summary["training_seconds"] = session_base_seconds + (time.time() - session_start)
        if skipped:
            summary["skipped_updates"].append(update_count)
            print(f"Update {update_count}: skipped by the gradient scaler (non-finite gradients)")

        history["update"].append(update_count)
        history["sft_loss"].append(accum_sft_loss)
        history["total_loss"].append(accum_sft_loss + scaled_omega)
        history["learning_rate"].append(current_lr)
        history["l2sp_omega"].append(raw_omega)
        history["l2sp_penalty"].append(scaled_omega)
        history["grad_norm"].append(float(grad_norm))
        history["skipped"].append(bool(skipped))
        history["gpu_peak_mb"].append(mem["peak_mb"])
        history["time_elapsed"].append(summary["training_seconds"])

        if update_count % 50 == 0 or update_count == max_updates:
            print(f"Update {update_count:4d}/{max_updates:4d} | SFT Loss: {accum_sft_loss:.4f} | LR: {current_lr:.2e} | Peak GPU: {mem['peak_mb']:.1f}MB | {summary['training_seconds']/60:.1f} min")

        if update_count in eval_updates:
            if reference is not None:
                summary["displacement"][str(update_count)] = reference.compute_displacement(model)
            save_evaluation_state(update_count)

        out_of_time = time_limit_min is not None and (time.time() - session_start) / 60.0 >= time_limit_min
        if update_count == max_updates:
            summary["completed"] = True
        if update_count in save_updates or (out_of_time and not summary["completed"]):
            save_recovery(update_count)
        if out_of_time and not summary["completed"]:
            print(f"[{run_name.upper()}] Time limit reached at update {update_count}/{max_updates}. Run again to resume.")
            return summary

    if not keep_all_recovery:
        # The finished run no longer needs resuming, and its evaluation states are kept
        for old_path in glob.glob(os.path.join(run_dir, "recovery_update_*.pt")):
            os.remove(old_path)
    write_json(history_path, history)
    write_json(summary_path, summary)
    return summary


def load_base_model(model_name: str, method: str = "baseline") -> nn.Module:
    """
    Assignment Manual Precision Specification:
    "FP16 computation with dynamic gradient scaling; FP32 master parameters for Full FT;
     frozen FP16 backbone with FP32 adapters for LoRA/OPLoRA."
    FP16 only exists on the GPU path; CPU smoke tests stay in FP32.
    """
    from transformers import AutoModelForCausalLM
    use_fp16 = method not in FULL_FT_METHODS and torch.cuda.is_available()
    return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if use_fp16 else torch.float32)


def run_is_complete(results_dir: str, run_name: str) -> bool:
    summary_path = os.path.join(results_dir, run_name, "summary.json")
    if not os.path.exists(summary_path):
        return False
    with open(summary_path, "r", encoding="utf-8") as f:
        return bool(json.load(f).get("completed"))


if __name__ == "__main__":
    import argparse
    from .data import prepare_metamath_dataset

    parser = argparse.ArgumentParser(description="Train Qwen3-0.6B on MetaMathQA with full FT / LoRA / OPLoRA / L2-SP")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--debug", action="store_true", help="Run 10 updates for quick pipeline verification")
    parser.add_argument("--save_dir", type=str, default="./task1_llm_finetuning/checkpoints")
    parser.add_argument("--results_dir", type=str, default="./task1_llm_finetuning/results/training")
    parser.add_argument("--ids_path", type=str, default="./task1_llm_finetuning/splits/metamath_ids_seed6304.json")
    parser.add_argument("--model_name", type=str, default=None, help="Override the config's model")
    parser.add_argument("--time_limit_min", type=float, default=None, help="Save a recovery checkpoint and stop after this many minutes")
    parser.add_argument("--keep_all_recovery", action="store_true", help="Keep every recovery checkpoint instead of only the newest")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["run_name"] = os.path.splitext(os.path.basename(args.config))[0]
    if args.model_name:
        cfg["model_name"] = args.model_name

    if args.debug:
        cfg["run_name"] += "_debug"
        cfg["max_updates"] = 10
        cfg["warmup_updates"] = 2
        cfg["save_checkpoints"] = [5, 10]
        cfg["eval_checkpoints"] = [5, 10]
        cfg["num_examples"] = cfg["micro_batch_size"] * cfg["gradient_accumulation_steps"] * cfg["max_updates"]

    if run_is_complete(args.results_dir, cfg["run_name"]):
        print(f"[Train CLI] {cfg['run_name']} already completed. Nothing to do.")
        raise SystemExit(0)

    from transformers import AutoTokenizer
    print(f"[Train CLI] Loading base model {cfg['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    method = cfg.get("method", "full_ft")
    model = load_base_model(cfg["model_name"], method)

    # Seed before adapter creation so that LoRA-16 and OPLoRA-16 start from identical factors
    set_seed(cfg.get("seed", 6304))
    preprocessing = {}
    if method == "lora":
        print(f"[Train CLI] Injecting LoRA adapters (r={cfg['lora_r']}, alpha={cfg['lora_alpha']})...")
        model = apply_lora_to_model(
            model,
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["target_modules"],
        )
    elif method == "oplora":
        print(f"[Train CLI] Injecting OPLoRA adapters (r={cfg['lora_r']}, alpha={cfg['lora_alpha']}, k={cfg['oplora_k']})...")
        svd_start = time.perf_counter()
        model = apply_oplora_to_model(
            model,
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            k=cfg["oplora_k"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["target_modules"],
        )
        preprocessing["svd_seconds"] = time.perf_counter() - svd_start
        preprocessing["basis_storage_mb"] = sum(b.numel() * b.element_size() for b in model.buffers()) / MIB

    print("[Train CLI] Loading & filtering MetaMathQA dataset...")
    train_ds, stats = prepare_metamath_dataset(
        tokenizer=tokenizer,
        target_count=cfg.get("num_examples", 40000),
        max_length=cfg.get("max_seq_length", 2048),
        seed=cfg.get("seed", 6304),
        ids_path=None if args.debug else args.ids_path,
    )

    result = train_task1_model(
        model=model,
        train_dataset=train_ds,
        config=cfg,
        tokenizer=tokenizer,
        save_dir=args.save_dir,
        results_dir=args.results_dir,
        time_limit_min=args.time_limit_min,
        keep_all_recovery=args.keep_all_recovery,
        preprocessing=preprocessing,
    )
    state = "completed" if result["completed"] else "stopped at the time limit; run again to resume"
    print(f"[Train CLI] {cfg['run_name']} {state}.")
