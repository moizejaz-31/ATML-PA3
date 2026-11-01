"""
Curriculum training for the Task 2 SigLIP-Qwen VLM: one model continues through three stages.
  Stage 1 (updates 1-2,250): caption alignment, connector only, peak LR 1e-3.
  Stage 2 (2,251-3,750): visual instruction tuning, connector 1e-4, LoRA 2e-5.
  Stage 3 (3,751-4,500): natural VQA specialization, connector 5e-5, LoRA 1e-5.
Microbatch 8, accumulation 2, effective batch 16. AdamW, weight decay 0.01, clipping 1.0.
Per-stage 5% warmup, then cosine decay to 10% of peak. FP16 autocast, frozen FP16 backbone, FP32 trainable parameters.
Resumable checkpoints every 250 updates; connector and adapter weights after each stage.
"""
import glob
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from .connector import VLMConnector
from .evaluate import development_metrics, example_weighted_loss, run_final_evaluation
from .feature_cache import FeatureStore
from .sampler import CaptionStream, CurriculumSampler
from .vlm_model import ExampleEncoder, SigLIPQwenVLM
from shared.src.seed import set_seed
from task1_llm_finetuning.src.lora import apply_lora_to_model

SEED = 6304
MICRO_BATCH = 8
ACCUMULATION = 2
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
DEFAULT_STAGES = [
    {"stage": 1, "updates": 2250, "connector_lr": 1e-3, "lora_lr": None},
    {"stage": 2, "updates": 1500, "connector_lr": 1e-4, "lora_lr": 2e-5},
    {"stage": 3, "updates": 750, "connector_lr": 5e-5, "lora_lr": 1e-5},
]

def stage_schedule(updates: int):
    """5% warmup, then cosine decay to 10% of peak."""
    warmup = max(1, int(round(0.05 * updates)))
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, updates - warmup))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return factor

def build_stage_optimizer(model: SigLIPQwenVLM, stage: Dict[str, Any]):
    groups = [{"params": list(model.connector.parameters()), "lr": stage["connector_lr"]}]
    groups[0]["params"] = [p for p in groups[0]["params"] if p.requires_grad]
    if stage["lora_lr"] is not None:
        groups.append({"params": [p for p in model.llm.parameters() if p.requires_grad], "lr": stage["lora_lr"]})
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    return optimizer, LambdaLR(optimizer, stage_schedule(stage["updates"]))

def capture_rng() -> Dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state

def restore_rng(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])

def calibrate_connector(model: SigLIPQwenVLM, manifest, store: FeatureStore, tokenizer, device, sample_images: int = 256) -> Dict[str, float]:
    """Training data only. Rule: scale = mean ||text embedding||_2 / mean ||connector output||_2 at initialization."""
    captions = manifest["caption_train"][:sample_images * 3]
    features = torch.stack([store.get(c["image"]) for c in captions[::3]]).to(device, dtype=torch.float32)
    token_ids = [tid for c in captions for tid in tokenizer.encode(c["caption"], add_special_tokens=False)]
    embeddings = model.llm.get_input_embeddings()(torch.tensor(token_ids, device=device))
    scale = model.connector.calibrate_scale(features, embeddings)
    return {"rule": "mean L2 norm of training-caption token embeddings / mean L2 norm of initial connector outputs on training features", "scale": float(model.connector.scale.item()), "sample_images": len(features)}

def trainable_state(model: SigLIPQwenVLM) -> Dict[str, torch.Tensor]:
    return {name: p.detach() for name, p in model.named_parameters() if p.requires_grad or name == "connector.scale"}

def enable_lora(model: SigLIPQwenVLM, device) -> None:
    """Adapters appear at Stage 2 with rank 16, alpha 32, dropout 0.05, random A, zero B, no trainable biases."""
    apply_lora_to_model(model.llm, target_modules=LORA_TARGETS, r=16, lora_alpha=32, lora_dropout=0.05)
    model.llm.to(device)
    for module in model.llm.modules():
        if hasattr(module, "lora_A"):
            module.lora_A.data = module.lora_A.data.float()
            module.lora_B.data = module.lora_B.data.float()

def load_model(model_name: str, device) -> SigLIPQwenVLM:
    from transformers import AutoModelForCausalLM
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    for param in llm.parameters():
        param.requires_grad = False
    llm.config.use_cache = False
    llm.generation_config.do_sample = False
    llm.generation_config.temperature = None
    llm.generation_config.top_p = None
    llm.generation_config.top_k = None
    llm.generation_config.repetition_penalty = 1.0
    hidden = llm.config.hidden_size
    return SigLIPQwenVLM(llm, VLMConnector(hidden_dim=hidden, out_dim=hidden)).to(device)

def latest_checkpoint(run_dir: str) -> Optional[str]:
    paths = glob.glob(os.path.join(run_dir, "resume_update_*.pt"))
    return max(paths, key=lambda p: int(p.rsplit("_", 1)[-1][:-3])) if paths else None

def write_json(path: str, payload: Any) -> None:
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    os.replace(path + ".tmp", path)

def train_curriculum(
    manifest: Dict[str, Any],
    feature_dir: str,
    run_dir: str,
    results_dir: str,
    model_name: str = "Qwen/Qwen3-0.6B",
    visual_tokens: int = 196,
    stages: Optional[List[Dict[str, Any]]] = None,
    save_every: int = 250,
    eval_every: int = 500,
    until_update: Optional[int] = None,
    time_limit_min: Optional[float] = None,
) -> Dict[str, Any]:
    """Run or resume the curriculum. Returns the run summary; summary['completed'] marks the finished model."""
    from transformers import AutoTokenizer
    stages = stages or DEFAULT_STAGES
    total_updates = sum(s["updates"] for s in stages)
    boundaries = np.cumsum([s["updates"] for s in stages]).tolist()
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = ExampleEncoder(tokenizer)
    store = FeatureStore(feature_dir, visual_tokens=visual_tokens)
    model = load_model(model_name, device)
    model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    caption_stream = CaptionStream(manifest["caption_train"], SEED)
    samplers = {2: CurriculumSampler(2, manifest, SEED + 2), 3: CurriculumSampler(3, manifest, SEED + 3)}

    history: List[Dict[str, Any]] = []
    dev_history: List[Dict[str, Any]] = []
    summary = {"visual_tokens": visual_tokens, "total_updates": total_updates, "full_recipe": stages == DEFAULT_STAGES, "training_seconds": 0.0, "peak_gpu_mb": 0.0,
               "skipped_updates": [], "stage_params": {}, "stage_seconds": {}, "mixture_counts": {}, "checkpoint_mb": {}, "completed": False}
    update = 0
    optimizer, scheduler = None, None
    stage_index = 0

    resume_path = latest_checkpoint(run_dir)
    rng_state = None
    if resume_path:
        print(f"[Train] Resuming from {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        update = ckpt["update"]
        # A checkpoint at exactly the Stage 1 boundary has no adapters yet; the loop creates them with their seed
        if update > boundaries[0]:
            enable_lora(model, device)
        missing = model.load_state_dict(ckpt["model"], strict=False)
        if missing.unexpected_keys:
            raise KeyError(f"Checkpoint does not match this model: {missing.unexpected_keys[:3]}")
        stage_index = next((i for i, b in enumerate(boundaries) if update < b), len(stages) - 1)
        if update not in boundaries and update < total_updates:
            optimizer, scheduler = build_stage_optimizer(model, stages[stage_index])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        caption_stream.load_state(ckpt["caption_stream"])
        for key, state in ckpt["samplers"].items():
            samplers[key].load_state(state)
        history, dev_history, summary = ckpt["history"], ckpt["dev_history"], ckpt["summary"]
        rng_state = ckpt["rng"]
        del ckpt
    else:
        write_json(os.path.join(results_dir, "calibration.json"), calibrate_connector(model, manifest, store, tokenizer, device))
        print("[Train] Evaluating before training...", flush=True)
        dev_history.append({"update": 0, **development_metrics(model, encoder, store, manifest)})
    if rng_state is not None:
        restore_rng(rng_state)

    def save(reason_update: int) -> None:
        for old in glob.glob(os.path.join(run_dir, "resume_update_*.pt")):
            os.remove(old)
        path = os.path.join(run_dir, f"resume_update_{reason_update}.pt")
        torch.save({"update": reason_update, "model": trainable_state(model), "optimizer": optimizer.state_dict() if optimizer else None, "scheduler": scheduler.state_dict() if scheduler else None,
                    "scaler": scaler.state_dict(), "caption_stream": caption_stream.state(), "samplers": {k: s.state() for k, s in samplers.items()}, "rng": capture_rng(),
                    "history": history, "dev_history": dev_history, "summary": summary}, path + ".tmp")
        os.replace(path + ".tmp", path)
        write_results()

    def write_results() -> None:
        write_json(os.path.join(results_dir, "training_history.json"), history)
        write_json(os.path.join(results_dir, "development_history.json"), dev_history)
        write_json(os.path.join(results_dir, "summary.json"), summary)

    session_start = time.time()
    session_base = summary["training_seconds"]
    limit = min(until_update or total_updates, total_updates)

    while update < limit:
        stage_index = next(i for i, b in enumerate(boundaries) if update < b)
        stage = stages[stage_index]
        if optimizer is None:
            if stage["lora_lr"] is not None and not any(hasattr(m, "lora_A") for m in model.llm.modules()):
                set_seed(SEED + stage["stage"])
                enable_lora(model, device)
            optimizer, scheduler = build_stage_optimizer(model, stage)
            summary["stage_params"][str(stage["stage"])] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model.train()

        model.train()
        stream = caption_stream if stage["stage"] == 1 else samplers[stage["stage"]]
        step_loss = 0.0
        for _ in range(ACCUMULATION):
            examples = stream.next_examples(MICRO_BATCH)
            encoded = [encoder.encode(e) for e in examples]
            features = [store.get(e["image"]) for e in examples]
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                logits, labels = model(encoded, features)
            loss = example_weighted_loss(logits, labels) / ACCUMULATION
            scaler.scale(loss).backward()
            step_loss += loss.item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], max_norm=1.0)
        lrs = scheduler.get_last_lr()
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < scale_before
        optimizer.zero_grad()
        scheduler.step()
        update += 1

        summary["training_seconds"] = session_base + time.time() - session_start
        if torch.cuda.is_available():
            summary["peak_gpu_mb"] = max(summary["peak_gpu_mb"], torch.cuda.max_memory_allocated() / (1024 ** 2))
        if skipped:
            summary["skipped_updates"].append(update)
        history.append({"update": update, "stage": stage["stage"], "loss": step_loss, "connector_lr": lrs[0], "lora_lr": lrs[1] if len(lrs) > 1 else None, "grad_norm": float(grad_norm), "seconds": summary["training_seconds"]})
        if update % 50 == 0:
            print(f"[Stage {stage['stage']}] {update}/{total_updates} | loss {step_loss:.4f} | {summary['training_seconds'] / 60:.1f} min", flush=True)

        at_boundary = update in boundaries
        if update % eval_every == 0 or at_boundary:
            dev_history.append({"update": update, **development_metrics(model, encoder, store, manifest)})
            print(f"[Dev] {dev_history[-1]}", flush=True)
        if at_boundary:
            summary["stage_seconds"][str(stage["stage"])] = summary["training_seconds"]
            summary["mixture_counts"] = {str(k): s.counts for k, s in samplers.items()}
            weights = os.path.join(run_dir, f"stage_{stage['stage']}_weights.pt")
            torch.save(trainable_state(model), weights)
            summary["checkpoint_mb"][f"stage_{stage['stage']}"] = os.path.getsize(weights) / (1024 ** 2)
            optimizer, scheduler = None, None
        out_of_time = time_limit_min is not None and (time.time() - session_start) / 60.0 >= time_limit_min
        if update % save_every == 0 or at_boundary or out_of_time or update == limit:
            summary["mixture_counts"] = {str(k): s.counts for k, s in samplers.items()}
            summary["completed"] = update == total_updates
            save(update)
        if out_of_time and update < limit:
            print(f"[Train] Time limit reached at update {update}. Run again to resume.", flush=True)
            return summary
    write_results()
    return summary

def run_final(manifest: Dict[str, Any], feature_dir: str, run_dir: str, results_dir: str, model_name: str = "Qwen/Qwen3-0.6B", visual_tokens: int = 196, final_stage: int = 3) -> None:
    """Load the completed Stage 3 model and evaluate both final-test suites."""
    from transformers import AutoTokenizer
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(model_name, device)
    enable_lora(model, device)
    weights = torch.load(os.path.join(run_dir, f"stage_{final_stage}_weights.pt"), map_location="cpu", weights_only=True)
    result = model.load_state_dict(weights, strict=False)
    if result.unexpected_keys or set(weights) & set(result.missing_keys):
        raise KeyError("Stage weights do not match this model")
    model.llm.config.use_cache = True
    run_final_evaluation(model, ExampleEncoder(tokenizer), FeatureStore(feature_dir, visual_tokens=visual_tokens), tokenizer, manifest, results_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train or evaluate the Task 2 VLM curriculum")
    parser.add_argument("--manifest", type=str, default="task2_vlm_grounding/splits/manifest_seed6304.json")
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--run_dir", type=str, default="task2_vlm_grounding/checkpoints/main")
    parser.add_argument("--results_dir", type=str, default="task2_vlm_grounding/results/main")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--visual_tokens", type=int, default=196, choices=[196, 49])
    parser.add_argument("--final_only", action="store_true", help="Evaluate the final model instead of training")
    parser.add_argument("--time_limit_min", type=float, default=None)
    parser.add_argument("--until_update", type=int, default=None)
    parser.add_argument("--stage_updates", type=str, default=None, help="Tests only: comma-separated stage lengths")
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--eval_every", type=int, default=500)
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if args.final_only:
        run_final(manifest, args.feature_dir, args.run_dir, args.results_dir, args.model_name, args.visual_tokens)
    else:
        stages = None
        if args.stage_updates:
            stages = [{**s, "updates": int(n)} for s, n in zip(DEFAULT_STAGES, args.stage_updates.split(","))]
        result = train_curriculum(manifest, args.feature_dir, args.run_dir, args.results_dir, args.model_name, args.visual_tokens, stages, args.save_every, args.eval_every, args.until_update, args.time_limit_min)
        print(f"[Train] Finished at {result.get('completed')}", flush=True)
