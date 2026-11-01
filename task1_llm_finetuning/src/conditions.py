"""
The seven Task 1 evaluation conditions, and how to rebuild each model from its saved state.
Adapters are merged into a fresh backbone copy, using projected factors for OPLoRA.
"""
import os
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

from .lora import apply_lora_to_model, merge_factors_into_backbone
from .oplora import apply_oplora_to_model
from .train import load_base_model

# condition name -> (training run, update). The baseline has no training run.
CONDITIONS: Dict[str, Tuple[Optional[str], int]] = {
    "baseline": (None, 0),
    "full_ft_625": ("full_ft", 625),
    "full_ft_1250": ("full_ft", 1250),
    "full_ft_l2sp_1250": ("full_ft_l2sp", 1250),
    "lora_r4_1250": ("lora_r4", 1250),
    "lora_r16_1250": ("lora_r16", 1250),
    "oplora_r16_1250": ("oplora_r16", 1250),
}

def load_evaluation_state(condition: str, checkpoint_dir: str) -> Dict[str, Any]:
    """Return {"kind": "baseline" | "full" | "adapters", ...} for a condition. Missing files are errors."""
    run, update = CONDITIONS[condition]
    if run is None:
        return {"kind": "baseline"}
    model_path = os.path.join(checkpoint_dir, run, f"model_update_{update}.pt")
    adapter_path = os.path.join(checkpoint_dir, run, f"adapters_update_{update}.pt")
    if os.path.exists(model_path):
        return {"kind": "full", "state_dict": torch.load(model_path, map_location="cpu", weights_only=True, mmap=True)}
    if os.path.exists(adapter_path):
        saved = torch.load(adapter_path, map_location="cpu", weights_only=True)
        return {"kind": "adapters", **saved}
    raise FileNotFoundError(f"No saved state for {condition}: looked for {model_path} and {adapter_path}")

def apply_evaluation_state(model: nn.Module, state: Dict[str, Any]) -> nn.Module:
    """Apply a saved state to a fresh backbone in its current dtype. Every key must match."""
    if state["kind"] == "full":
        model.load_state_dict(state["state_dict"], strict=True)
    elif state["kind"] == "adapters":
        merge_factors_into_backbone(model, state["adapters"])
    return model

def load_condition_model(condition: str, checkpoint_dir: str, model_name: str, device: torch.device) -> nn.Module:
    """Fresh backbone plus the condition's saved state, ready for FP16 evaluation with dropout disabled."""
    model = load_base_model(model_name, "baseline")
    state = load_evaluation_state(condition, checkpoint_dir)
    model = apply_evaluation_state(model, state).to(device).eval()
    # The checkpoint's generation_config asks for sampling; every Task 1 generation is greedy
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.repetition_penalty = 1.0
    return model

@torch.no_grad()
def verify_merged_logits(
    condition: str,
    checkpoint_dir: str,
    model_name: str,
    sample_input_ids: torch.Tensor,
    tolerance: float = 1e-2,
) -> Dict[str, Any]:
    """
    Compare merged and unmerged logits on a small batch, both in FP32, using the real adapter classes.
    A mismatch above tolerance stops the evaluation.
    """
    state = load_evaluation_state(condition, checkpoint_dir)
    if state["kind"] != "adapters":
        return {"condition": condition, "checked": False}

    from transformers import AutoModelForCausalLM
    def fresh_fp32():
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).eval()

    merged = apply_evaluation_state(fresh_fp32(), state)
    merged_logits = merged(sample_input_ids).logits
    del merged

    unmerged = fresh_fp32()
    target_modules = sorted({name.split(".")[-1] for name in state["adapters"]})
    if state["method"] == "oplora":
        unmerged = apply_oplora_to_model(unmerged, target_modules=target_modules, r=state["r"], lora_alpha=state["alpha"], k=state["k"])
    else:
        unmerged = apply_lora_to_model(unmerged, target_modules=target_modules, r=state["r"], lora_alpha=state["alpha"])
    modules = dict(unmerged.named_modules())
    for name, factors in state["adapters"].items():
        modules[name].lora_A.data.copy_(factors["A"])
        modules[name].lora_B.data.copy_(factors["B"])
    unmerged.eval()
    unmerged_logits = unmerged(sample_input_ids).logits

    max_diff = (merged_logits - unmerged_logits).abs().max().item()
    if max_diff > tolerance:
        raise ValueError(f"{condition}: merged and unmerged logits differ by {max_diff:.2e} (tolerance {tolerance})")
    return {"condition": condition, "checked": True, "max_logit_diff": max_diff, "tolerance": tolerance}
