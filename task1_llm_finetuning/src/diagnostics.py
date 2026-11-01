"""
Weight-space and activation diagnostics for Task 1.
Layers examined: down_proj in layers 0, 13, 27 (zero-based).
Checkpoints: Full FT Midpoint (625) and all 5 Final models (1250). All computed in FP32.

Metrics:
1. Subspace-interference ratio:
   rho_k = ||Uk Uk^T Delta W||_F^2 / ||Delta W||_F^2
2. Relative update magnitude:
   eta = ||Delta W||_F / ||W0||_F
3. Local activation correction ratio:
   R(z) = ||Delta W z||_2 / (||W0 z||_2 + 10^-8)
   where z is the starting model's input activation to down_proj at the last non-padding prompt token.
"""
import json
import os
from typing import Dict, List, Tuple, Any
import torch
import torch.nn as nn

from .conditions import CONDITIONS, load_evaluation_state
from .data import build_prompt, load_gsm8k_test, load_hellaswag_validation
from .lora import delta_from_factors

TARGET_LAYERS = [0, 13, 27]
PROTECTION_RANK = 128

def compute_subspace_interference(Uk: torch.Tensor, delta_W: torch.Tensor) -> float:
    """
    rho_k = ||Uk Uk^T delta_W||_F^2 / ||delta_W||_F^2
    Uk: (dout, k), delta_W: (dout, din)
    """
    Uk = Uk.float()
    delta_W = delta_W.float()
    proj = Uk @ (Uk.T @ delta_W)
    num = torch.sum(proj ** 2).item()
    denom = torch.sum(delta_W ** 2).item() + 1e-12
    return float(num / denom)

def compute_relative_magnitude(W0: torch.Tensor, delta_W: torch.Tensor) -> float:
    """
    eta = ||delta_W||_F / ||W0||_F
    W0: (dout, din), delta_W: (dout, din)
    """
    delta_norm = torch.linalg.norm(delta_W.float(), ord="fro").item()
    base_norm = torch.linalg.norm(W0.float(), ord="fro").item() + 1e-12
    return float(delta_norm / base_norm)

def compute_activation_ratios(W0: torch.Tensor, delta_W: torch.Tensor, z: torch.Tensor) -> List[float]:
    """
    R(z) = ||delta_W z||_2 / (||W0 z||_2 + 10^-8) for each row of z: (batch, din).
    """
    num = torch.linalg.norm(z.float() @ delta_W.float().T, dim=-1)
    denom = torch.linalg.norm(z.float() @ W0.float().T, dim=-1) + 1e-8
    return (num / denom).cpu().tolist()

class ActivationProbeHook:
    """
    Forward hook to capture input activation z to down_proj at the last token.
    Prompts run one at a time without padding, so the last token is the last non-padding prompt token.
    """
    def __init__(self, target_layer_idx: int):
        self.target_layer_idx = target_layer_idx
        self.cached_activations: List[torch.Tensor] = []
        self.hook_handle = None

    def hook_fn(self, module, input, output):
        # input is a tuple (x,) where x is (batch, seq_len, din)
        self.cached_activations.append(input[0][:, -1, :].detach().float().cpu())

    def register(self, module: nn.Module):
        self.hook_handle = module.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()

    def clear(self):
        self.cached_activations.clear()

def extract_down_proj_layers(model: nn.Module) -> Dict[int, nn.Module]:
    """
    Extract down_proj modules from Qwen model transformer layers.
    Qwen architecture: model.model.layers[i].mlp.down_proj
    """
    down_projs = {}
    layers = model.model.layers
    for idx in TARGET_LAYERS:
        if idx < len(layers):
            down_projs[idx] = layers[idx].mlp.down_proj
    return down_projs

def build_probe_prompts(tokenizer, probe_path: str) -> List[Tuple[str, int, str]]:
    """Probe prompts in their benchmark formats: chat GSM8K, stock completion HellaSwag."""
    from lm_eval.tasks.hellaswag.utils import process_docs

    with open(probe_path, "r", encoding="utf-8") as f:
        probe = json.load(f)

    prompts = []
    gsm_test = load_gsm8k_test()
    for idx in probe["gsm8k_indices"]:
        prompts.append(("math", idx, build_prompt(tokenizer, f"Q: {gsm_test[idx]['question']}\nA:")))
    hella_docs = process_docs(load_hellaswag_validation())
    for idx in probe["hellaswag_indices"]:
        prompts.append(("non_math", idx, hella_docs[idx]["query"]))
    return prompts

@torch.no_grad()
def cache_probe_activations(model: nn.Module, tokenizer, prompts: List[Tuple[str, int, str]], device: torch.device) -> Dict[int, torch.Tensor]:
    """Run the starting model over the probe set and return {layer: (n_prompts, din)}."""
    hooks = {}
    for layer, module in extract_down_proj_layers(model).items():
        hooks[layer] = ActivationProbeHook(layer)
        hooks[layer].register(module)
    for _, _, text in prompts:
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        model(**enc)
    cached = {layer: torch.cat(hook.cached_activations, dim=0) for layer, hook in hooks.items()}
    for hook in hooks.values():
        hook.remove()
    return cached

def layer_delta(state: Dict[str, Any], layer: int, W0: torch.Tensor) -> torch.Tensor:
    """Local update of down_proj: W_trained - W0 for Full FT, projected factors for OPLoRA."""
    name = f"model.layers.{layer}.mlp.down_proj"
    if state["kind"] == "full":
        return state["state_dict"][f"{name}.weight"].float() - W0
    return delta_from_factors(state["adapters"][name])

def run_diagnostics(base_model: nn.Module, tokenizer, prompts, checkpoint_dir: str, device: torch.device) -> Tuple[List[dict], List[dict]]:
    activations = cache_probe_activations(base_model, tokenizer, prompts, device)
    starting_weights = {layer: m.weight.detach().float().cpu() for layer, m in extract_down_proj_layers(base_model).items()}

    bases = {}
    for layer, W0 in starting_weights.items():
        U, _, Vh = torch.linalg.svd(W0, full_matrices=False)
        k = min(PROTECTION_RANK, U.shape[1])
        bases[layer] = (U[:, :k], Vh[:k, :].T)

    weight_rows, activation_rows = [], []
    for condition in CONDITIONS:
        if condition == "baseline":
            continue
        state = load_evaluation_state(condition, checkpoint_dir)
        for layer, W0 in starting_weights.items():
            delta = layer_delta(state, layer, W0)
            Uk, Vk = bases[layer]
            weight_rows.append({
                "condition": condition,
                "layer": layer,
                "rho_k": compute_subspace_interference(Uk, delta),
                "rho_k_input": compute_subspace_interference(Vk, delta.T),
                "eta": compute_relative_magnitude(W0, delta),
            })
            ratios = compute_activation_ratios(W0, delta, activations[layer])
            for (domain, example_id, _), ratio in zip(prompts, ratios):
                activation_rows.append({"condition": condition, "layer": layer, "domain": domain, "example_id": example_id, "R": ratio})
    return weight_rows, activation_rows


if __name__ == "__main__":
    import argparse
    import pandas as pd
    from transformers import AutoTokenizer, AutoModelForCausalLM

    parser = argparse.ArgumentParser(description="Compute Task 1 weight and activation diagnostics")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--checkpoint_dir", type=str, default="./task1_llm_finetuning/checkpoints")
    parser.add_argument("--output_dir", type=str, default="./task1_llm_finetuning/results")
    parser.add_argument("--probe_path", type=str, default="./task1_llm_finetuning/splits/probe_set_seed6304.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Diagnostics CLI] Loading base model: {args.base_model}...")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    prompts = build_probe_prompts(tokenizer, args.probe_path)
    weight_rows, activation_rows = run_diagnostics(base_model, tokenizer, prompts, args.checkpoint_dir, device)

    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(weight_rows).to_csv(os.path.join(args.output_dir, "weight_diagnostics.csv"), index=False)
    pd.DataFrame(activation_rows).to_csv(os.path.join(args.output_dir, "activation_diagnostics.csv"), index=False)
    print(f"[Diagnostics CLI] Saved {len(weight_rows)} weight rows and {len(activation_rows)} activation rows to {args.output_dir}")
