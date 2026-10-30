"""
Weight-space and activation diagnostics for Task 1.
Layers examined: down_proj in layers 0, 13, 27 (zero-based).
Checkpoints: Baseline, Full FT Midpoint (625), and all 5 Final models (1250).

Metrics:
1. Subspace-interference ratio:
   rho_k = ||Uk Uk^T Delta W||_F^2 / ||Delta W||_F^2
2. Relative update magnitude:
   eta = ||Delta W||_F / ||W0||_F
3. Local activation correction ratio:
   R(z) = ||Delta W z||_2 / (||W0 z||_2 + 10^-8)
   where z is input activation to down_proj at last prompt token.
"""
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import numpy as np

TARGET_LAYERS = [0, 13, 27]

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

def compute_activation_ratio(W0: torch.Tensor, delta_W: torch.Tensor, z: torch.Tensor) -> float:
    """
    R(z) = ||delta_W z||_2 / (||W0 z||_2 + 10^-8)
    z: (din,) or (batch, din) input activation at last prompt token.
    """
    W0 = W0.float()
    delta_W = delta_W.float()
    z = z.float()
    if z.dim() == 1:
        num = torch.linalg.norm(delta_W @ z).item()
        denom = torch.linalg.norm(W0 @ z).item() + 1e-8
        return float(num / denom)
    else:
        # Batch mode: z is (batch, din)
        # delta_W @ z.T -> (dout, batch)
        num = torch.linalg.norm(z @ delta_W.T, dim=-1)  # (batch,)
        denom = torch.linalg.norm(z @ W0.T, dim=-1) + 1e-8
        ratios = (num / denom).cpu().numpy().tolist()
        return float(np.mean(ratios))

class ActivationProbeHook:
    """
    Forward hook to capture input activation z to down_proj at last prompt token.
    """
    def __init__(self, target_layer_idx: int):
        self.target_layer_idx = target_layer_idx
        self.cached_activations: List[torch.Tensor] = []
        self.hook_handle = None

    def hook_fn(self, module, input, output):
        # input is a tuple (x,) where x is (batch, seq_len, din)
        x = input[0].detach()
        self.cached_activations.append(x)

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
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is not None:
        for idx in TARGET_LAYERS:
            if idx < len(layers):
                mlp = getattr(layers[idx], "mlp", None)
                if mlp is not None and hasattr(mlp, "down_proj"):
                    down_projs[idx] = mlp.down_proj
    return down_projs


if __name__ == "__main__":
    import argparse
    import json
    import pandas as pd
    from transformers import AutoTokenizer, AutoModelForCausalLM

    parser = argparse.ArgumentParser(description="Compute Task 1 weight and activation diagnostics")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--output_dir", type=str, default="./task1_llm_finetuning/results")
    args = parser.parse_args()

    print(f"[Diagnostics CLI] Loading base model: {args.base_model}...")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32)

    os.makedirs(args.output_dir, exist_ok=True)
    print("[Diagnostics CLI] Diagnostics utility initialized.")
