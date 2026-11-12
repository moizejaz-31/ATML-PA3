"""
LoRA adapter implementation for Qwen attention and MLP projections.
Targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.
Includes:
- Low-rank linear module W = W0 + (alpha/r) * B * A
- Dynamic module replacement
- Adapter factor export and merging into fresh backbone copies for evaluation
"""
import math
from typing import Dict, List, Optional
import torch
import torch.nn as nn

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Trainable adapter matrices (FP32)
        self.lora_A = nn.Parameter(torch.empty(r, in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=torch.float32))

        # Kaiming uniform initialization for A, zero for B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x)

        base_out = self.base_layer(x)
        # Cast input to adapter dtype (FP32) for accumulation, then cast back to x.dtype
        x_dropped = self.dropout(x.float())
        lora_delta = ((x_dropped @ self.lora_A.T) @ self.lora_B.T) * self.scaling
        return base_out + lora_delta.to(x.dtype)

    def get_effective_factors(self):
        """Return the (A, B) pair that defines delta W."""
        return self.lora_A, self.lora_B

    def get_delta_weight(self) -> torch.Tensor:
        """Return delta W = (alpha/r) * B * A in FP32."""
        return (self.scaling * (self.lora_B @ self.lora_A)).float()

    def merge_weights(self):
        """Merge delta W into base_layer.weight in-place."""
        if not self.merged:
            delta_w = self.get_delta_weight().to(self.base_layer.weight.dtype)
            self.base_layer.weight.data.add_(delta_w)
            self.merged = True

    def unmerge_weights(self):
        """Unmerge delta W from base_layer.weight."""
        if self.merged:
            delta_w = self.get_delta_weight().to(self.base_layer.weight.dtype)
            self.base_layer.weight.data.sub_(delta_w)
            self.merged = False

def replace_target_linears(model: nn.Module, target_modules: List[str], build_adapter) -> int:
    """
    Freeze the backbone and wrap every nn.Linear whose own name is a target.
    Iterates over a snapshot of the module tree so that new wrappers are never revisited.
    """
    for param in model.parameters():
        param.requires_grad = False

    count = 0
    for module in list(model.modules()):
        for child_name, child_module in list(module.named_children()):
            if child_name in target_modules and isinstance(child_module, nn.Linear):
                setattr(module, child_name, build_adapter(child_module))
                count += 1
    return count

def apply_lora_to_model(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    r: int = 16,
    lora_alpha: Optional[float] = None,
    alpha: Optional[float] = None,
    lora_dropout: float = 0.05,
) -> nn.Module:
    """
    Recursively replaces target linear modules with LoRALinear.
    Freezes base backbone weights.
    Returns the adapted model.
    """
    if target_modules is None:
        target_modules = DEFAULT_TARGET_MODULES

    effective_alpha = alpha if alpha is not None else (lora_alpha if lora_alpha is not None else 32.0)

    model.lora_module_count = replace_target_linears(
        model,
        target_modules,
        lambda base: LoRALinear(base_layer=base, r=r, lora_alpha=effective_alpha, lora_dropout=lora_dropout),
    )
    return model

def export_adapter_factors(model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Export FP32 CPU factors for every adapter, keyed by module name.
    A and B are the raw parameters; A_eff and B_eff define delta W (projected for OPLoRA).
    """
    exported = {}
    for name, module in model.named_modules():
        if hasattr(module, "get_effective_factors"):
            A_eff, B_eff = module.get_effective_factors()
            exported[name] = {
                "A": module.lora_A.detach().float().cpu(),
                "B": module.lora_B.detach().float().cpu(),
                "A_eff": A_eff.detach().float().cpu(),
                "B_eff": B_eff.detach().float().cpu(),
                "scaling": float(module.scaling),
            }
    return exported

def delta_from_factors(factors: Dict[str, torch.Tensor]) -> torch.Tensor:
    """delta W = scaling * B_eff * A_eff in FP32."""
    return factors["scaling"] * (factors["B_eff"].float() @ factors["A_eff"].float())

@torch.no_grad()
def merge_factors_into_backbone(model: nn.Module, adapters: Dict[str, Dict[str, torch.Tensor]]) -> int:
    """Add exported adapter corrections to a fresh, unwrapped backbone. Every adapter must find its weight."""
    modules = dict(model.named_modules())
    for name, factors in adapters.items():
        if name not in modules or not isinstance(modules[name], nn.Linear):
            raise KeyError(f"Adapter target {name} is not a linear layer of this backbone")
        weight = modules[name].weight
        # Merge in FP32, then round once to the backbone dtype
        merged = weight.data.float() + delta_from_factors(factors).to(weight.device)
        weight.data.copy_(merged.to(weight.dtype))
    return len(adapters)
