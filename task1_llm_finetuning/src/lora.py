"""
LoRA adapter implementation for Qwen attention and MLP projections.
Targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.
Includes:
- Low-rank linear module W = W0 + (alpha/r) * B * A
- Dynamic module replacement
- Clean weight merging into base weights for evaluation
"""
import math
from typing import List, Optional
import torch
import torch.nn as nn

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
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    effective_alpha = alpha if alpha is not None else (lora_alpha if lora_alpha is not None else 32.0)

    # Freeze all backbone parameters
    for param in model.parameters():
        param.requires_grad = False

    count = 0
    for name, module in model.named_modules():
        for child_name, child_module in list(module.named_children()):
            if any(child_name == t or name.endswith(t) for t in target_modules):
                if isinstance(child_module, nn.Linear):
                    lora_layer = LoRALinear(
                        base_layer=child_module,
                        r=r,
                        lora_alpha=effective_alpha,
                        lora_dropout=lora_dropout,
                    )
                    setattr(module, child_name, lora_layer)
                    count += 1
    model.lora_module_count = count
    return model
