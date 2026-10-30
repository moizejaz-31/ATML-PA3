"""
Orthogonal Projection LoRA (OPLoRA) implementation for Task 1.
Reference: Xiong and Xie, Sections 3.2-3.3 & Proposition 2.

Mathematical Formulation:
For starting weight W0 = U Sigma V^T, compute leading k=128 singular vectors:
  Uk in R^{dout x k}, Vk in R^{din x k} in FP32.
Adapter parameters:
  A in R^{r x din}, B in R^{dout x r}
Effective projected factors:
  B_tilde = B - Uk (Uk^T B)
  A_tilde = A - (A Vk) Vk^T
Delta W = (alpha / r) * B_tilde * A_tilde

Features:
- Effective factor projection (avoids dense d x d projection matrices)
- Merging and unmerging into base weights for zero-shot evaluation
- Exporting projected factors for inference and diagnostics
"""
import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn

class OPLoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        lora_alpha: float = 32.0,
        k: int = 128,
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
        self.k = k

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Compute SVD of base weight W0 in FP32
        W0_fp32 = base_layer.weight.data.float()
        # full_matrices=False gives U: (dout, min(dout, din)), Vh: (min(dout, din), din)
        U, S, Vh = torch.linalg.svd(W0_fp32, full_matrices=False)
        actual_k = min(k, U.shape[1], Vh.shape[0])
        Uk = U[:, :actual_k].contiguous()  # (dout, k)
        Vk = Vh[:actual_k, :].T.contiguous()  # (din, k)

        self.register_buffer("Uk", Uk)
        self.register_buffer("Vk", Vk)

        # Unprojected adapter parameters (FP32)
        self.lora_A = nn.Parameter(torch.empty(r, in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.merged = False

    def get_projected_factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute effective projected factors:
        B_tilde = B - Uk (Uk^T B)
        A_tilde = A - (A Vk) Vk^T
        """
        Uk = self.Uk.to(self.lora_B.device, dtype=torch.float32)
        Vk = self.Vk.to(self.lora_A.device, dtype=torch.float32)

        # Uk @ (Uk.T @ B)
        B_proj = Uk @ (Uk.T @ self.lora_B)
        B_tilde = self.lora_B - B_proj

        # (A @ Vk) @ Vk.T
        A_proj = (self.lora_A @ Vk) @ Vk.T
        A_tilde = self.lora_A - A_proj

        return A_tilde, B_tilde

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x)

        base_out = self.base_layer(x)
        A_tilde, B_tilde = self.get_projected_factors()

        x_dropped = self.dropout(x.float())
        lora_delta = (x_dropped @ A_tilde.T @ B_tilde.T) * self.scaling
        return base_out + lora_delta.to(x.dtype)

    def get_delta_weight(self) -> torch.Tensor:
        """Compute delta W = (alpha / r) * B_tilde * A_tilde in FP32."""
        A_tilde, B_tilde = self.get_projected_factors()
        return (self.scaling * (B_tilde @ A_tilde)).float()

    def merge_weights(self):
        """Merge projected delta W into base_layer.weight in-place."""
        if not self.merged:
            delta_w = self.get_delta_weight().to(self.base_layer.weight.dtype)
            self.base_layer.weight.data.add_(delta_w)
            self.merged = True

    def unmerge_weights(self):
        """Unmerge projected delta W from base_layer.weight."""
        if self.merged:
            delta_w = self.get_delta_weight().to(self.base_layer.weight.dtype)
            self.base_layer.weight.data.sub_(delta_w)
            self.merged = False

def apply_oplora_to_model(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    r: int = 16,
    lora_alpha: float = 32.0,
    k: int = 128,
    lora_dropout: float = 0.05,
) -> int:
    """
    Recursively replaces target linear modules with OPLoRALinear.
    Freezes base backbone weights.
    Returns count of modules adapted.
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    count = 0
    for name, module in model.named_modules():
        for child_name, child_module in list(module.named_children()):
            if any(child_name == t or name.endswith(t) for t in target_modules):
                if isinstance(child_module, nn.Linear):
                    oplora_layer = OPLoRALinear(
                        base_layer=child_module,
                        r=r,
                        lora_alpha=lora_alpha,
                        k=k,
                        lora_dropout=lora_dropout,
                    )
                    setattr(module, child_name, oplora_layer)
                    count += 1
    return count
