"""
Two-layer MLP connector for mapping SigLIP patch features to Qwen embedding space.
Linear 768 -> 1024, GELU, Linear 1024 -> 1024.
"""
import torch
import torch.nn as nn

class VLMConnector(nn.Module):
    def __init__(self, in_dim: int = 768, hidden_dim: int = 1024, out_dim: int = 1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def calibrate_scale(self, sample_visual_features: torch.Tensor, text_embedding_table: torch.Tensor):
        """
        Calibrate connector output magnitude against text embedding magnitude.
        Scale factor = mean(||E_text||_2) / mean(||connector(F)||_2)
        """
        with torch.no_grad():
            projected = self.mlp(sample_visual_features.float())
            proj_norm = torch.linalg.norm(projected, dim=-1).mean().item()
            text_norm = torch.linalg.norm(text_embedding_table.float(), dim=-1).mean().item()
            scale_val = text_norm / max(proj_norm, 1e-8)
            self.scale.copy_(torch.tensor(scale_val))
            print(f"[Connector Calibration] proj_norm={proj_norm:.4f}, text_norm={text_norm:.4f}, scale={scale_val:.4f}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x) * self.scale
