"""
L2-SP regularization implementation for Task 1.
Reference: Li et al., Explicit Inductive Bias for Transfer Learning with Convolutional Networks.

Formula:
  Omega(theta, theta_0) = 0.5 * sum_{j=1}^P (theta_j - theta_{0,j})^2
  Total Loss = L_SFT(theta) + lambda * Omega(theta, theta_0)
  lambda = 10^-3

Key Requirements:
- Reference parameters theta_0 are frozen in memory or disk.
- Sum over all unique parameters (including biases and normalization weights), counting tied parameters once.
- Do not average Omega over parameters.
- Record Omega, lambda * Omega, total loss, and parameter displacement ||theta - theta_0||_2.
"""
from typing import Dict, Set, Tuple
import torch
import torch.nn as nn

class L2SPRegularizer:
    def __init__(self, reference_model: nn.Module, lambda_reg: float = 1e-3, device: str = "cpu"):
        """
        Store frozen reference parameter snapshot theta_0 on CPU or target device.
        """
        self.lambda_reg = lambda_reg
        self.reference_params: Dict[str, torch.Tensor] = {}
        seen_ptrs: Set[int] = set()

        for name, param in reference_model.named_parameters():
            # Track unique parameters to avoid duplicate regularization on tied weights
            ptr = param.data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                self.reference_params[name] = param.detach().clone().to(device)

    def compute_penalty(self, current_model: nn.Module) -> Tuple[torch.Tensor, float, float]:
        """
        Compute un-averaged sum of squared parameter departures:
        Omega = 0.5 * sum (theta - theta_0)^2
        Returns:
          penalty_tensor (with autograd for backprop),
          raw_omega (float),
          scaled_penalty (lambda * Omega, float)
        """
        first_param = next(current_model.parameters())
        penalty = torch.tensor(0.0, device=first_param.device, dtype=torch.float32)
        seen_ptrs: Set[int] = set()

        for name, param in current_model.named_parameters():
            ptr = param.data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)

            if name in self.reference_params:
                ref = self.reference_params[name].to(param.device, dtype=torch.float32)
                cur = param.float()
                diff = cur - ref
                penalty = penalty + 0.5 * torch.sum(diff ** 2)

        raw_omega = penalty.item()
        scaled_penalty = self.lambda_reg * raw_omega
        return penalty * self.lambda_reg, raw_omega, scaled_penalty

    def compute_displacement(self, current_model: nn.Module) -> float:
        """
        Compute Euclidean norm of parameter displacement: ||theta - theta_0||_2.
        """
        sum_sq = 0.0
        seen_ptrs: Set[int] = set()

        for name, param in current_model.named_parameters():
            ptr = param.data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)

            if name in self.reference_params:
                ref = self.reference_params[name].to(param.device, dtype=torch.float32)
                diff = param.detach().float() - ref
                sum_sq += torch.sum(diff ** 2).item()

        return float(sum_sq ** 0.5)
