"""Task 1 modules: data, training, adapters (LoRA/OPLoRA), L2-SP, diagnostics, evaluation."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from .data import prepare_metamath_dataset
from .train import train_task1_model, train_task1_model as train_task1
from .l2sp import L2SPRegularizer
from .lora import LoRALinear
from .oplora import OPLoRALinear
from .diagnostics import compute_subspace_interference, compute_relative_magnitude

__all__ = [
    "prepare_metamath_dataset",
    "train_task1_model",
    "train_task1",
    "L2SPRegularizer",
    "LoRALinear",
    "OPLoRALinear",
    "compute_subspace_interference",
    "compute_relative_magnitude",
]
