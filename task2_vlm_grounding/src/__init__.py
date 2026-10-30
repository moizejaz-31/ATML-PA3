"""Task 2 modules: feature caching, connector, VLM architecture, curriculum training, evaluation."""
from .connector import VLMConnector
from .vlm_model import SigLIPQwenVLM
from .evaluate import compute_vqa_consensus_score, compute_image_swap_penalty

__all__ = [
    "VLMConnector",
    "SigLIPQwenVLM",
    "compute_vqa_consensus_score",
    "compute_image_swap_penalty",
]
