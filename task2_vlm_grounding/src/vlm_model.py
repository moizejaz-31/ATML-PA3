"""
Hybrid Vision-Language Model: SigLIP connector + Qwen3-0.6B language backbone.
"""
import torch
import torch.nn as nn
from .connector import VLMConnector

class SigLIPQwenVLM(nn.Module):
    def __init__(self, llm_backbone, connector: VLMConnector):
        super().__init__()
        self.llm = llm_backbone
        self.connector = connector

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: torch.Tensor = None,
        labels: torch.Tensor = None,
        position_ids: torch.Tensor = None,
    ):
        """
        Forward pass with optional continuous visual token prefix in user turn.
        """
        if visual_features is not None:
            # Map visual features through connector
            visual_tokens = self.connector(visual_features)
            # Embed text tokens and concatenate
            text_embeds = self.llm.get_input_embeddings()(input_ids)
            # Insert visual tokens before question in user turn
            # Pass inputs_embeds to llm
            pass
        return self.llm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
