"""
Hybrid Vision-Language Model: SigLIP connector + Qwen3-0.6B language backbone.
Prefixes continuous visual tokens Z(I) into Qwen's input embedding space.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .connector import VLMConnector

class SigLIPQwenVLM(nn.Module):
    def __init__(self, llm_backbone: nn.Module, connector: VLMConnector):
        super().__init__()
        self.llm = llm_backbone
        self.connector = connector

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Any:
        """
        Forward pass with continuous visual token prefix prepended to user turn text embeddings.
        - visual_features: (batch_size, num_visual_tokens, 768)
        - input_ids: (batch_size, seq_len)
        """
        if visual_features is not None:
            batch_size = visual_features.shape[0]
            # 1. Project visual features to language embedding space: (B, N_v, 1024)
            visual_tokens = self.connector(visual_features.to(self.connector.mlp[0].weight.dtype))
            n_visual = visual_tokens.shape[1]

            # 2. Get text embeddings: (B, seq_len, 1024)
            text_embeds = self.llm.get_input_embeddings()(input_ids)

            # 3. Concatenate visual tokens and text embeddings: (B, N_v + seq_len, 1024)
            inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

            # 4. Expand attention mask: (B, N_v + seq_len)
            visual_mask = torch.ones((batch_size, n_visual), dtype=attention_mask.dtype, device=attention_mask.device)
            new_attention_mask = torch.cat([visual_mask, attention_mask], dim=1)

            # 5. Expand labels with -100 for visual tokens if labels provided
            new_labels = None
            if labels is not None:
                visual_labels = torch.full((batch_size, n_visual), -100, dtype=labels.dtype, device=labels.device)
                new_labels = torch.cat([visual_labels, labels], dim=1)

            return self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=new_attention_mask,
                labels=new_labels,
                **kwargs
            )
        else:
            # No-image condition: direct text pass
            return self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )

    @torch.no_grad()
    def generate_answer(
        self,
        input_ids: torch.Tensor,
        visual_features: Optional[torch.Tensor] = None,
        max_new_tokens: int = 16,
        tokenizer = None,
    ) -> str:
        """
        Greedy autoregressive generation for VQA answers.
        """
        self.eval()
        device = input_ids.device
        if visual_features is not None:
            visual_tokens = self.connector(visual_features.to(self.connector.mlp[0].weight.dtype))
            text_embeds = self.llm.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
            attention_mask = torch.ones((1, inputs_embeds.shape[1]), dtype=torch.long, device=device)
            
            out = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id if tokenizer else None,
                eos_token_id=tokenizer.eos_token_id if tokenizer else None,
            )
            gen_tokens = out[0]
        else:
            attention_mask = torch.ones_like(input_ids)
            out = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id if tokenizer else None,
                eos_token_id=tokenizer.eos_token_id if tokenizer else None,
            )
            gen_tokens = out[0][input_ids.shape[1]:]

        if tokenizer is not None:
            return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return ""
